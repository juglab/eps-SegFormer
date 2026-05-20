from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from models_vit import ConvDecoder, NeighborhoodConcatClassifier, NeighborhoodConcatClassifierV2
except ModuleNotFoundError:
    from vit.models_vit import ConvDecoder, NeighborhoodConcatClassifier, NeighborhoodConcatClassifierV2


@dataclass
class MuViTV2Output:
    reconstruction: torch.Tensor
    token_logits: torch.Tensor
    center_logits: torch.Tensor
    visible_mask: torch.Tensor
    finest_tokens: torch.Tensor


def token_centers_from_bbox(bbox: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    if bbox.ndim != 4 or bbox.shape[-2:] != (2, 2):
        raise ValueError(f"Expected bbox with shape [B, L, 2, 2], got {tuple(bbox.shape)}.")
    mins = bbox[:, :, 0]
    maxs = bbox[:, :, 1]
    spans = maxs - mins

    ys = (torch.arange(grid_h, device=bbox.device, dtype=bbox.dtype) + 0.5) / grid_h
    xs = (torch.arange(grid_w, device=bbox.device, dtype=bbox.dtype) + 0.5) / grid_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    local = torch.stack([yy, xx], dim=-1).reshape(1, 1, grid_h * grid_w, 2)
    return mins[:, :, None, :] + local * spans[:, :, None, :]


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RoPEAttention2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        coord_scale: float = 25.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE.")
        if coord_scale <= 0:
            raise ValueError("coord_scale must be positive.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.axis_dim = head_dim // 2
        self.coord_scale = float(coord_scale)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        inv_freq = rope_base ** (
            -torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        y_part, x_part = x[..., : self.axis_dim], x[..., self.axis_dim :]
        norm_coords = coords.to(dtype=x.dtype) / self.coord_scale
        y_angles = norm_coords[:, None, :, 0, None] * self.inv_freq.to(dtype=x.dtype).view(1, 1, 1, -1)
        x_angles = norm_coords[:, None, :, 1, None] * self.inv_freq.to(dtype=x.dtype).view(1, 1, 1, -1)
        y_cos = y_angles.cos().repeat_interleave(2, dim=-1)
        y_sin = y_angles.sin().repeat_interleave(2, dim=-1)
        x_cos = x_angles.cos().repeat_interleave(2, dim=-1)
        x_sin = x_angles.sin().repeat_interleave(2, dim=-1)
        y_rot = y_part * y_cos + _rotate_pairs(y_part) * y_sin
        x_rot = x_part * x_cos + _rotate_pairs(x_part) * x_sin
        return torch.cat((y_rot, x_rot), dim=-1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        if coords.shape != (batch_size, num_tokens, 2):
            raise ValueError(f"Expected coords with shape [{batch_size}, {num_tokens}, 2], got {tuple(coords.shape)}.")

        qkv = self.qkv(x).view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = self._apply_rope(q, coords)
        k = self._apply_rope(k, coords)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        return self.proj_dropout(self.proj(attn))


class RoPETransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        coord_scale: float = 25.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention2D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope_base=rope_base,
            dropout=dropout,
            coord_scale=coord_scale,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens), coords)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class WildWestMuViTV2(nn.Module):
    def __init__(
        self,
        image_size: int = 25,
        patch_size: int = 5,
        in_channels: int = 1,
        embed_dim: int = 192,
        depth: int = 14,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        decoder_channels: tuple[int, ...] = (256, 128, 64),
        num_classes: int = 4,
        num_levels: int = 2,
        rope_base: float = 10000.0,
        segmentation_head: str = "linear",
        classifier_context_kernel_size: int = 1,
        classifier_hidden_dim: int | None = None,
        masking_mode: str = "token",
    ) -> None:
        super().__init__()
        if image_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be positive and divisible by patch_size.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE.")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")
        if masking_mode not in ("token", "drop"):
            raise ValueError("masking_mode must be one of 'token' or 'drop'.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.rope_base = rope_base
        self.segmentation_head = segmentation_head
        self.classifier_context_kernel_size = classifier_context_kernel_size
        self.classifier_hidden_dim = classifier_hidden_dim
        self.masking_mode = masking_mode
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.level_embed = nn.Embedding(num_levels, embed_dim)
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                RoPETransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    rope_base=rope_base,
                    coord_scale=float(image_size),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = ConvDecoder(
            in_channels=embed_dim,
            out_channels=in_channels,
            upsample_factor=patch_size,
            hidden_channels=decoder_channels,
        )
        if segmentation_head == "linear":
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif segmentation_head == "neighbor_concat":
            self.classifier = NeighborhoodConcatClassifier(
                embed_dim=embed_dim,
                num_classes=num_classes,
                kernel_size=classifier_context_kernel_size,
                hidden_dim=classifier_hidden_dim,
            )
        elif segmentation_head == "neighbor_concat_v2":
            self.classifier = NeighborhoodConcatClassifierV2(
                embed_dim=embed_dim,
                num_classes=num_classes,
                kernel_size=classifier_context_kernel_size,
                hidden_dim=classifier_hidden_dim,
            )
        else:
            raise ValueError(
                f"Unsupported segmentation_head '{segmentation_head}'. "
                "Use 'linear', 'neighbor_concat', or 'neighbor_concat_v2'."
            )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.level_embed.weight, std=0.02)

    def _apply_random_mask(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
        protect_center: torch.Tensor | None = None,
        force_mask_center: torch.Tensor | None = None,
        random_mask_exclusion_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in the range [0.0, 1.0).")

        batch_size, num_tokens, _ = tokens.shape
        visible_mask = torch.ones(batch_size, num_tokens, device=tokens.device, dtype=torch.bool)
        masked = tokens.clone()
        if protect_center is not None:
            if protect_center.shape != (batch_size,):
                raise ValueError(f"Expected protect_center with shape [{batch_size}], got {tuple(protect_center.shape)}.")
            protect_center = protect_center.to(device=tokens.device, dtype=torch.bool)

        center_index = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
        if force_mask_center is not None:
            if force_mask_center.shape != (batch_size,):
                raise ValueError(
                    f"Expected force_mask_center with shape [{batch_size}], got {tuple(force_mask_center.shape)}."
                )
            force_mask_center = force_mask_center.to(device=tokens.device, dtype=torch.bool)
            masked[force_mask_center, center_index] = self.mask_token[0, 0]
            visible_mask[force_mask_center, center_index] = False

        if random_mask_exclusion_mask is not None:
            if random_mask_exclusion_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    "Expected random_mask_exclusion_mask with shape "
                    f"[{batch_size}, {num_tokens}], got {tuple(random_mask_exclusion_mask.shape)}."
                )
            random_mask_exclusion_mask = random_mask_exclusion_mask.to(device=tokens.device, dtype=torch.bool)

        num_masked_per_level = int(self.num_patches * mask_ratio)
        if num_masked_per_level <= 0:
            return masked, visible_mask

        noise = torch.rand(batch_size, num_tokens, device=tokens.device)
        unavailable = ~visible_mask
        if protect_center is not None:
            unavailable[:, center_index] = protect_center
        if random_mask_exclusion_mask is not None:
            unavailable |= random_mask_exclusion_mask
        noise[unavailable] = float("inf")
        for level_index in range(self.num_levels):
            start = level_index * self.num_patches
            end = start + self.num_patches
            level_noise = noise[:, start:end]
            level_unavailable = unavailable[:, start:end]
            level_mask_indices = level_noise.argsort(dim=1)
            for batch_index in range(batch_size):
                available_count = int((~level_unavailable[batch_index]).sum().item())
                random_mask_count = min(num_masked_per_level, available_count)
                if random_mask_count <= 0:
                    continue
                selected = level_mask_indices[batch_index, :random_mask_count] + start
                masked[batch_index, selected] = self.mask_token[0, 0]
                visible_mask[batch_index, selected] = False
        return masked, visible_mask

    def _encode_tokens(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.masking_mode == "token":
            encoded = self.pos_dropout(tokens)
            for block in self.blocks:
                encoded = block(encoded, coords)
            return self.norm(encoded)
        if self.masking_mode != "drop":
            raise ValueError("masking_mode must be one of 'token' or 'drop'.")

        batch_size, num_tokens, embed_dim = tokens.shape
        encoded_full = self.mask_token.expand(batch_size, num_tokens, -1).clone()
        for batch_index in range(batch_size):
            sample_visible = visible_mask[batch_index]
            if not sample_visible.any():
                raise ValueError("drop masking requires at least one visible token per sample.")
            sample_tokens = self.pos_dropout(tokens[batch_index : batch_index + 1, sample_visible])
            sample_coords = coords[batch_index : batch_index + 1, sample_visible]
            for block in self.blocks:
                sample_tokens = block(sample_tokens, sample_coords)
            sample_tokens = self.norm(sample_tokens)
            encoded_full[batch_index, sample_visible] = sample_tokens[0]
        return encoded_full

    def classify_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.classifier, (NeighborhoodConcatClassifier, NeighborhoodConcatClassifierV2)):
            return self.classifier(tokens, grid_size=self.grid_size)
        return self.classifier(tokens)

    def forward_with_aux(
        self,
        imgs: torch.Tensor,
        bbox: torch.Tensor,
        mask_ratio: float = 0.0,
        protect_center: torch.Tensor | None = None,
        force_mask_center: torch.Tensor | None = None,
        random_mask_exclusion_mask: torch.Tensor | None = None,
    ) -> MuViTV2Output:
        if imgs.ndim != 5:
            raise ValueError(f"Expected imgs with shape [B, L, C, H, W], got {tuple(imgs.shape)}.")
        batch_size, num_levels, channels, height, width = imgs.shape
        if num_levels != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} levels, got {num_levels}.")
        if channels != self.in_channels or height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected imgs with C={self.in_channels}, H=W={self.image_size}, got {tuple(imgs.shape)}."
            )
        if bbox.shape != (batch_size, num_levels, 2, 2):
            raise ValueError(f"Expected bbox with shape [{batch_size}, {num_levels}, 2, 2], got {tuple(bbox.shape)}.")

        tokens = self.patch_embed(imgs.reshape(batch_size * num_levels, channels, height, width))
        tokens = tokens.flatten(2).transpose(1, 2).reshape(
            batch_size,
            num_levels,
            self.num_patches,
            self.embed_dim,
        )
        level_ids = torch.arange(num_levels, device=imgs.device)
        tokens = tokens + self.level_embed(level_ids).view(1, num_levels, 1, self.embed_dim)
        coords = token_centers_from_bbox(bbox.to(dtype=imgs.dtype), self.grid_size, self.grid_size)

        flat_tokens = tokens.reshape(batch_size, num_levels * self.num_patches, self.embed_dim)
        flat_coords = coords.reshape(batch_size, num_levels * self.num_patches, 2)
        flat_tokens, visible_mask = self._apply_random_mask(
            flat_tokens,
            mask_ratio=mask_ratio,
            protect_center=protect_center,
            force_mask_center=force_mask_center,
            random_mask_exclusion_mask=random_mask_exclusion_mask,
        )
        flat_tokens = self._encode_tokens(flat_tokens, flat_coords, visible_mask)

        level_tokens = flat_tokens.reshape(batch_size, num_levels, self.num_patches, self.embed_dim)
        finest_tokens = level_tokens[:, 0]
        feature_maps = finest_tokens.transpose(1, 2).reshape(
            batch_size,
            self.embed_dim,
            self.grid_size,
            self.grid_size,
        )
        reconstruction = self.decoder(feature_maps)
        center_index = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
        token_logits = self.classify_tokens(finest_tokens)
        center_logits = token_logits[:, center_index]
        return MuViTV2Output(
            reconstruction=reconstruction,
            token_logits=token_logits,
            center_logits=center_logits,
            visible_mask=visible_mask,
            finest_tokens=finest_tokens,
        )

    def forward(self, imgs: torch.Tensor, bbox: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        return self.forward_with_aux(imgs, bbox=bbox, mask_ratio=mask_ratio).reconstruction
