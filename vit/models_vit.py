from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ViTEncoderOutput:
    tokens: torch.Tensor
    feature_map: torch.Tensor
    pooled: torch.Tensor
    visible_mask: torch.Tensor


@dataclass
class ViTAutoencoderAuxOutput:
    reconstruction: torch.Tensor
    token_logits: torch.Tensor
    visible_mask: torch.Tensor
    encoded: ViTEncoderOutput


class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        hidden_channels: tuple[int, ...] = (256, 128, 64),
    ) -> None:
        super().__init__()
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")

        blocks: list[nn.Module] = []
        current_channels = in_channels
        remaining_scale = upsample_factor

        for hidden in hidden_channels:
            if remaining_scale <= 1:
                break
            step_scale = 2 if remaining_scale % 2 == 0 else remaining_scale
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels,
                        hidden,
                        kernel_size=step_scale,
                        stride=step_scale,
                    ),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                ]
            )
            current_channels = hidden
            remaining_scale //= step_scale

        if remaining_scale != 1:
            raise ValueError(
                f"Decoder could not resolve upsample factor {upsample_factor}."
            )

        blocks.extend(
            [
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(current_channels, out_channels, kernel_size=1),
            ]
        )
        self.decoder = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class NeighborhoodConcatClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        kernel_size: int = 3,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.local_norm = nn.LayerNorm(embed_dim)
        fusion_dim = hidden_dim or max(embed_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * kernel_size * kernel_size, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, num_classes),
        )

    def _extract_local_tokens(
        self,
        tokens: torch.Tensor,
        grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, embed_dim = tokens.shape
        if num_tokens != grid_size * grid_size:
            raise ValueError(
                f"Expected {grid_size * grid_size} tokens for a {grid_size}x{grid_size} grid, got {num_tokens}."
            )

        feature_map = tokens.transpose(1, 2).reshape(batch_size, embed_dim, grid_size, grid_size)
        local_tokens = F.unfold(feature_map, kernel_size=self.kernel_size, padding=self.padding)
        local_tokens = local_tokens.view(
            batch_size,
            embed_dim,
            self.kernel_size * self.kernel_size,
            num_tokens,
        ).permute(0, 3, 2, 1)

        valid_mask = torch.ones(
            batch_size,
            1,
            grid_size,
            grid_size,
            device=tokens.device,
            dtype=feature_map.dtype,
        )
        local_mask = F.unfold(valid_mask, kernel_size=self.kernel_size, padding=self.padding)
        local_mask = local_mask.view(batch_size, 1, self.kernel_size * self.kernel_size, num_tokens)
        local_mask = local_mask.permute(0, 3, 2, 1)

        local_tokens = self.local_norm(local_tokens)
        local_tokens = local_tokens * local_mask
        return local_tokens, local_mask

    def forward(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = tokens.shape
        local_tokens, _ = self._extract_local_tokens(tokens, grid_size=grid_size)
        flattened = local_tokens.reshape(batch_size, num_tokens, self.kernel_size * self.kernel_size * embed_dim)
        return self.fusion(flattened)


class NeighborhoodConcatClassifierV2(NeighborhoodConcatClassifier):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        kernel_size: int = 3,
        hidden_dim: int | None = None,
    ) -> None:
        if kernel_size < 3:
            raise ValueError("neighbor_concat_v2 requires kernel_size >= 3 so at least one neighbor remains.")
        super().__init__(
            embed_dim=embed_dim,
            num_classes=num_classes,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
        )
        fusion_dim = hidden_dim or max(embed_dim, 256)
        neighbor_count = kernel_size * kernel_size - 1
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * neighbor_count, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, num_classes),
        )
        self.center_slot = (kernel_size * kernel_size) // 2

    def forward(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = tokens.shape
        local_tokens, _ = self._extract_local_tokens(tokens, grid_size=grid_size)
        local_tokens = torch.cat(
            (
                local_tokens[:, :, : self.center_slot, :],
                local_tokens[:, :, self.center_slot + 1 :, :],
            ),
            dim=2,
        )
        flattened = local_tokens.reshape(batch_size, num_tokens, (self.kernel_size * self.kernel_size - 1) * embed_dim)
        return self.fusion(flattened)


class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        image_size: int = 80,
        patch_size: int = 5,
        in_channels: int = 1,
        embed_dim: int = 192,
        token_embed_dim: int = 192,
        depth: int = 14,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_embed_scale: float = 0.08,
        decoder_channels: tuple[int, ...] = (256, 128, 64),
        num_classes: int = 4,
        segmentation_head: str = "linear",
        classifier_context_kernel_size: int = 1,
        classifier_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if token_embed_dim != embed_dim:
            raise ValueError("token_embed_dim must match embed_dim when zero padding is disabled")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.token_embed_dim = token_embed_dim
        self.pos_embed_scale = pos_embed_scale
        self.num_classes = num_classes
        self.segmentation_head = segmentation_head
        self.classifier_context_kernel_size = classifier_context_kernel_size
        self.classifier_hidden_dim = classifier_hidden_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            token_embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def _apply_random_mask(
        self, tokens: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask_ratio <= 0.0:
            visible_mask = torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                device=tokens.device,
                dtype=torch.bool,
            )
            return tokens, visible_mask
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in the range [0.0, 1.0).")

        num_masked = int(tokens.shape[1] * mask_ratio)
        if num_masked == 0:
            visible_mask = torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                device=tokens.device,
                dtype=torch.bool,
            )
            return tokens, visible_mask
        batch_size, num_tokens, _ = tokens.shape
        noise = torch.rand(batch_size, num_tokens, device=tokens.device)
        mask_indices = noise.argsort(dim=1)[:, :num_masked]
        masked = tokens.clone()
        visible_mask = torch.ones(
            batch_size,
            num_tokens,
            device=tokens.device,
            dtype=torch.bool,
        )
        masked.scatter_(
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, masked.shape[-1]),
            self.mask_token.expand(batch_size, num_masked, -1),
        )
        visible_mask.scatter_(1, mask_indices, False)
        return masked, visible_mask

    def encode(self, x: torch.Tensor, mask_ratio: float = 0.0) -> ViTEncoderOutput:
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens, visible_mask = self._apply_random_mask(tokens, mask_ratio=mask_ratio)
        tokens = self.pos_dropout(tokens + self.pos_embed_scale * self.pos_embed)
        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)
        feature_map = tokens.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, self.grid_size, self.grid_size
        )
        pooled = tokens.mean(dim=1)
        return ViTEncoderOutput(
            tokens=tokens,
            feature_map=feature_map,
            pooled=pooled,
            visible_mask=visible_mask,
        )

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        encoded = self.encode(x, mask_ratio=mask_ratio)
        return self.decoder(encoded.feature_map)

    def classify_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.classifier, (NeighborhoodConcatClassifier, NeighborhoodConcatClassifierV2)):
            return self.classifier(tokens, grid_size=self.grid_size)
        return self.classifier(tokens)

    def forward_with_aux(
        self, x: torch.Tensor, mask_ratio: float = 0.0
    ) -> ViTAutoencoderAuxOutput:
        encoded = self.encode(x, mask_ratio=mask_ratio)
        reconstruction = self.decoder(encoded.feature_map)
        token_logits = self.classify_tokens(encoded.tokens)
        return ViTAutoencoderAuxOutput(
            reconstruction=reconstruction,
            token_logits=token_logits,
            visible_mask=encoded.visible_mask,
            encoded=encoded,
        )

    @torch.inference_mode()
    def predict_token_logits(
        self, x: torch.Tensor, mask_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x, mask_ratio=mask_ratio)
        token_logits = self.classify_tokens(encoded.tokens)
        return token_logits, encoded.visible_mask

    @torch.inference_mode()
    def extract_embeddings(
        self, x: torch.Tensor, pooling: str = "mean"
    ) -> torch.Tensor:
        encoded = self.encode(x)
        if pooling == "mean":
            return encoded.pooled
        if pooling == "tokens":
            return encoded.tokens
        if pooling == "feature_map":
            return encoded.feature_map
        raise ValueError(
            f"Unsupported pooling '{pooling}'. Use 'mean', 'tokens', or 'feature_map'."
        )
