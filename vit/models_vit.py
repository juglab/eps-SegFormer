from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTEncoderOutput:
    tokens: torch.Tensor
    feature_map: torch.Tensor
    pooled: torch.Tensor


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


class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 1,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        decoder_channels: tuple[int, ...] = (256, 128, 64),
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
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
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def encode(self, x: torch.Tensor) -> ViTEncoderOutput:
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.pos_dropout(tokens + self.pos_embed)
        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)
        feature_map = tokens.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, self.grid_size, self.grid_size
        )
        pooled = tokens.mean(dim=1)
        return ViTEncoderOutput(tokens=tokens, feature_map=feature_map, pooled=pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)
        return self.decoder(encoded.feature_map)

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
