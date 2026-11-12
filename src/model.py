"""
DWT-MobileUNet Architecture & Multi-Task Forensic Loss Module.

Architecture Specifications:
- Encoder: Custom MobileNetV3-Small with width_multiplier=0.5 and in_channels=9 (or 11 with MAD+ELA).
  Extracts multi-scale feature maps at 1/2, 1/4, 1/8, 1/16, and bottleneck 1/32 resolution (6x6 for 192x192 input).
- Head 1 (Detection): Bottleneck (1/32) -> Global Average Pooling -> Dense(64) -> Hardswish -> Dropout(0.2) -> Dense(1) -> Sigmoid.
  Outputs forgery probability P_fake in [0, 1].
- Head 2 (Localization): 4 Decoder stages of 2x Bilinear Upsampling + Skip Connection Concatenation +
  Depthwise Separable Convolution -> final 2x Upsampling + Depthwise Separable Refine -> Conv 1x1 + Sigmoid.
  Outputs pixel-level forgery probability mask of shape (B, 1, 192, 192).
- Multi-Task Loss:
  L_total = 0.3 * L_BCE_cls + 0.7 * (L_Dice + L_Focal)
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight Depthwise Separable Convolution block (3x3 Depthwise + 1x1 Pointwise)
    to minimize parameters and MACs for Raspberry Pi 3 deployment.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # 3x3 Depthwise Convolution
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(inplace=True),
            # 1x1 Pointwise Convolution
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderUpBlock(nn.Module):
    """
    Single Decoder Stage:
    2x Bilinear Upsampling -> Concatenate Encoder Skip Feature -> 2x Depthwise Separable Conv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch + skip_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x_cat = torch.cat([x_up, skip], dim=1)
        return self.conv(x_cat)


class DWTMobileUNet(nn.Module):
    """
    Dual-Head DWT-MobileUNet for Embedded Image Forgery Detection and Localization.
    Designed to keep INT8 quantized weight footprint strictly < 2 MB (~0.45M parameters).
    """

    def __init__(
        self,
        in_channels: int = 9,
        width_multiplier: float = 0.5,
        decoder_channels: List[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [96, 48, 24, 16]

        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        # 1. Build MobileNetV3-Small encoder with width_multiplier=0.5
        base_mnet = mobilenet_v3_small(weights=None, width_mult=width_multiplier)
        features = base_mnet.features

        # Adapt the first stem Conv2d from 3 channels to 9 (or 11) MRA channels
        first_conv: nn.Conv2d = features[0][0]
        out_stem_ch = first_conv.out_channels
        features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_stem_ch,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        # Split MobileNetV3-Small features into 5 spatial scales for 192x192 input:
        # Stage 1 (scale 1/2,  96x96): features[0:1]   -> 16 ch (when width_mult=0.5)
        # Stage 2 (scale 1/4,  48x48): features[1:2]   -> 8 ch
        # Stage 3 (scale 1/8,  24x24): features[2:4]   -> 16 ch
        # Stage 4 (scale 1/16, 12x12): features[4:9]   -> 24 ch
        # Stage 5 (scale 1/32,  6x6 ): features[9:13]  -> 288 ch (bottleneck)
        self.enc_stage1 = nn.Sequential(*features[0:1])
        self.enc_stage2 = nn.Sequential(*features[1:2])
        self.enc_stage3 = nn.Sequential(*features[2:4])
        self.enc_stage4 = nn.Sequential(*features[4:9])
        self.enc_stage5 = nn.Sequential(*features[9:13])

        # Probe dynamic channel dimensions across encoder stages
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 192, 192)
            s1 = self.enc_stage1(dummy)  # (1, C1, 96, 96)
            s2 = self.enc_stage2(s1)     # (1, C2, 48, 48)
            s3 = self.enc_stage3(s2)     # (1, C3, 24, 24)
            s4 = self.enc_stage4(s3)     # (1, C4, 12, 12)
            s5 = self.enc_stage5(s4)     # (1, C5, 6, 6)
            c1, c2, c3, c4, c5 = (
                s1.shape[1],
                s2.shape[1],
                s3.shape[1],
                s4.shape[1],
                s5.shape[1],
            )

        # 2. Head 1 — Detection (Image-level Authentic / Forged Classification)
        # Bottleneck (1/32) -> Global Average Pooling -> Dense(64) -> Dropout(0.2) -> Dense(1) -> Sigmoid
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(c5, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 3. Head 2 — Localization (Pixel-level Forgery Mask Decoder)
        # 4 Upsampling Decoder blocks + Skip Connections from Encoder
        d1_ch, d2_ch, d3_ch, d4_ch = decoder_channels
        self.up1 = DecoderUpBlock(in_ch=c5, skip_ch=c4, out_ch=d1_ch)  # 6x6   -> 12x12
        self.up2 = DecoderUpBlock(in_ch=d1_ch, skip_ch=c3, out_ch=d2_ch)  # 12x12 -> 24x24
        self.up3 = DecoderUpBlock(in_ch=d2_ch, skip_ch=c2, out_ch=d3_ch)  # 24x24 -> 48x48
        self.up4 = DecoderUpBlock(in_ch=d3_ch, skip_ch=c1, out_ch=d4_ch)  # 48x48 -> 96x96

        # Final 2x upsampling refinement from 96x96 -> 192x192 + Conv 1x1 + Sigmoid
        self.final_refine = DepthwiseSeparableConv(d4_ch, d4_ch)
        self.seg_head = nn.Sequential(
            nn.Conv2d(d4_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, 192, 192) where C = 9 or 11.

        Returns:
            Tuple of:
                - prob_fake: Image-level forgery probability (B, 1) in [0, 1]
                - mask_prob: Pixel-level localization mask (B, 1, 192, 192) in [0, 1]
        """
        input_size = x.shape[-2:]

        # Encoder forward with skip connections
        s1 = self.enc_stage1(x)   # 1/2  (96x96)
        s2 = self.enc_stage2(s1)  # 1/4  (48x48)
        s3 = self.enc_stage3(s2)  # 1/8  (24x24)
        s4 = self.enc_stage4(s3)  # 1/16 (12x12)
        s5 = self.enc_stage5(s4)  # 1/32 (6x6)

        # Head 1: Global Forgery Detection
        prob_fake = self.detection_head(s5)  # (B, 1)

        # Head 2: Multi-scale Decoder Localization
        d1 = self.up1(s5, s4)  # (B, d1_ch, 12, 12)
        d2 = self.up2(d1, s3)  # (B, d2_ch, 24, 24)
        d3 = self.up3(d2, s2)  # (B, d3_ch, 48, 48)
        d4 = self.up4(d3, s1)  # (B, d4_ch, 96, 96)

        # Final 2x upsampling to full 192x192 resolution
        d_full = F.interpolate(d4, size=input_size, mode="bilinear", align_corners=False)
        d_full = self.final_refine(d_full)
        mask_prob = self.seg_head(d_full)  # (B, 1, 192, 192)

        return prob_fake, mask_prob


class MultiTaskForensicLoss(nn.Module):
    """
    Composite Multi-Task Loss for DWT-MobileUNet:
        L_total = 0.3 * L_BCE_cls + 0.7 * (L_Dice + L_Focal)

    Computes loss in FP32 even under AMP autocast so Sigmoid outputs remain numerically stable.
    """

    def __init__(
        self,
        cls_weight: float = 0.3,
        seg_weight: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth

    def _dice_loss(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_mask.contiguous().view(pred_mask.size(0), -1)
        target_flat = target_mask.contiguous().view(target_mask.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice_score).mean()

    def _focal_loss(
        self, pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        pred_clamped = torch.clamp(pred_mask, min=eps, max=1.0 - eps)
        bce = -target_mask * torch.log(pred_clamped) - (1.0 - target_mask) * torch.log(
            1.0 - pred_clamped
        )
        p_t = target_mask * pred_clamped + (1.0 - target_mask) * (1.0 - pred_clamped)
        alpha_t = target_mask * self.focal_alpha + (1.0 - target_mask) * (1.0 - self.focal_alpha)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.focal_gamma)
        return (focal_weight * bce).mean()

    def _bce_loss(
        self, pred_prob: torch.Tensor, target_label: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        pred_clamped = torch.clamp(pred_prob, min=eps, max=1.0 - eps)
        bce = -target_label * torch.log(pred_clamped) - (1.0 - target_label) * torch.log(
            1.0 - pred_clamped
        )
        return bce.mean()

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_mask: torch.Tensor,
        target_cls: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total multi-task loss and individual loss components.

        Args:
            pred_cls: Predicted forgery probability (B, 1) in [0, 1].
            pred_mask: Predicted forgery localization mask (B, 1, H, W) in [0, 1].
            target_cls: Ground-truth binary class label (B, 1) or (B,) in {0, 1}.
            target_mask: Ground-truth binary forgery mask (B, 1, H, W) in {0, 1}.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Cast to float32 for AMP safety with Sigmoid probabilities
        pred_cls_f32 = pred_cls.float().view(-1, 1)
        target_cls_f32 = target_cls.float().view(-1, 1)
        pred_mask_f32 = pred_mask.float()
        target_mask_f32 = target_mask.float()

        loss_bce_cls = self._bce_loss(pred_cls_f32, target_cls_f32)
        loss_dice = self._dice_loss(pred_mask_f32, target_mask_f32)
        loss_focal = self._focal_loss(pred_mask_f32, target_mask_f32)

        total_loss = self.cls_weight * loss_bce_cls + self.seg_weight * (
            loss_dice + loss_focal
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_bce_cls": float(loss_bce_cls.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_focal": float(loss_focal.detach().item()),
        }
        return total_loss, loss_dict


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    """
    Returns total trainable parameter count and estimated INT8 size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_int8_mb = total_params / (1024.0 * 1024.0)  # 1 byte per parameter in INT8
    return total_params, est_int8_mb


if __name__ == "__main__":
    model = DWTMobileUNet(in_channels=9, width_multiplier=0.5)
    params, int8_mb = count_parameters(model)
    dummy_in = torch.randn(2, 9, 192, 192)
    p_fake, mask = model(dummy_in)
    criterion = MultiTaskForensicLoss()
    loss, loss_info = criterion(
        p_fake,
        mask,
        torch.tensor([[1.0], [0.0]]),
        torch.randint(0, 2, (2, 1, 192, 192)).float(),
    )
    print(f"[Phase 2 Model OK] Parameters: {params:,} (~{int8_mb:.2f} MB in INT8)")
    print(f"[Phase 2 Model OK] Head 1 shape: {p_fake.shape}, Head 2 shape: {mask.shape}")
    print(f"[Phase 2 Loss  OK] {loss_info}")
