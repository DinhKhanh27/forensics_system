"""
DWT-MobileUNet Architecture, GPU-Accelerated 2-Level Haar DWT Stem & Multi-Task Loss.

Key Features:
1. Built-in 2-Level 2D Haar DWT (`TorchHaarDWT2Level`) inside the network graph:
   - Accepts 3-channel RGB input (B, 3, 192, 192) in [0, 1] and computes Level-1
     [LH1, HL1, HH1] (W/2 x H/2) + Level-2 [LH2, HL2, HH2] (W/4 x H/4) high-frequency
     subbands on GPU/ONNX, bilinearly upsamples them to 192x192, applies per-channel
     Z-score normalization, and fuses them with RGB into a 9-channel MRA tensor.
   - Also supports pre-computed 9-channel or 11-channel MRA inputs.
2. Full-Resolution MRA Edge Skip (`enc_stage0` at 192x192) + 5-Stage MobileNetV3-Small
   (`width_multiplier=0.5`) Encoder-Decoder so pixel-level splicing boundaries remain crisp
   at 192x192 while keeping the INT8 quantized model size strictly < 2.0 MB (~0.45 MB).
3. ONNX-Safe Pooling Operations (`mean` & `max` over spatial dims instead of `adaptive_max_pool2d`)
   for 100% trouble-free export to ONNX Runtime INT8.
4. Explicit `return_logits` control:
   - `return_logits=True` during training for AMP-stable `BCEWithLogitsLoss` + `Focal` + `Dice`.
   - `return_logits=False` during inference & ONNX export so `torch.sigmoid` is baked into the graph.
"""

from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class TorchHaarDWT2Level(nn.Module):
    """
    Differentiable & ONNX-exportable 2-Level 2D Haar Discrete Wavelet Transform (MRA)
    operating directly on RGB tensors (B, 3, H, W) in [0, 1].
    Produces a 9-channel Early-Fusion tensor (B, 9, H, W).
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    @staticmethod
    def _dwt1level(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x00 = y[:, :, 0::2, 0::2]
        x01 = y[:, :, 0::2, 1::2]
        x10 = y[:, :, 1::2, 0::2]
        x11 = y[:, :, 1::2, 1::2]

        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, lh, hl, hh

    def _zscore(self, t: torch.Tensor) -> torch.Tensor:
        mean = t.mean(dim=(2, 3), keepdim=True)
        var = ((t - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(var + self.eps)
        return (t - mean) / std

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        target_size = rgb.shape[-2:]

        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        ll1, lh1, hl1, hh1 = self._dwt1level(lum)
        _, lh2, hl2, hh2 = self._dwt1level(ll1)

        l1_stack = torch.cat([lh1, hl1, hh1], dim=1)
        l2_stack = torch.cat([lh2, hl2, hh2], dim=1)

        l1_up = F.interpolate(l1_stack, size=target_size, mode="bilinear", align_corners=False)
        l2_up = F.interpolate(l2_stack, size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([rgb, l1_up, l2_up], dim=1)
        return self._zscore(fused)


class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight Depthwise Separable Convolution block (3x3 Depthwise + 1x1 Pointwise).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
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
    """

    def __init__(
        self,
        in_channels: int = 3,
        width_multiplier: float = 0.5,
        decoder_channels: List[int] = None,
        dropout: float = 0.2,
        return_logits: bool = False,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [96, 48, 24, 16]

        self.in_channels = in_channels
        self.width_multiplier = width_multiplier
        self.return_logits = return_logits

        if in_channels == 3:
            self.mra_stem = TorchHaarDWT2Level()
            stem_in_channels = 9
        else:
            self.mra_stem = None
            stem_in_channels = in_channels

        # Full-resolution (192x192) MRA skip branch to preserve sharp high-frequency wavelet edges
        self.enc_stage0 = DepthwiseSeparableConv(stem_in_channels, 16)

        # MobileNetV3-Small encoder (width_multiplier=0.5)
        base_mnet = mobilenet_v3_small(weights=None, width_mult=width_multiplier)
        features = base_mnet.features

        first_conv: nn.Conv2d = features[0][0]
        out_stem_ch = first_conv.out_channels
        features[0][0] = nn.Conv2d(
            in_channels=stem_in_channels,
            out_channels=out_stem_ch,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        self.enc_stage1 = nn.Sequential(*features[0:1])   # 1/2  (96x96)
        self.enc_stage2 = nn.Sequential(*features[1:2])   # 1/4  (48x48)
        self.enc_stage3 = nn.Sequential(*features[2:4])   # 1/8  (24x24)
        self.enc_stage4 = nn.Sequential(*features[4:9])   # 1/16 (12x12)
        self.enc_stage5 = nn.Sequential(*features[9:13])  # 1/32 (6x6)

        with torch.no_grad():
            dummy = torch.zeros(1, stem_in_channels, 192, 192)
            s1 = self.enc_stage1(dummy)
            s2 = self.enc_stage2(s1)
            s3 = self.enc_stage3(s2)
            s4 = self.enc_stage4(s3)
            s5 = self.enc_stage5(s4)
            c1, c2, c3, c4, c5 = (
                s1.shape[1],
                s2.shape[1],
                s3.shape[1],
                s4.shape[1],
                s5.shape[1],
            )

        # Head 2 — Localization (Pixel-level Forgery Mask Decoder)
        d1_ch, d2_ch, d3_ch, d4_ch = decoder_channels
        self.up1 = DecoderUpBlock(in_ch=c5, skip_ch=c4, out_ch=d1_ch)     # 6x6   -> 12x12
        self.up2 = DecoderUpBlock(in_ch=d1_ch, skip_ch=c3, out_ch=d2_ch)  # 12x12 -> 24x24
        self.up3 = DecoderUpBlock(in_ch=d2_ch, skip_ch=c2, out_ch=d3_ch)  # 24x24 -> 48x48
        self.up4 = DecoderUpBlock(in_ch=d3_ch, skip_ch=c1, out_ch=d4_ch)  # 48x48 -> 96x96

        # Full-resolution fusion (96x96 -> 192x192 concatenated with 192x192 enc_stage0 skip)
        self.final_refine = DecoderUpBlock(in_ch=d4_ch, skip_ch=16, out_ch=d4_ch)
        self.seg_head_conv = nn.Conv2d(d4_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Head 1 — Detection (Global Average + Spatial Max Pooling on Bottleneck + Mask Response)
        self.detection_fc = nn.Sequential(
            nn.Linear(c5 * 2 + 2, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
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
        if self.seg_head_conv.bias is not None:
            nn.init.constant_(self.seg_head_conv.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mra_stem is not None and x.shape[1] == 3:
            x_mra = self.mra_stem(x)
        else:
            x_mra = x

        s0 = self.enc_stage0(x_mra)  # 1/1  (192x192)
        s1 = self.enc_stage1(x_mra)  # 1/2  (96x96)
        s2 = self.enc_stage2(s1)     # 1/4  (48x48)
        s3 = self.enc_stage3(s2)     # 1/8  (24x24)
        s4 = self.enc_stage4(s3)     # 1/16 (12x12)
        s5 = self.enc_stage5(s4)     # 1/32 (6x6)

        d1 = self.up1(s5, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)
        d_full = self.final_refine(d4, s0)  # (B, d4_ch, 192, 192)

        logits_mask = self.seg_head_conv(d_full)  # (B, 1, 192, 192)
        prob_mask = torch.sigmoid(logits_mask)

        # ONNX-friendly spatial Global Average Pool and Global Max Pool
        gap_feat = s5.mean(dim=(2, 3))                  # (B, c5)
        gmp_feat = s5.max(dim=3)[0].max(dim=2)[0]       # (B, c5)
        mask_mean = prob_mask.mean(dim=(2, 3))          # (B, 1)
        mask_max = prob_mask.max(dim=3)[0].max(dim=2)[0]  # (B, 1)

        cls_in = torch.cat([gap_feat, gmp_feat, mask_mean, mask_max], dim=1)
        logits_cls = self.detection_fc(cls_in)          # (B, 1)

        if self.return_logits:
            return logits_cls, logits_mask

        prob_cls = torch.sigmoid(logits_cls)
        return prob_cls, prob_mask


class LegacyDoubleConv(nn.Module):
    """Compatibility block for checkpoints trained with the 2-down UNet in commit 8d0513a."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyDWTUNet(nn.Module):
    """
    Backward-compatible architecture matching commit 8d0513a (`inc`, `down1`, `down2`, `up1`, `up2`),
    FIXED to apply `torch.sigmoid` when `return_logits=False` so inference & ONNX export output
    true probabilities in [0, 1] instead of raw unbounded logits.
    """

    def __init__(self, return_logits: bool = False) -> None:
        super().__init__()
        self.in_channels = 3
        self.return_logits = return_logits
        self.inc = LegacyDoubleConv(15, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), LegacyDoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), LegacyDoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = LegacyDoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = LegacyDoubleConv(64, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)
        self.out_mask = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _haar_dwt_2d(x: torch.Tensor) -> torch.Tensor:
        ll = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
        lh = (-x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
        hl = (-x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
        hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dwt_feat = self._haar_dwt_2d(x)
        dwt_upsampled = F.interpolate(dwt_feat, size=x.shape[2:], mode="nearest")
        x_in = torch.cat([x, dwt_upsampled], dim=1)
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        cls_feat = self.gap(x3).view(x3.size(0), -1)
        logits_cls = self.fc(cls_feat)
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        logits_mask = self.out_mask(u2)
        if self.return_logits:
            return logits_cls, logits_mask
        return torch.sigmoid(logits_cls), torch.sigmoid(logits_mask)


class MultiTaskForensicLoss(nn.Module):
    """
    Corrected Multi-Task Forensic Loss:
        L_total = 0.3 * L_BCE_cls + 0.7 * (L_Focal_All + L_BCE_All + L_Dice_Tampered + L_Au_Suppression)
    """

    def __init__(
        self,
        cls_weight: float = 0.3,
        seg_weight: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_pixel_weight: float = 3.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_pixel_weight = pos_pixel_weight
        self.smooth = smooth

    def _focal_loss_with_logits(
        self, logits_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits_mask, target_mask, reduction="none")
        prob = torch.sigmoid(logits_mask)
        p_t = target_mask * prob + (1.0 - target_mask) * (1.0 - prob)
        alpha_t = target_mask * self.focal_alpha + (1.0 - target_mask) * (1.0 - self.focal_alpha)
        focal = alpha_t * ((1.0 - p_t) ** self.focal_gamma) * bce
        return focal.mean()

    def _dice_loss_from_logits(
        self, logits_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits_mask)
        intersection = (prob * target_mask).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()

    def forward(
        self,
        logits_cls: torch.Tensor,
        logits_mask: torch.Tensor,
        target_cls: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits_cls_f32 = logits_cls.float().view(-1, 1)
        target_cls_f32 = target_cls.float().view(-1, 1)
        logits_mask_f32 = logits_mask.float()
        target_mask_f32 = target_mask.float()

        loss_bce_cls = F.binary_cross_entropy_with_logits(logits_cls_f32, target_cls_f32)

        pos_w = torch.tensor([self.pos_pixel_weight], device=logits_mask_f32.device)
        loss_bce_all = F.binary_cross_entropy_with_logits(
            logits_mask_f32, target_mask_f32, pos_weight=pos_w
        )
        loss_focal_all = self._focal_loss_with_logits(logits_mask_f32, target_mask_f32)

        tp_indices = (target_cls_f32.view(-1) > 0.5)
        if tp_indices.any():
            loss_dice_tp = self._dice_loss_from_logits(
                logits_mask_f32[tp_indices], target_mask_f32[tp_indices]
            )
        else:
            loss_dice_tp = torch.tensor(0.0, device=logits_mask_f32.device)

        au_indices = ~tp_indices
        if au_indices.any():
            au_prob = torch.sigmoid(logits_mask_f32[au_indices])
            loss_au_fp = (au_prob ** 2).mean()
        else:
            loss_au_fp = torch.tensor(0.0, device=logits_mask_f32.device)

        loss_seg = 0.5 * loss_bce_all + loss_focal_all + loss_dice_tp + 0.5 * loss_au_fp
        total_loss = self.cls_weight * loss_bce_cls + self.seg_weight * loss_seg

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_bce_cls": float(loss_bce_cls.detach().item()),
            "loss_dice": float(loss_dice_tp.detach().item()),
            "loss_focal": float(loss_focal_all.detach().item()),
        }
        return total_loss, loss_dict


def load_forensic_checkpoint(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    return_logits: bool = False,
    default_in_channels: int = 3,
) -> Tuple[nn.Module, int, str]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if "inc.net.0.weight" in state_dict:
        model = LegacyDWTUNet(return_logits=return_logits).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model, 3, "LegacyDWTUNet"

    in_channels = (
        ckpt.get("in_channels", default_in_channels)
        if isinstance(ckpt, dict)
        else default_in_channels
    )
    width_mult = ckpt.get("width_multiplier", 0.5) if isinstance(ckpt, dict) else 0.5
    model = DWTMobileUNet(
        in_channels=in_channels,
        width_multiplier=width_mult,
        return_logits=return_logits,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model, in_channels, "DWTMobileUNet"


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_int8_mb = total_params / (1024.0 * 1024.0)
    return total_params, est_int8_mb
