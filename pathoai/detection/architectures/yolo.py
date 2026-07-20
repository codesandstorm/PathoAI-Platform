"""
pathoai/detection/architectures/yolo.py
======================================
YOLO Detector Architecture Wrapper.

Initial registered object detector backend using a custom/lightweight YOLO-style
feature pyramid network for cell detection.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.2
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathoai.detection.registry import register_detector


class ConvBlock(nn.Module):
    """Standard Convolution-BatchNorm-SiLU layer block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@register_detector("yolo")
class YOLODetector(nn.Module):
    """YOLO Object Detector architecture for single/multi-class cell detection.

    Processes batch tensors of shape (B, 3, H, W) and outputs bounding boxes,
    class probabilities, and confidence scores.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 4,
        backbone_channels: Tuple[int, ...] = (32, 64, 128, 256),
        num_anchors: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input image channels (typically 3 for RGB).
        n_classes : int
            Number of target cell classes.
        backbone_channels : Tuple[int, ...]
            Feature channel dimensions for backbone stages.
        num_anchors : int
            Number of anchors per grid cell.
        """
        super().__init__()
        self.n_classes = n_classes
        self.num_anchors = num_anchors

        # Stem & Backbone
        c1, c2, c3, c4 = backbone_channels
        self.stem = ConvBlock(in_channels, c1, kernel_size=3, stride=1)
        self.stage1 = ConvBlock(c1, c2, kernel_size=3, stride=2)
        self.stage2 = ConvBlock(c2, c3, kernel_size=3, stride=2)
        self.stage3 = ConvBlock(c3, c4, kernel_size=3, stride=2)

        # Detection Head Output channels: (num_anchors * (5 + n_classes))
        # 5 attributes: center_x, center_y, width, height, objectness
        self.out_channels = num_anchors * (5 + n_classes)
        self.head = nn.Conv2d(c4, self.out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch images of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor or List[Dict[str, torch.Tensor]]
            If training: raw detection grid predictions tensor of shape (B, out_channels, H_out, W_out).
            If eval: parsed predictions list containing 'boxes', 'scores', 'labels'.
        """
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        raw_pred = self.head(feat)

        if self.training:
            return raw_pred

        # Post-process grid predictions during eval/inference mode
        return self._decode_predictions(x, raw_pred)

    def _decode_predictions(
        self, x: torch.Tensor, raw_pred: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode raw grid tensor into list of box dicts per image."""
        batch_size, _, h_out, w_out = raw_pred.shape
        img_h, img_w = x.shape[2], x.shape[3]

        # Reshape to (B, num_anchors, 5 + n_classes, H_out, W_out)
        pred = raw_pred.view(batch_size, self.num_anchors, 5 + self.n_classes, h_out, w_out)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # (B, num_anchors, H_out, W_out, 5 + n_classes)

        results = []
        for b in range(batch_size):
            b_pred = pred[b].view(-1, 5 + self.n_classes)  # (N_anchors*H*W, 5 + n_classes)

            # Objectness confidence
            obj_conf = torch.sigmoid(b_pred[:, 4])
            class_probs = torch.softmax(b_pred[:, 5:], dim=-1)

            # Best class score and label
            max_class_scores, class_labels = torch.max(class_probs, dim=-1)
            total_scores = obj_conf * max_class_scores

            # Grid center decoding
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h_out, device=x.device),
                torch.arange(w_out, device=x.device),
                indexing="ij",
            )
            grid_x = grid_x.repeat_interleave(self.num_anchors).view(-1)
            grid_y = grid_y.repeat_interleave(self.num_anchors).view(-1)

            # Raw offsets
            bx = (torch.sigmoid(b_pred[:, 0]) + grid_x) * (img_w / w_out)
            by = (torch.sigmoid(b_pred[:, 1]) + grid_y) * (img_h / h_out)
            bw = torch.exp(b_pred[:, 2]) * (img_w / w_out)
            bh = torch.exp(b_pred[:, 3]) * (img_h / h_out)

            x1 = torch.clamp(bx - bw / 2.0, min=0.0, max=float(img_w))
            y1 = torch.clamp(by - bh / 2.0, min=0.0, max=float(img_h))
            x2 = torch.clamp(bx + bw / 2.0, min=0.0, max=float(img_w))
            y2 = torch.clamp(by + bh / 2.0, min=0.0, max=float(img_h))

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            results.append({
                "boxes": boxes,
                "scores": total_scores,
                "labels": class_labels,
            })

        return results
