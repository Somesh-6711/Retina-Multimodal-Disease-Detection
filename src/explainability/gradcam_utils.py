# src/explainability/gradcam_utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class GradCAMConfig:
    target_layer_name: str = "conv_head"  # for EfficientNet from timm
    use_cuda: bool = False


class GradCAMEffNet:
    """
    Simple Grad-CAM implementation for EfficientNet-like models from timm.

    Assumes the model has a final convolutional layer named `conv_head`.
    """

    def __init__(self, model: torch.nn.Module, cfg: Optional[GradCAMConfig] = None):
        self.model = model
        self.cfg = cfg or GradCAMConfig()
        self.model.eval()

        self.device = next(model.parameters()).device

        # Will be populated by hooks
        self.activations = None
        self.gradients = None

        # Register hooks on the target conv layer
        target_layer = getattr(self.model, self.cfg.target_layer_name, None)
        if target_layer is None:
            raise AttributeError(
                f"Model has no layer named '{self.cfg.target_layer_name}'. "
                f"Available attributes: {dir(self.model)}"
            )

        def forward_hook(module, inp, output):
            # Save activations from forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Save gradients from backward pass
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        input_tensor: shape (1, 3, H, W), already normalized.
        class_idx: which class index to backprop; if None, uses predicted class.

        Returns:
            cam: numpy array of shape (H, W), normalized to [0, 1]
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.to(self.device)

        # Forward
        outputs = self.model(input_tensor)
        if outputs.ndim == 2:
            # shape (1, num_classes)
            probs = outputs[0]
        else:
            raise ValueError(f"Unexpected model output shape: {outputs.shape}")

        if class_idx is None:
            class_idx = int(torch.argmax(probs).item())

        score = probs[class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # activations: (1, C, H', W')
        # gradients:   (1, C, H', W')
        grads = self.gradients
        acts = self.activations

        # Global average pooling over spatial dims to get weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Normalize cam to [0, 1]
        cam = cam.squeeze(0).squeeze(0)  # (H', W')
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    @staticmethod
    def overlay_heatmap_on_image(
        cam: np.ndarray,
        orig_img: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        cam: (H, W) in [0, 1]
        orig_img: (H, W, 3) RGB uint8
        Returns:
            overlay: (H, W, 3) RGB uint8
        """
        h, w, _ = orig_img.shape

        # Resize CAM to image resolution
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = np.uint8(255 * cam_resized)

        heatmap = cv2.applyColorMap(cam_uint8, colormap)  # BGR
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig_img, 1 - alpha, heatmap, alpha, 0)
        return overlay

    @staticmethod
    def save_overlay(
        overlay: np.ndarray,
        save_path: str,
    ):
        """
        overlay: (H, W, 3) RGB uint8
        """
        Image.fromarray(overlay).save(save_path)
        print(f"[SAVE] Grad-CAM overlay -> {save_path}")
