"""
Explainable AI (XAI) Module for MedSAM
Provides attention visualization and interpretability for medical image segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class MedSAMExplainer:
    """
    Explainability module for MedSAM segmentation.
    Provides attention maps and visual explanations.
    """
    
    def __init__(self, model, device="cpu"):
        """
        Initialize the explainer.
        
        Args:
            model: Trained MedSAM model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.attention_maps = {}
        self.gradients = {}
        
        # Register hooks to capture attention
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention from the model."""
        
        def get_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.attention_maps[name] = output[1] if len(output) > 1 else output[0]
                else:
                    self.attention_maps[name] = output
            return hook
        
        # Hook into the image encoder's attention blocks
        if hasattr(self.model, 'sam') and hasattr(self.model.sam, 'image_encoder'):
            encoder = self.model.sam.image_encoder
            if hasattr(encoder, 'blocks'):
                # Register hook on the last transformer block
                encoder.blocks[-1].register_forward_hook(get_attention_hook('encoder_attention'))
    
    def generate_attention_map(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate attention map for the input image.
        
        Args:
            image: Input image (H, W, 3) in RGB
            box: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Tuple of (attention_map, segmentation_mask)
        """
        self.model.eval()
        
        # Preprocess image
        img_tensor = self._preprocess_image(image)
        
        # Forward pass
        with torch.no_grad():
            image_embedding = self.model(img_tensor)
            
            # Get prompt embeddings
            box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device).unsqueeze(0)
            sparse_emb, dense_emb = self.model.get_prompt_embeddings(box_tensor)
            
            # Decode
            logits = self.model.decode(image_embedding, sparse_emb, dense_emb)
            
            # Get segmentation mask
            pred = F.interpolate(logits, size=(image.shape[0], image.shape[1]), 
                               mode='bilinear', align_corners=False)
            mask = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()
        
        # Extract attention from image embeddings
        # Use the magnitude of embeddings as a proxy for attention
        attention = torch.abs(image_embedding).mean(dim=1).squeeze()  # (64, 64)
        attention = F.interpolate(
            attention.unsqueeze(0).unsqueeze(0),
            size=(image.shape[0], image.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        return attention, mask
    
    def create_overlay_visualization(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization overlay with attention map and mask.
        
        Args:
            image: Original image (H, W, 3)
            attention_map: Attention heatmap (H, W)
            mask: Segmentation mask (H, W)
            alpha: Transparency for overlay
        
        Returns:
            Visualization image (H, W, 3)
        """
        # Normalize image to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Create colored attention heatmap
        attention_colored = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
        
        # Blend image with attention
        overlay = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)
        
        # Draw mask contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = overlay.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        return overlay
    
    def generate_feature_importance(
        self,
        attention_map: np.ndarray,
        mask: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Generate feature importance scores for different regions.
        
        Args:
            attention_map: Attention heatmap
            mask: Segmentation mask
            top_k: Number of top regions to return
        
        Returns:
            Dictionary with region importance scores
        """
        # Divide image into quadrants/regions
        h, w = attention_map.shape
        regions = {
            'center': attention_map[h//3:2*h//3, w//3:2*w//3],
            'top': attention_map[0:h//3, :],
            'bottom': attention_map[2*h//3:, :],
            'left': attention_map[:, 0:w//3],
            'right': attention_map[:, 2*w//3:],
        }
        
        # Calculate average attention in each region
        importance_scores = {}
        for region_name, region_data in regions.items():
            # Weight by mask presence
            region_mask = None
            if region_name == 'center':
                region_mask = mask[h//3:2*h//3, w//3:2*w//3]
            elif region_name == 'top':
                region_mask = mask[0:h//3, :]
            elif region_name == 'bottom':
                region_mask = mask[2*h//3:, :]
            elif region_name == 'left':
                region_mask = mask[:, 0:w//3]
            elif region_name == 'right':
                region_mask = mask[:, 2*w//3:]
            
            # Calculate weighted importance
            if region_mask is not None and region_mask.sum() > 0:
                importance = (region_data * region_mask).sum() / (region_mask.sum() + 1e-8)
            else:
                importance = region_data.mean()
            
            importance_scores[region_name] = float(importance)
        
        # Sort by importance
        sorted_regions = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_regions': sorted_regions[:top_k],
            'all_scores': importance_scores,
        }
    
    def generate_explanation_report(
        self,
        image: np.ndarray,
        box: np.ndarray,
        confidence: float = None
    ) -> Dict:
        """
        Generate comprehensive explanation report.
        
        Args:
            image: Input image
            box: Bounding box
            confidence: Model confidence score
        
        Returns:
            Dictionary with explanation data
        """
        # Generate attention and mask
        attention_map, mask = self.generate_attention_map(image, box)
        
        # Calculate mask statistics
        mask_area = mask.sum() / (mask.shape[0] * mask.shape[1])
        
        # Get feature importance
        feature_importance = self.generate_feature_importance(attention_map, mask)
        
        # Create visualization
        visualization = self.create_overlay_visualization(image, attention_map, mask)
        
        # Generate description
        top_region = feature_importance['top_regions'][0]
        description = f"The model focused primarily on the {top_region[0]} region "
        description += f"(importance: {top_region[1]:.2%}). "
        description += f"Detected tissue coverage: {mask_area:.2%} of image area."
        
        return {
            'attention_map': attention_map,
            'segmentation_mask': mask,
            'visualization': visualization,
            'feature_importance': feature_importance,
            'mask_coverage': float(mask_area),
            'confidence': confidence,
            'description': description,
        }
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Preprocessed tensor (1, 3, 1024, 1024)
        """
        # Resize to 1024x1024
        if image.shape[:2] != (1024, 1024):
            image = cv2.resize(image, (1024, 1024))
        
        # Normalize
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def save_explanation_visualization(
        self,
        visualization: np.ndarray,
        save_path: str
    ):
        """Save explanation visualization to file."""
        Image.fromarray(visualization).save(save_path)


def create_comparison_plot(
    original_image: np.ndarray,
    attention_map: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    save_path: str = None
):
    """
    Create a comparison plot with original, attention, mask, and overlay.
    
    Args:
        original_image: Original input image
        attention_map: Attention heatmap
        mask: Segmentation mask
        overlay: Overlay visualization
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title('XAI Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
