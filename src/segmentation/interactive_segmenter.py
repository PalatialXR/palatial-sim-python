from typing import List, Dict, Union, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class InteractiveSegmenter:
    """
    A class to handle interactive segmentation using SAM2
    Supports both automatic and point-based segmentation
    """
    
    def __init__(self, model_name: str = "facebook/sam2.1-hiera-large", device: str = "cuda"):
        """Initialize the interactive segmenter with SAM2
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        
        # Initialize SAM2 model
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        self.image = None
        
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Set the current image for segmentation
        
        Args:
            image: Input image (PIL Image or numpy array)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        self.image = image
        self.predictor.set_image(image)
        
    def predict_from_points(self, 
                          points: np.ndarray,
                          labels: np.ndarray,
                          box: Optional[np.ndarray] = None,
                          multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks from point prompts
        
        Args:
            points: Nx2 array of point coordinates (x, y)
            labels: N array of point labels (1 for foreground, 0 for background)
            box: Optional 4-element array for bounding box in xyxy format
            multimask_output: Whether to return multiple masks per prompt
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
            logits: Raw logits for each mask
        """
        if self.image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        return self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box[None, :] if box is not None else None,
            multimask_output=multimask_output
        )
        
    def predict_from_boxes(self, 
                         boxes: np.ndarray,
                         multimask_output: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks from box prompts
        
        Args:
            boxes: Nx4 array of box coordinates in xyxy format
            multimask_output: Whether to return multiple masks per prompt
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
            logits: Raw logits for each mask
        """
        if self.image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        return self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=multimask_output
        )
        
    def predict_batch(self,
                     images: List[np.ndarray],
                     points_batch: Optional[List[np.ndarray]] = None,
                     labels_batch: Optional[List[np.ndarray]] = None,
                     boxes_batch: Optional[List[np.ndarray]] = None,
                     multimask_output: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Batch prediction for multiple images
        
        Args:
            images: List of input images
            points_batch: Optional list of point coordinate arrays
            labels_batch: Optional list of point label arrays
            boxes_batch: Optional list of box coordinate arrays
            multimask_output: Whether to return multiple masks per prompt
            
        Returns:
            masks_batch: List of mask arrays
            scores_batch: List of confidence score arrays
            logits_batch: List of logit arrays
        """
        self.predictor.set_image_batch(images)
        return self.predictor.predict_batch(
            points_batch,
            labels_batch,
            box_batch=boxes_batch,
            multimask_output=multimask_output
        )
        
    def visualize_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Visualize masks with optional points and box."""
        # Start with original image
        vis_image = self.image.copy()
        
        # Create mask overlay
        mask_image = np.zeros_like(vis_image)
        
        # Draw each mask with a different color
        for i, (mask, score) in enumerate(zip(masks, scores)):
            color = np.random.random(3) * 255
            mask_bool = mask.astype(bool)
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[mask_bool] = color
            
            # Add mask to visualization
            cv2.addWeighted(vis_image, 1, mask_overlay, 0.5, 0, vis_image)
            
            # Add score text
            score_text = f"Mask {i}: {score:.3f}"
            cv2.putText(vis_image, score_text,
                       (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2)
        
        # Draw points if provided
        if points is not None and labels is not None:
            for point, label in zip(points, labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(vis_image, tuple(map(int, point)), 5, color, -1)
                cv2.circle(vis_image, tuple(map(int, point)), 6, (255, 255, 255), 1)
        
        # Draw box if provided
        if box is not None:
            cv2.rectangle(vis_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0), 2)
        
        return vis_image 