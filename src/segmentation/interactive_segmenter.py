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
    
    def __init__(self, checkpoint_path: Union[str, Path], model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"):
        """
        Initialize the SAM2 model and predictor
        
        Args:
            checkpoint_path: Path to the SAM2 model checkpoint
            model_cfg: Path to the model configuration file
        """
        # Select device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Use bfloat16 for better performance
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        self.device = device
        self.model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.model)
        self._current_image = None
        
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Set the current image for segmentation
        
        Args:
            image: Input image (PIL Image or numpy array)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        self._current_image = image
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
        if self._current_image is None:
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
        if self._current_image is None:
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
        
    def visualize_masks(self, 
                       masks: np.ndarray,
                       scores: np.ndarray,
                       points: Optional[np.ndarray] = None,
                       labels: Optional[np.ndarray] = None,
                       box: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a visualization of the predicted masks
        
        Args:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
            points: Optional point coordinates
            labels: Optional point labels
            box: Optional box coordinates
            
        Returns:
            visualization: RGB image with overlaid masks and prompts
        """
        if self._current_image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        vis_image = self._current_image.copy()
        
        # Overlay masks with different colors
        for mask, score in zip(masks, scores):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1) * color.reshape(1, 1, -1)
            
            # Draw contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(vis_image, contours, -1, color[:3].tolist(), thickness=2)
            
            # Blend mask
            vis_image = cv2.addWeighted(vis_image, 1, (mask_image * 255).astype(np.uint8), 0.5, 0)
            
        # Draw points if provided
        if points is not None and labels is not None:
            for point, label in zip(points, labels):
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.drawMarker(vis_image, tuple(point.astype(int)), color, 
                             markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                
        # Draw box if provided
        if box is not None:
            cv2.rectangle(vis_image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 255, 0), 2)
            
        return vis_image 