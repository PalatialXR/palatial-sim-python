from typing import List, Dict, Union, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import json
import datetime
import os
import logging
from .interactive_segmenter import InteractiveSegmenter
from .spatial_analyzer import GeminiSpatialUnderstanding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticSegmenter:
    """
    A class to handle semantic segmentation with support for both automatic and interactive modes
    """
    
    def __init__(self, sam2_checkpoint: Union[str, Path], model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"):
        """
        Initialize the segmenter
        
        Args:
            sam2_checkpoint: Path to the SAM2 model checkpoint
            model_cfg: Path to the model configuration file
        """
        self.interactive_segmenter = InteractiveSegmenter(sam2_checkpoint, model_cfg)
        self._current_image = None
        self._current_masks = None
        self._current_scores = None
        self._current_logits = None
        
    def set_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> None:
        """
        Set the current image for segmentation
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            self._current_image = np.array(image)
        else:
            self._current_image = image
            
        self.interactive_segmenter.set_image(self._current_image)
        
    def segment_interactive(self,
                          points: Optional[np.ndarray] = None,
                          labels: Optional[np.ndarray] = None,
                          box: Optional[np.ndarray] = None,
                          multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform interactive segmentation using points and/or boxes
        
        Args:
            points: Optional Nx2 array of point coordinates (x, y)
            labels: Optional N array of point labels (1 for foreground, 0 for background)
            box: Optional 4-element array for bounding box in xyxy format
            multimask_output: Whether to return multiple masks per prompt
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
        """
        if points is not None and labels is not None:
            masks, scores, logits = self.interactive_segmenter.predict_from_points(
                points, labels, box, multimask_output
            )
        elif box is not None:
            masks, scores, logits = self.interactive_segmenter.predict_from_boxes(
                box[None, :], multimask_output
            )
        else:
            raise ValueError("Must provide either points with labels or a box")
            
        self._current_masks = masks
        self._current_scores = scores
        self._current_logits = logits
        
        return masks, scores
        
    def segment_automatic(self, max_objects: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform automatic segmentation using grid-based points
        
        Args:
            max_objects: Maximum number of objects to detect
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
        """
        # Create a grid of points
        h, w = self._current_image.shape[:2]
        grid_size = int(np.sqrt(max_objects))
        x = np.linspace(w * 0.1, w * 0.9, grid_size)
        y = np.linspace(h * 0.1, h * 0.9, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        labels = np.ones(len(points))
        
        # Predict masks for each point
        all_masks = []
        all_scores = []
        
        for i in range(0, len(points), 4):  # Process in batches of 4
            batch_points = points[i:i+4]
            batch_labels = labels[i:i+4]
            
            masks, scores, _ = self.interactive_segmenter.predict_from_points(
                batch_points, batch_labels, multimask_output=True
            )
            
            # Keep only the highest scoring mask for each point
            best_masks = masks[np.arange(len(masks)), np.argmax(scores, axis=1)]
            best_scores = np.max(scores, axis=1)
            
            all_masks.append(best_masks)
            all_scores.append(best_scores)
            
        # Combine results
        masks = np.concatenate(all_masks)
        scores = np.concatenate(all_scores)
        
        # Sort by score and keep top max_objects
        sorted_idx = np.argsort(scores)[::-1][:max_objects]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        
        self._current_masks = masks
        self._current_scores = scores
        
        return masks, scores
        
    def save_results(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Any]:
        """
        Save segmentation results
        
        Args:
            output_dir: Directory to save results
            prefix: Optional prefix for output files
            
        Returns:
            metadata: Dictionary containing paths to saved files and metadata
        """
        if self._current_masks is None:
            raise ValueError("No segmentation results available. Run segment_interactive() or segment_automatic() first.")
            
        output_dir = Path(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"segmentation_output_{timestamp}"
        
        # Create directories
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "segments").mkdir(exist_ok=True)
        (result_dir / "masks").mkdir(exist_ok=True)
        
        metadata = {
            "timestamp": timestamp,
            "num_segments": len(self._current_masks),
            "scores": self._current_scores.tolist(),
            "segments": [],
            "masks": []
        }
        
        # Save individual segments and masks
        for i, (mask, score) in enumerate(zip(self._current_masks, self._current_scores)):
            # Save segment (masked image)
            segment = self._current_image.copy()
            segment[~mask] = 0
            segment_path = result_dir / "segments" / f"{prefix}segment_{i:04d}.png"
            cv2.imwrite(str(segment_path), cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
            
            # Save binary mask
            mask_path = result_dir / "masks" / f"{prefix}mask_{i:04d}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
            
            metadata["segments"].append({
                "id": i,
                "path": str(segment_path),
                "score": float(score)
            })
            metadata["masks"].append({
                "id": i,
                "path": str(mask_path),
                "score": float(score)
            })
            
        # Save visualization
        vis_path = result_dir / f"{prefix}segmentation_visualization.png"
        vis_image = self.interactive_segmenter.visualize_masks(
            self._current_masks,
            self._current_scores
        )
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        metadata["visualization_path"] = str(vis_path)
        
        # Save metadata
        metadata_path = result_dir / f"{prefix}metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return metadata 
        
    def visualize_gemini_analysis(self,
                                boxes_3d: List[Dict],
                                points: Optional[List[np.ndarray]] = None,
                                save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Visualize Gemini's spatial analysis results
        
        Args:
            boxes_3d: List of 3D box data from Gemini
            points: Optional list of point arrays for each box
            save_path: Optional path to save the visualization
            
        Returns:
            visualization: RGB image with overlaid analysis
        """
        if self._current_image is None:
            raise ValueError("No image has been set")
            
        vis_image = self._current_image.copy()
        
        # Draw boxes and labels
        for i, box_data in enumerate(boxes_3d):
            box_3d = np.array(box_data['box_3d'])
            label = box_data['label']
            
            # Extract 2D box
            x_center, y_center = box_3d[0], box_3d[1]
            x_size, y_size = box_3d[3], box_3d[4]
            x1 = max(0, int(x_center - x_size/2))
            y1 = max(0, int(y_center - y_size/2))
            x2 = min(vis_image.shape[1], int(x_center + x_size/2))
            y2 = min(vis_image.shape[0], int(y_center + y_size/2))
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(vis_image, f"{i+1}. {label}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw points if provided
            if points is not None and i < len(points):
                for point in points[i]:
                    cv2.drawMarker(vis_image, tuple(point.astype(int)),
                                 (255, 0, 0), cv2.MARKER_STAR, 20, 2)
        
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image
    
    def segment_with_gemini(self, 
                           gemini_api_key: str,
                           max_items: int = 10,
                           point_offset_ratio: float = 0.2,
                           temperature: float = 0.5,
                           debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation using Gemini's spatial understanding to guide SAM2
        
        Args:
            gemini_api_key: API key for Gemini
            max_items: Maximum number of objects to detect
            point_offset_ratio: Ratio of bounding box size to offset corner points (0.2 = 20%)
            temperature: Temperature parameter for Gemini generation
            debug: Whether to save debug visualizations
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
        """
        if self._current_image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        logger.info("Starting Gemini-guided segmentation")
        
        # Initialize Gemini spatial understanding
        spatial_analyzer = GeminiSpatialUnderstanding(gemini_api_key)
        logger.info("Initialized Gemini spatial analyzer")
        
        # Convert numpy array to PIL Image for Gemini
        pil_image = Image.fromarray(self._current_image)
        
        # Get 3D bounding boxes from Gemini
        logger.info(f"Requesting 3D box detection from Gemini (max_items={max_items})")
        boxes_3d = spatial_analyzer.detect_3d_boxes(
            pil_image,
            max_items=max_items,
            temperature=temperature
        )
        logger.info(f"Received {len(boxes_3d)} 3D boxes from Gemini")
        
        # Process each box with SAM2
        all_masks = []
        all_scores = []
        all_points = []
        
        for i, box_data in enumerate(boxes_3d):
            logger.info(f"Processing box {i+1}/{len(boxes_3d)}: {box_data['label']}")
            
            # Extract 2D bounding box from 3D box
            box_3d = np.array(box_data['box_3d'])
            x_center, y_center = box_3d[0], box_3d[1]
            x_size, y_size = box_3d[3], box_3d[4]
            
            # Convert to xyxy format
            x1 = max(0, x_center - x_size/2)
            y1 = max(0, y_center - y_size/2)
            x2 = min(self._current_image.shape[1], x_center + x_size/2)
            y2 = min(self._current_image.shape[0], y_center + y_size/2)
            box = np.array([x1, y1, x2, y2])
            
            # Calculate point offsets
            offset_x = (x2 - x1) * point_offset_ratio
            offset_y = (y2 - y1) * point_offset_ratio
            
            # Generate 5 points: corners + center with offset
            points = np.array([
                [x1 + offset_x, y1 + offset_y],  # Top-left
                [x2 - offset_x, y1 + offset_y],  # Top-right
                [x1 + offset_x, y2 - offset_y],  # Bottom-left
                [x2 - offset_x, y2 - offset_y],  # Bottom-right
                [x_center, y_center]             # Center
            ])
            all_points.append(points)
            
            # All points are foreground
            labels = np.ones(len(points))
            
            # Predict masks using points and box
            logger.info(f"Predicting mask with {len(points)} points")
            masks, scores, _ = self.segment_interactive(
                points=points,
                labels=labels,
                box=box,
                multimask_output=True
            )
            
            # Keep only the best mask
            best_idx = np.argmax(scores)
            all_masks.append(masks[best_idx])
            all_scores.append(scores[best_idx])
            logger.info(f"Selected best mask with score {scores[best_idx]:.3f}")
        
        # Save debug visualization if requested
        if debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = Path("debug") / f"gemini_debug_{timestamp}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save Gemini analysis visualization
            analysis_path = debug_dir / "gemini_analysis.png"
            self.visualize_gemini_analysis(boxes_3d, all_points, analysis_path)
            logger.info(f"Saved Gemini analysis visualization to {analysis_path}")
            
            # Save intermediate results
            with open(debug_dir / "gemini_boxes.json", "w") as f:
                json.dump(boxes_3d, f, indent=2)
        
        # Combine results
        masks = np.stack(all_masks)
        scores = np.array(all_scores)
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        
        self._current_masks = masks
        self._current_scores = scores
        
        logger.info(f"Completed segmentation with {len(masks)} masks")
        return masks, scores 