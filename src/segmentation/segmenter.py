import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import datetime
import json
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import logging
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .spatial_analyzer import GeminiSpatialUnderstanding

logger = logging.getLogger(__name__)

class SemanticSegmenter:
    def __init__(self, model_name="facebook/sam2.1-hiera-large", device="cuda", batch_size=64):
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info("Initializing SAM2 model...")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        
        # Initialize with default settings
        self.mask_generator = self._create_mask_generator()
        
        self.image = None
        self.current_masks = None
        self.current_scores = None
        
    def _create_mask_generator(self, config=None):
        """Create mask generator with settings optimized for complete object segmentation"""
        # Optimized settings for larger, complete objects
        default_config = {
            'points_per_side': 48,          # Higher density for better detail
            'points_per_batch': self.batch_size,  # Balanced batch size
            'pred_iou_thresh': 0.97,        # Very high to ensure accurate boundaries
            'stability_score_thresh': 0.98,  # High stability to prevent over-segmentation
            'stability_score_offset': 0.8,   # Higher offset for more stable masks
            'crop_n_layers': 1,             # Single layer since objects are of similar scale
            'box_nms_thresh': 0.45,         # Lower NMS to treat shelf as one unit
            'min_mask_region_area': 500.0,  # Higher minimum area to avoid small segments
            'use_m2m': True                 # Enable mask refinement for better boundaries
        }
        
        # Use provided config or default
        config = config or default_config
        
        logger.info("Initializing mask generator with settings:")
        for key, value in config.items():
            logger.info(f"- {key}: {value}")
        
        return SAM2AutomaticMaskGenerator.from_pretrained(
            self.model_name,
            **config
        )
        
    def set_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> None:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            self.image = np.array(image)
        else:
            self.image = image
            
    def segment_automatic(self, max_objects: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Perform automatic segmentation using the mask generator"""
        if self.image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        logger.info("Starting automatic segmentation...")
        logger.info(f"Input image shape: {self.image.shape}")
        
        # Generate masks
        masks = self.generate_masks(self.image)
        
        # Convert to format expected by rest of pipeline
        mask_array = np.stack([mask['segmentation'] for mask in masks[:max_objects]])
        scores = np.array([mask['predicted_iou'] for mask in masks[:max_objects]])
        
        self.current_masks = mask_array
        self.current_scores = scores
        
        return mask_array, scores
        
    def generate_masks(self, image, monitor_memory=True):
        logger.info("Starting mask generation...")
        logger.info(f"Input image shape: {image.shape}")
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Resize image if needed and get scale
        resized_image, scale = self._resize_image(image)
        if scale != 1.0:
            logger.info(f"Resized image from {(original_height, original_width)} to {resized_image.shape[:2]}")
        
        try:
            logger.info("Generating masks with SAM2...")
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                masks = self.mask_generator.generate(resized_image)
                logger.info(f"Generated {len(masks)} initial masks")
                
                # Resize masks back to original dimensions if needed
                if scale != 1.0:
                    logger.info("Rescaling masks to original dimensions...")
                    for i, mask in enumerate(masks):
                        mask['segmentation'] = cv2.resize(
                            mask['segmentation'].astype(np.uint8), 
                            (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        mask['bbox'] = [
                            int(coord / scale) for coord in mask['bbox']
                        ]
                        mask['area'] = int(mask['area'] / (scale * scale))
                    logger.info("Mask rescaling complete")
                
        except RuntimeError as e:
            logger.error(f"Error during mask generation: {str(e)}")
            raise e
        
        logger.info("Mask generation completed successfully")
        return masks
        
    def _resize_image(self, image, max_size=1024):
        """Resize image if larger than max_size while maintaining aspect ratio"""
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height)), scale
        return image, 1.0
        
    def save_results(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Any]:
        if self.current_masks is None:
            raise ValueError("No segmentation results available. Run segment_automatic() first.")
            
        output_dir = Path(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"segmentation_output_{timestamp}"
        
        # Create directories
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "segments").mkdir(exist_ok=True)
        (result_dir / "masks").mkdir(exist_ok=True)
        
        metadata = {
            "timestamp": timestamp,
            "num_segments": len(self.current_masks),
            "scores": self.current_scores.tolist(),
            "segments": [],
            "masks": []
        }
        
        # Save individual segments and masks
        for i, (mask, score) in enumerate(zip(self.current_masks, self.current_scores)):
            # Save segment (masked image)
            segment = self.image.copy()
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
        vis_image = self.visualize_segmentation(self.image, self.current_masks)
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        metadata["visualization_path"] = str(vis_path)
        
        # Save metadata
        metadata_path = result_dir / f"{prefix}metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    def visualize_segmentation(self, image, masks):
        """Visualize segmentation with proper dimension handling"""
        overlay = image.copy()
        
        # Create random colors for each mask
        colors = [np.concatenate([np.random.random(3), [0.35]]) for _ in range(len(masks))]
        
        # Apply masks with colors
        for mask, color in zip(masks, colors):
            mask_image = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
            mask_image[mask] = (np.array(color) * 255).astype(np.uint8)
            
            # Blend the mask with the image
            alpha = mask_image[:, :, 3:] / 255.0
            overlay = overlay * (1 - alpha) + mask_image[:, :, :3] * alpha
            
            # Draw contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color[:3] * 255, 2)
        
        return overlay.astype(np.uint8) 
        
    def segment_with_gemini(self, 
                          gemini_api_key: str,
                          max_items: int = 10,
                          point_offset_ratio: float = 0.2,
                          temperature: float = 0.5,
                          debug: bool = False,
                          mode: str = "2d") -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation using Gemini's spatial understanding to guide SAM2
        
        Args:
            gemini_api_key: API key for Gemini
            max_items: Maximum number of objects to detect
            point_offset_ratio: Ratio of bounding box size to offset corner points (0.2 = 20%)
            temperature: Temperature parameter for Gemini generation
            debug: Whether to save debug visualizations
            mode: Either "2d" or "3d" box detection mode
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
        """
        if self.image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        logger.info(f"Starting Gemini-guided segmentation with {mode} mode")
        
        # Initialize Gemini spatial understanding
        spatial_analyzer = GeminiSpatialUnderstanding(gemini_api_key)
        logger.info("Initialized Gemini spatial analyzer")
        
        # Convert numpy array to PIL Image for Gemini
        pil_image = Image.fromarray(self.image)
        
        # Get boxes from Gemini based on mode
        logger.info(f"Requesting {mode} box detection from Gemini (max_items={max_items})")
        if mode == "2d":
            boxes = spatial_analyzer.detect_2d_boxes(
                pil_image,
                max_items=max_items,
                temperature=temperature
            )
        else:  # 3d mode
            boxes = spatial_analyzer.detect_3d_boxes(
                pil_image,
                max_items=max_items,
                temperature=temperature
            )
        logger.info(f"Received {len(boxes)} {mode} boxes from Gemini")

        # Process boxes to generate points
        all_points = []
        for box_data in boxes:
            if mode == "2d":
                # Extract coordinates from 2D box
                y1, x1, y2, x2 = box_data['box_2d']
            else:
                # Extract coordinates from 3D box center and size
                box_3d = np.array(box_data['box_3d'])
                x_center, y_center = box_3d[0], box_3d[1]
                x_size, y_size = box_3d[3], box_3d[4]
                x1 = max(0, x_center - x_size/2)
                y1 = max(0, y_center - y_size/2)
                x2 = min(self.image.shape[1], x_center + x_size/2)
                y2 = min(self.image.shape[0], y_center + y_size/2)
            
            # Calculate point offsets
            width = x2 - x1
            height = y2 - y1
            offset_x = width * point_offset_ratio
            offset_y = height * point_offset_ratio
            
            # Generate 5 points: corners + center with offset
            points = np.array([
                [x1 + offset_x, y1 + offset_y],  # Top-left
                [x2 - offset_x, y1 + offset_y],  # Top-right
                [x1 + offset_x, y2 - offset_y],  # Bottom-left
                [x2 - offset_x, y2 - offset_y],  # Bottom-right
                [(x1 + x2)/2, (y1 + y2)/2]      # Center
            ])
            all_points.append(points)

        # Save visualization before starting segmentation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = Path("debug") / f"gemini_debug_{timestamp}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Gemini analysis visualization with boxes and points
        analysis_path = debug_dir / "gemini_analysis.png"
        spatial_analyzer.visualize_analysis(
            self.image, boxes, all_points, 
            save_path=analysis_path, mode=mode
        )
        logger.info(f"Saved Gemini analysis visualization to {analysis_path}")
        
        # Save box data
        with open(debug_dir / "gemini_boxes.json", "w") as f:
            json.dump(boxes, f, indent=2)
        logger.info("Saved Gemini box data to JSON")

        # Set image for predictor
        logger.info("Setting image for SAM2 predictor")
        self.predictor.set_image(self.image)
        
        # Process each box with SAM2
        all_masks = []
        all_scores = []
        
        for i, (box_data, points) in enumerate(zip(boxes, all_points)):
            logger.info(f"Processing box {i+1}/{len(boxes)}: {box_data['label']}")
            
            if mode == "2d":
                y1, x1, y2, x2 = box_data['box_2d']
            else:
                box_3d = np.array(box_data['box_3d'])
                x_center, y_center = box_3d[0], box_3d[1]
                x_size, y_size = box_3d[3], box_3d[4]
                x1 = max(0, x_center - x_size/2)
                y1 = max(0, y_center - y_size/2)
                x2 = min(self.image.shape[1], x_center + x_size/2)
                y2 = min(self.image.shape[0], y_center + y_size/2)
            
            # All points are foreground
            labels = np.ones(len(points))
            
            # Predict masks using points and box
            logger.info(f"Predicting mask with {len(points)} points")
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=np.array([x1, y1, x2, y2])[None, :],
                multimask_output=True
            )
            
            # Keep only the best mask
            best_idx = np.argmax(scores)
            all_masks.append(masks[best_idx].astype(bool))
            all_scores.append(scores[best_idx])
            logger.info(f"Selected best mask with score {scores[best_idx]:.3f}")
        
        # Combine results
        masks = np.stack(all_masks)
        scores = np.array(all_scores)
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        
        self.current_masks = masks
        self.current_scores = scores
        
        logger.info(f"Completed segmentation with {len(masks)} masks")
        return masks, scores
        
    def visualize_gemini_analysis(self,
                                boxes_3d: List[Dict],
                                points: Optional[List[np.ndarray]] = None,
                                save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Visualize Gemini's spatial analysis results with 3D boxes and selected points
        
        Args:
            boxes_3d: List of 3D box data from Gemini
            points: Optional list of point arrays for each box
            save_path: Optional path to save the visualization
            
        Returns:
            visualization: RGB image with overlaid analysis
        """
        if self.image is None:
            raise ValueError("No image has been set")
            
        vis_image = self.image.copy()
        height, width = vis_image.shape[:2]
        
        # Camera intrinsics (assuming standard perspective)
        f = width / (2 * np.tan(60/2 * np.pi/180))  # 60 degree FOV
        cx = width/2
        cy = height/2
        intrinsics = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        
        # Draw boxes and labels
        for i, box_data in enumerate(boxes_3d):
            box_3d = np.array(box_data['box_3d'])
            label = box_data['label']
            
            # Extract center, size and rotation
            center = box_3d[:3]
            size = box_3d[3:6]
            rpy = box_3d[6:] * np.pi / 180  # Convert to radians
            
            # Create rotation matrix from Euler angles
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rpy[0]), -np.sin(rpy[0])],
                          [0, np.sin(rpy[0]), np.cos(rpy[0])]])
            Ry = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])],
                          [0, 1, 0],
                          [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
            Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0],
                          [np.sin(rpy[2]), np.cos(rpy[2]), 0],
                          [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            # Generate box corners
            half_size = size / 2
            corners = []
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        corner = center + R @ (half_size * np.array([sx, sy, sz]))
                        corners.append(corner)
            corners = np.array(corners)
            
            # Project corners to image plane
            points_2d = []
            for corner in corners:
                point = intrinsics @ corner
                points_2d.append([point[0]/point[2], point[1]/point[2]])
            points_2d = np.array(points_2d)
            
            # Draw box edges
            edges = [(0,1), (1,3), (3,2), (2,0),  # Bottom face
                    (4,5), (5,7), (7,6), (6,4),  # Top face
                    (0,4), (1,5), (2,6), (3,7)]  # Vertical edges
            
            color = (0, 255, 0)  # Green color for boxes
            for start, end in edges:
                pt1 = tuple(points_2d[start].astype(int))
                pt2 = tuple(points_2d[end].astype(int))
                cv2.line(vis_image, pt1, pt2, color, 2)
            
            # Draw label at center of top face
            top_center = np.mean(points_2d[4:8], axis=0).astype(int)
            cv2.putText(vis_image, f"{i+1}. {label}", 
                       tuple(top_center), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            
            # Draw points if provided
            if points is not None and i < len(points):
                # Draw lines connecting points to show the order
                point_coords = points[i].astype(int)
                for j in range(len(point_coords)):
                    # Draw point with index
                    pt = tuple(point_coords[j])
                    cv2.drawMarker(vis_image, pt, (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
                    cv2.circle(vis_image, pt, 5, (255, 0, 0), -1)
                    cv2.putText(vis_image, str(j+1), 
                              (pt[0]+5, pt[1]-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # Draw line to next point
                    if j < len(point_coords) - 1:
                        next_pt = tuple(point_coords[j+1])
                        cv2.line(vis_image, pt, next_pt, (255, 0, 0), 1, cv2.LINE_AA)
        
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image 
        
    def segment_with_points(self, 
                          gemini_api_key: str,
                          max_items: int = 10,
                          temperature: float = 0.5,
                          debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation using Gemini's point detection to guide SAM2
        
        Args:
            gemini_api_key: API key for Gemini
            max_items: Maximum number of items to detect
            temperature: Temperature parameter for Gemini generation
            debug: Whether to save debug visualizations
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
        """
        if self.image is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        logger.info("Starting Gemini-guided point-based segmentation")
        
        # Initialize Gemini spatial understanding
        spatial_analyzer = GeminiSpatialUnderstanding(gemini_api_key)
        logger.info("Initialized Gemini spatial analyzer")
        
        # Convert numpy array to PIL Image for Gemini
        pil_image = Image.fromarray(self.image)
        
        # Get points from Gemini
        logger.info(f"Requesting point detection from Gemini (max_items={max_items})")
        points_data = spatial_analyzer.point_to_items(
            pil_image,
            max_items=max_items,
            temperature=temperature
        )
        logger.info(f"Received {len(points_data)} points from Gemini")

        # Save visualization before starting segmentation
        if debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = Path("debug") / f"gemini_debug_{timestamp}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save points visualization
            analysis_path = debug_dir / "gemini_points.png"
            self._visualize_points(self.image, points_data, save_path=analysis_path)
            logger.info(f"Saved Gemini points visualization to {analysis_path}")
            
            # Save points data
            with open(debug_dir / "gemini_points.json", "w") as f:
                json.dump(points_data, f, indent=2)
            logger.info("Saved Gemini points data to JSON")

        # Set image for predictor
        logger.info("Setting image for SAM2 predictor")
        self.predictor.set_image(self.image)
        
        # Process each point with SAM2
        all_masks = []
        all_scores = []
        
        for i, point_data in enumerate(points_data):
            logger.info(f"Processing point {i+1}/{len(points_data)}: {point_data['label']}")
            
            # Extract point coordinates
            point = np.array([point_data['point']])  # Shape: (1, 2) for [y, x]
            
            # Point is foreground
            labels = np.ones(1)  # Single point
            
            # Predict masks using point
            logger.info("Predicting mask with point prompt")
            masks, scores, _ = self.predictor.predict(
                point_coords=point,
                point_labels=labels,
                multimask_output=True
            )
            
            # Keep only the best mask
            best_idx = np.argmax(scores)
            all_masks.append(masks[best_idx].astype(bool))
            all_scores.append(scores[best_idx])
            logger.info(f"Selected best mask with score {scores[best_idx]:.3f}")
        
        # Combine results
        if not all_masks:
            logger.warning("No masks were generated")
            return np.array([]), np.array([])
            
        masks = np.stack(all_masks)
        scores = np.array(all_scores)
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        
        self.current_masks = masks
        self.current_scores = scores
        
        logger.info(f"Completed segmentation with {len(masks)} masks")
        return masks, scores
        
    def _visualize_points(self,
                         image: np.ndarray,
                         points_data: List[Dict],
                         save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Helper method to visualize points from Gemini"""
        vis_image = image.copy()
        
        # Draw points and labels
        for i, point_data in enumerate(points_data):
            # Extract coordinates
            y, x = point_data['point']
            label = point_data['label']
            
            # Draw point
            pt = (int(x), int(y))
            cv2.drawMarker(vis_image, pt, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(vis_image, pt, 8, (0, 255, 0), -1)
            
            # Draw label
            cv2.putText(vis_image, f"{i+1}. {label}", 
                       (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image 