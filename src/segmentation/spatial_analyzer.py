from typing import List, Dict, Union, Any, Optional
from PIL import Image
import json
from google import genai
from google.genai import types
import numpy as np
import logging
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

class GeminiSpatialUnderstanding:
    """
    A class to handle spatial understanding features of Gemini 2.0 Flash
    Including pointing and 3D spatial understanding capabilities
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Gemini client with API key
        
        Args:
            api_key (str): Google API key for Gemini access
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-exp"
        
    def _resize_image(self, image: Image.Image, target_width: int = 800) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image (PIL.Image): Input image
            target_width (int): Desired width in pixels
            
        Returns:
            PIL.Image: Resized image
        """
        aspect_ratio = image.size[1] / image.size[0]
        target_height = int(target_width * aspect_ratio)
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def point_to_items(self, 
                      image: Image.Image, 
                      max_items: int = 10, 
                      temperature: float = 0.5) -> List[Dict[str, Any]]:
        """
        Point to items in the image using Gemini's pointing capability
        
        Args:
            image: PIL Image to analyze (should be original size)
            max_items: Maximum number of items to detect
            temperature: Temperature parameter for generation
            
        Returns:
            List[Dict]: List of detected items with points and labels
        """
        logger.info("Requesting point detection from Gemini")
        
        prompt = f"""
        Point to no more than {max_items} items in the image.
        Output a json list where each entry contains:
        1. "label": descriptive name of the object
        2. "point": [y, x] coordinates where:
           - y is the vertical coordinate (0 at top)
           - x is the horizontal coordinate (0 at left)
           - coordinates should be in original image dimensions
        Example format:
        [
          {{"label": "object name", "point": [y, x]}},
          ...
        ]
        """
        
        system_instructions = """
        Return points as a JSON array with labels. Never return masks or code fencing.
        If an object is present multiple times, name them according to their unique characteristic 
        (colors, size, position, unique characteristics, etc.).
        Use the original image dimensions for coordinates.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                temperature=temperature
            )
        )
        
        # Sanitize response text by removing code block delimiters
        text = response.text.strip()
        if text.startswith("```"):
            # Remove opening delimiter and any language specification
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            # Remove closing delimiter
            text = text.rsplit("\n", 1)[0]
            
        # Remove any remaining json or other language tags
        text = text.replace("```json", "").replace("```", "").strip()
        
        logger.info(f"Sanitized response: {text}")
        
        # Parse response text to get points
        points_data = json.loads(text)
        logger.info(f"Received {len(points_data)} points from Gemini")
        
        # Validate coordinates are within image bounds
        width, height = image.size
        for point in points_data:
            y, x = point['point']
            point['point'] = [
                max(0, min(y, height-1)),
                max(0, min(x, width-1))
            ]
        
        return points_data

    def detect_3d_boxes(self, 
                       image: Image.Image, 
                       max_items: int = 10, 
                       temperature: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect 3D bounding boxes of objects in the image
        
        Args:
            image: PIL Image to analyze (should be original size)
            max_items: Maximum number of items to detect
            temperature: Temperature parameter for generation
            
        Returns:
            List[Dict]: List of detected items with 3D bounding box parameters
        """
        prompt = f"""
        Detect the 3D bounding boxes of no more than {max_items} items.
        Output a json list where each entry contains:
        1. "label": descriptive name of the object
        2. "box_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
           - coordinates should be in original image dimensions
           - center coordinates (x,y) should be within image bounds
           - sizes should be proportional to image dimensions
           - angles in degrees
        """
        
        system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing.
        If an object is present multiple times, name them according to their unique characteristic 
        (colors, size, position, unique characteristics, etc.).
        Use the original image dimensions for coordinates and sizes.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                temperature=temperature
            )
        )
        
        # Sanitize response text by removing code block delimiters
        text = response.text.strip()
        if text.startswith("```"):
            # Remove opening delimiter and any language specification
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            # Remove closing delimiter
            text = text.rsplit("\n", 1)[0]
            
        # Remove any remaining json or other language tags
        text = text.replace("```json", "").replace("```", "").strip()
        
        logger.info(f"Sanitized response: {text}")
        
        # Parse and validate boxes
        boxes_data = json.loads(text)
        
        # Validate coordinates are within image bounds
        width, height = image.size
        for box in boxes_data:
            box_3d = box['box_3d']
            # Clamp center coordinates to image bounds
            box_3d[0] = max(0, min(box_3d[0], width-1))   # x_center
            box_3d[1] = max(0, min(box_3d[1], height-1))  # y_center
            # Ensure sizes are positive and reasonable
            box_3d[3:6] = [abs(s) for s in box_3d[3:6]]
            box['box_3d'] = box_3d
            
        logger.info(f"Received {len(boxes_data)} 3D boxes from Gemini")
        return boxes_data

    def search_3d_boxes(self, 
                       image: Image.Image, 
                       search_items: List[str],
                       temperature: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for specific items and detect their 3D bounding boxes
        
        Args:
            image (PIL.Image): Input image
            search_items (List[str]): List of items to search for
            temperature (float): Temperature parameter for generation
            
        Returns:
            List[Dict]: List of detected items with 3D bounding box parameters
        """
        items_str = ", ".join(search_items)
        prompt = f"""
        Detect the 3D bounding boxes of {items_str}.
        Output a json list where each entry contains the object name in "label" and its 3D bounding box in "box_3d"
        The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image, prompt],
            config=types.GenerateContentConfig(temperature=temperature)
        )
        print(response.text)
        return json.loads(response.text) 

    def detect_2d_boxes(self,
                      pil_image: Image.Image,
                      max_items: int = 25,
                      temperature: float = 0.5) -> List[Dict]:
        """
        Detect 2D bounding boxes in the image using Gemini
        
        Args:
            pil_image: PIL Image to analyze (should be original size)
            max_items: Maximum number of items to detect
            temperature: Temperature for generation
            
        Returns:
            List of dictionaries containing box_2d coordinates and labels
        """
        logger.info("Requesting 2D box detection from Gemini")
        
        prompt = f"""
        Detect the 2D bounding boxes of no more than {max_items} items.
        Output a json list where each entry contains:
        1. "label": descriptive name of the object
        2. "box_2d": [y1, x1, y2, x2] coordinates where:
           - (x1, y1) is the top-left corner
           - (x2, y2) is the bottom-right corner
           - coordinates should be in original image dimensions
        Example format:
        [
          {{"label": "object name", "box_2d": [y1, x1, y2, x2]}},
          ...
        ]
        """
        
        system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing.
        If an object is present multiple times, name them according to their unique characteristic 
        (colors, size, position, unique characteristics, etc.).
        Ensure coordinates are within image bounds and in the correct order [y1, x1, y2, x2].
        Use the original image dimensions for coordinates.
        """
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                temperature=temperature
            )
        )
        
        # Sanitize response text by removing code block delimiters
        text = response.text.strip()
        if text.startswith("```"):
            # Remove opening delimiter and any language specification
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            # Remove closing delimiter
            text = text.rsplit("\n", 1)[0]
            
        # Remove any remaining json or other language tags
        text = text.replace("```json", "").replace("```", "").strip()
        
        logger.info(f"Sanitized response: {text}")
        
        # Parse response text to get boxes
        boxes_data = json.loads(text)
        logger.info(f"Received {len(boxes_data)} 2D boxes from Gemini")
        
        # Validate coordinates are within image bounds
        width, height = pil_image.size
        for box in boxes_data:
            y1, x1, y2, x2 = box['box_2d']
            box['box_2d'] = [
                max(0, min(y1, height-1)),
                max(0, min(x1, width-1)),
                max(0, min(y2, height-1)),
                max(0, min(x2, width-1))
            ]
        
        return boxes_data
        
    def visualize_analysis(self,
                         image: np.ndarray,
                         boxes: List[Dict],
                         points: Optional[List[np.ndarray]] = None,
                         save_path: Optional[Union[str, Path]] = None,
                         mode: str = "2d") -> np.ndarray:
        """
        Visualize spatial analysis results with either 2D or 3D boxes
        
        Args:
            image: RGB image to draw on
            boxes: List of box data from Gemini (2D or 3D)
            points: Optional list of point arrays for each box
            save_path: Optional path to save visualization
            mode: Either "2d" or "3d" visualization mode
            
        Returns:
            Visualization image with boxes and labels
        """
        if mode == "2d":
            return self._visualize_2d_boxes(image, boxes, points, save_path)
        elif mode == "3d":
            return self._visualize_3d_boxes(image, boxes, points, save_path)
        else:
            raise ValueError(f"Invalid visualization mode: {mode}")
            
    def _visualize_2d_boxes(self,
                          image: np.ndarray,
                          boxes_2d: List[Dict],
                          points: Optional[List[np.ndarray]] = None,
                          save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Internal method for 2D box visualization"""
        vis_image = image.copy()
        
        # Draw boxes and labels
        for i, box_data in enumerate(boxes_2d):
            # Extract coordinates
            y1, x1, y2, x2 = box_data['box_2d']
            label = box_data['label']
            
            # Draw box
            color = (0, 255, 0)  # Green for boxes
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(vis_image, f"{i+1}. {label}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            
            # Draw points if provided
            if points is not None and i < len(points):
                self._draw_points_and_lines(vis_image, points[i])
        
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image
        
    def _visualize_3d_boxes(self,
                          image: np.ndarray,
                          boxes_3d: List[Dict],
                          points: Optional[List[np.ndarray]] = None,
                          save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Internal method for 3D box visualization"""
        vis_image = image.copy()
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
            
            # Create rotation matrices
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
                self._draw_points_and_lines(vis_image, points[i])
        
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image
        
    def _draw_points_and_lines(self, image: np.ndarray, points: np.ndarray) -> None:
        """Helper method to draw points and connecting lines"""
        point_coords = points.astype(int)
        for j in range(len(point_coords)):
            # Draw point with index
            pt = tuple(point_coords[j])
            cv2.drawMarker(image, pt, (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.circle(image, pt, 5, (255, 0, 0), -1)
            cv2.putText(image, str(j+1), 
                      (pt[0]+5, pt[1]-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw line to next point
            if j < len(point_coords) - 1:
                next_pt = tuple(point_coords[j+1])
                cv2.line(image, pt, next_pt, (255, 0, 0), 1, cv2.LINE_AA) 