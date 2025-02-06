from typing import List, Dict, Union, Any, Optional
from typing import Any, Dict, List
from PIL import Image
import json
from google import generativeai as genai
from google.generativeai import types
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
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
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
        
        response = self.model.generate_content(
            contents=[prompt, image],
            generation_config=types.GenerationConfig(
                temperature=temperature
            )
        )
        
        # Sanitize response text by removing code block delimiters
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
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
        logger.info("Requesting 3D box detection from Gemini")
        
        # Fix image orientation
        if hasattr(image, '_getexif'):
            try:
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # 274 is the orientation tag
                    if orientation is not None:
                        # Rotate or flip the image based on EXIF orientation
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
            except Exception as e:
                logger.warning(f"Error handling EXIF orientation: {str(e)}")
        
        prompt = f"""
        Detect the 3D bounding boxes of no more than {max_items} items.
        Output a json list where each entry contains:
        1. "label": descriptive name of the object
        2. "box_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
           - x_center, y_center: center coordinates in image frame (0 to image width/height)
           - z_center: depth from camera (use 500 for close objects, 1000 for far objects)
           - x_size, y_size: dimensions in pixels (typically 50-500)
           - z_size: depth in same scale as z_center (typically 50-200)
           - roll, pitch, yaw: rotation angles in degrees
           All 9 values must be provided for each box.
           Make sure sizes are proportional to the actual objects.
        Example format:
        [
          {{"label": "monitor", "box_3d": [500, 300, 800, 400, 300, 100, 0, 0, 0]}},
          {{"label": "keyboard", "box_3d": [500, 600, 500, 300, 100, 50, 0, 0, 10]}}
        ]
        """
        
        response = self.model.generate_content(
            contents=[prompt, image],
            generation_config=types.GenerationConfig(
                temperature=temperature
            )
        )
        
        # Sanitize response text by removing code block delimiters
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
        text = text.replace("```json", "").replace("```", "").strip()
        
        logger.info(f"Sanitized response: {text}")
        
        try:
            # Parse and validate boxes
            boxes_data = json.loads(text)
            
            # Validate and fix box format
            width, height = image.size
            validated_boxes = []
            
            for box in boxes_data:
                if 'label' not in box or 'box_3d' not in box:
                    logger.warning(f"Skipping invalid box format: {box}")
                    continue
                    
                box_3d = box['box_3d']
                
                # If box_3d doesn't have 9 values, try to fix it
                if len(box_3d) != 9:
                    # Create a default 3D box
                    if len(box_3d) >= 4:  # If we at least have x, y, width, height
                        x, y, w, h = box_3d[:4]
                        default_box = [
                            x,                    # x_center
                            y,                    # y_center
                            500,                  # z_center (default depth)
                            max(w, 50),           # x_size (minimum 50 pixels)
                            max(h, 50),           # y_size (minimum 50 pixels)
                            max(min(w, h) / 2, 50),  # z_size (minimum 50)
                            0,                    # roll (default)
                            0,                    # pitch (default)
                            0                     # yaw (default)
                        ]
                        box_3d = default_box
                    else:
                        logger.warning(f"Cannot fix box format: {box}")
                        continue
                
                # Ensure all values are numeric
                box_3d = [float(v) for v in box_3d]
                
                # Clamp center coordinates to image bounds
                box_3d[0] = max(0, min(box_3d[0], width-1))   # x_center
                box_3d[1] = max(0, min(box_3d[1], height-1))  # y_center
                
                # Ensure z_center is reasonable
                box_3d[2] = max(box_3d[2], 500)  # Minimum z_center of 500
                
                # Ensure sizes are reasonable
                box_3d[3] = max(box_3d[3], 50)  # Minimum x_size of 50
                box_3d[4] = max(box_3d[4], 50)  # Minimum y_size of 50
                box_3d[5] = max(box_3d[5], 50)  # Minimum z_size of 50
                
                # Normalize angles to [-180, 180] degrees
                box_3d[6:] = [((angle + 180) % 360) - 180 for angle in box_3d[6:]]
                
                validated_boxes.append({
                    'label': box['label'],
                    'box_3d': box_3d
                })
            
            logger.info(f"Received {len(validated_boxes)} 3D boxes from Gemini")
            return validated_boxes
            
        except Exception as e:
            logger.error(f"Error processing Gemini response: {str(e)}")
            raise

    def search_3d_boxes(self, 
                       image: Image.Image, 
                       search_items: List[str] = None,
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
        img = Image.open(image)

        image_response = self.client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[
                img,
                """
                Detect the 3D bounding boxes of no more than 10 items. Use common sense to understand the scale of the objects.
                Output a json list where each entry contains the object name in "label" and its 3D bounding box in "box_3d"
                The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].
                """
            ],
            config = types.GenerateContentConfig(
                temperature=0.5
            )
        )

        # Check response
        print(image_response.text)

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
        
        response = self.model.generate_content(
            contents=[prompt, pil_image],
            generation_config=types.GenerationConfig(
                temperature=temperature
            )
        )
        
        # Sanitize response text by removing code block delimiters
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
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
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Convert boxes to JSON string
        boxes_json = json.dumps(boxes_3d, indent=2)
        
        # Generate HTML visualization
        html_content = self.generate_3d_box_html(pil_image, boxes_json)
        
        # Save HTML file in the same directory as the image output
        if save_path:
            output_dir = Path(save_path).parent
            html_path = output_dir / "3d_boxes.html"
            with open(html_path, "w") as f:
                f.write(html_content)
            
            # Save the image as well
            cv2.imwrite(str(save_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        return image

    def generate_3d_box_html(self, pil_image: Image.Image, boxes_json: str) -> str:
        """
        Generate HTML content for 3D box visualization
        
        Args:
            pil_image: PIL Image to visualize
            boxes_json: JSON string containing 3D box data
            
        Returns:
            str: HTML content for visualization
        """
        # Convert PIL image to base64 string
        import base64
        from io import BytesIO
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Box Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #fff;
            color: #000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        .view-container {{
            display: flex;
            gap: 20px;
            padding: 20px;
            flex-direction: column;
            align-items: center;
        }}

        .canvas-container {{
            display: flex;
            gap: 20px;
        }}

        .box-line {{
            position: absolute;
            background: #2962FF;
            transform-origin: 0 0;
            opacity: 1;
            box-shadow: 0 0 30px rgba(41, 98, 255, 0.4);
            transition: all 0.3s ease;
            pointer-events: none;
        }}

        .box-line.highlight {{
            background: #FF4081;
            box-shadow: 0 0 30px rgba(255, 64, 129, 0.4);
            z-index: 100;
            border-color: #FF4081 !important;
        }}

        .box-line.fade {{
            opacity: 0.3;
        }}

        .box-label {{
            position: absolute;
            color: white;
            font-size: 12px;
            font-family: Arial;
            transform: translate(-50%, -50%);
            opacity: 1;
            background: #2962FF;
            padding: 2px 8px;
            border-radius: 4px;
            box-shadow: 0 0 30px rgba(41, 98, 255, 0.4);
            transition: all 0.3s ease;
            cursor: pointer;
            z-index: 1000;
        }}

        .box-label.highlight {{
            background: #FF4081;
            box-shadow: 0 0 30px rgba(255, 64, 129, 0.4);
            transform: translate(-50%, -50%) scale(1.1);
            z-index: 1001;
        }}

        .box-label.fade {{
            opacity: 0.3;
        }}

        .box-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}

        .box-overlay .box-label {{
            pointer-events: auto;
        }}

        .controls {{
            margin-top: 10px;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .slider-label {{
            color: white;
            font-size: 12px;
        }}

        input[type="range"] {{
            width: 200px;
        }}

        #topView {{
            width: 500px;
            height: 500px;
            background: #fff;
            border: 1px solid #333;
            position: relative;
            overflow: hidden;
        }}

        .grid-line {{
            position: absolute;
            background: #333333;
            pointer-events: none;
        }}

        .grid-label {{
            position: absolute;
            color: #666666;
            font-size: 10px;
            pointer-events: none;
        }}

        .axis-line {{
            position: absolute;
            background: #666666;
            pointer-events: none;
        }}

        .camera-triangle {{
            position: absolute;
            width: 0;
            height: 0;
            border-left: 10px solid transparent;
            border-right: 10px solid transparent;
            border-bottom: 20px solid #0000ff;
            pointer-events: none;
        }}

        .top-view-container {{
            position: relative;
        }}
    </style>
</head>
<body>
    <div class="view-container">
        <div class="canvas-container">
            <div id="container" style="position: relative;">
                <canvas id="canvas" style="background: #000;"></canvas>
                <div id="boxOverlay" class="box-overlay"></div>
                <div class="controls">
                    <span class="slider-label">FOV:</span>
                    <input type="range" id="fovSlider" min="50" max="120" value="60" step="1">
                    <span id="fovValue">60</span>
                </div>
            </div>
            <div class="top-view-container">
                <div id="topView">
                    <div id="topViewOverlay" class="box-overlay"></div>
                </div>
                <div class="controls">
                    <span class="slider-label">Zoom:</span>
                    <input type="range" id="zoomSlider" min="0.5" max="3" value="1.5" step="0.1">
                    <span id="zoomValue">1.5x</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isDragging = {{left: false, right: false}};
        let lastX = 0;
        let lastY = 0;
        let panOffset = {{x: 0, y: 150}};
        let boxesData = {boxes_json};

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('container');
        const topView = document.getElementById('topView');
        const topViewOverlay = document.getElementById('topViewOverlay');

        // Load and draw the image
        const img = new Image();
        img.onload = () => {{
            const aspectRatio = img.height / img.width;
            canvas.height = 500;
            canvas.width = Math.round(500 / aspectRatio);
            container.style.width = canvas.width + 'px';
            container.style.height = canvas.height + 'px';

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            frame.width = canvas.width;
            frame.height = canvas.height;
            annotateFrame(frame, parseFloat(fovSlider.value));
        }};
        img.src = 'data:image/png;base64,{img_str}';

        function highlightBox(label, highlight) {{
            const boxOverlay = document.getElementById('boxOverlay');
            const topViewOverlay = document.getElementById('topViewOverlay');

            [boxOverlay, topViewOverlay].forEach(overlay => {{
                const elements = overlay.querySelectorAll('.box-line, .box-label');

                elements.forEach(element => {{
                    if(element.dataset.label === label) {{
                        if(highlight) {{
                            element.classList.add('highlight');
                            element.classList.remove('fade');
                        }} else {{
                            element.classList.remove('highlight');
                            element.classList.remove('fade');
                        }}
                    }} else {{
                        if(highlight) {{
                            element.classList.add('fade');
                            element.classList.remove('highlight');
                        }} else {{
                            element.classList.remove('fade');
                            element.classList.remove('highlight');
                        }}
                    }}
                }});
            }});
        }}

        function drawTopView() {{
            topViewOverlay.innerHTML = '';

            const zoom = parseFloat(zoomSlider.value);
            const viewWidth = 400;
            const viewHeight = 400;
            const centerX = viewWidth / 2 + panOffset.x;
            const centerY = viewHeight / 2 + panOffset.y;

            for(let x = -5; x <= 5; x++) {{
                const xPixel = centerX + x * (viewWidth/10) * zoom;
                const gridLine = document.createElement('div');
                gridLine.className = 'grid-line';
                gridLine.style.left = `${{xPixel}}px`;
                gridLine.style.top = '0';
                gridLine.style.width = '1px';
                gridLine.style.height = '100%';
                topViewOverlay.appendChild(gridLine);

                const label = document.createElement('div');
                label.className = 'grid-label';
                label.textContent = x.toString();
                label.style.left = `${{xPixel}}px`;
                label.style.bottom = '5px';
                topViewOverlay.appendChild(label);
            }}

            for(let y = -5; y <= 10; y++) {{
                const yPixel = centerY - y * (viewHeight/10) * zoom;
                const gridLine = document.createElement('div');
                gridLine.className = 'grid-line';
                gridLine.style.left = '0';
                gridLine.style.top = `${{yPixel}}px`;
                gridLine.style.width = '100%';
                gridLine.style.height = '1px';
                topViewOverlay.appendChild(gridLine);

                const label = document.createElement('div');
                label.className = 'grid-label';
                label.textContent = y.toString();
                label.style.left = '5px';
                label.style.top = `${{yPixel}}px`;
                topViewOverlay.appendChild(label);
            }}

            const xAxis = document.createElement('div');
            xAxis.className = 'axis-line';
            xAxis.style.left = `${{centerX}}px`;
            xAxis.style.top = '0';
            xAxis.style.width = '2px';
            xAxis.style.height = '100%';
            topViewOverlay.appendChild(xAxis);

            const yAxis = document.createElement('div');
            yAxis.className = 'axis-line';
            yAxis.style.left = '0';
            yAxis.style.top = `${{centerY}}px`;
            yAxis.style.width = '100%';
            yAxis.style.height = '2px';
            topViewOverlay.appendChild(yAxis);

            const camera = document.createElement('div');
            camera.className = 'camera-triangle';
            camera.style.left = `${{centerX - 10}}px`;
            camera.style.top = `${{centerY - 20}}px`;
            topViewOverlay.appendChild(camera);

            boxesData.forEach(boxData => {{
                const center = boxData.box_3d.slice(0,3);
                const size = boxData.box_3d.slice(3,6);
                const rpy = boxData.box_3d.slice(6).map(x => x * Math.PI / 180);

                const centerX = viewWidth/2 + center[0] * (viewWidth/10) * zoom + panOffset.x;
                const centerY = viewHeight/2 - center[1] * (viewHeight/10) * zoom + panOffset.y;

                const box = document.createElement('div');
                box.className = 'box-line';
                box.dataset.label = boxData.label;
                box.style.width = `${{size[0] * (viewWidth/10) * zoom}}px`;
                box.style.height = `${{size[1] * (viewHeight/10) * zoom}}px`;
                box.style.left = `${{centerX - (size[0] * (viewWidth/20) * zoom)}}px`;
                box.style.top = `${{centerY - (size[1] * (viewHeight/20) * zoom)}}px`;
                box.style.transform = `rotate(${{-rpy[2]}}rad)`;
                box.style.border = '2px solid #2962FF';
                box.style.background = 'transparent';
                topViewOverlay.appendChild(box);

                const label = document.createElement('div');
                label.className = 'box-label';
                label.dataset.label = boxData.label;
                label.textContent = boxData.label;
                label.style.left = `${{centerX}}px`;
                label.style.top = `${{centerY}}px`;

                label.addEventListener('mouseenter', () => highlightBox(boxData.label, true));
                label.addEventListener('mouseleave', () => highlightBox(boxData.label, false));

                topViewOverlay.appendChild(label);
            }});
        }}

        function annotateFrame(frame, fov) {{
            const boxOverlay = document.getElementById('boxOverlay');
            boxOverlay.innerHTML = '';

            boxesData.forEach(boxData => {{
                const center = boxData.box_3d.slice(0,3);
                const size = boxData.box_3d.slice(3,6);
                const rpy = boxData.box_3d.slice(6).map(x => x * Math.PI / 180);

                const [sr, sp, sy] = rpy.map(x => Math.sin(x/2));
                const [cr, cp, cz] = rpy.map(x => Math.cos(x/2));
                const quaternion = [
                    sr * cp * cz - cr * sp * sy,
                    cr * sp * cz + sr * cp * sy,
                    cr * cp * sy - sr * sp * cz,
                    cr * cp * cz + sr * sp * sy
                ];

                const height = frame.height;
                const width = frame.width;
                const f = width / (2 * Math.tan(fov/2 * Math.PI/180));
                const cx = width/2;
                const cy = height/2;
                const intrinsics = [[f, 0, cx], [0, f, cy], [0, 0, 1]];

                const halfSize = size.map(s => s/2);
                let corners = [];
                for(let x of [-halfSize[0], halfSize[0]]) {{
                    for(let y of [-halfSize[1], halfSize[1]]) {{
                        for(let z of [-halfSize[2], halfSize[2]]) {{
                            corners.push([x, y, z]);
                        }}
                    }}
                }}
                corners = [
                    corners[1], corners[3], corners[7], corners[5],
                    corners[0], corners[2], corners[6], corners[4]
                ];

                const q = quaternion;
                const rotationMatrix = [
                    [1 - 2*q[1]**2 - 2*q[2]**2, 2*q[0]*q[1] - 2*q[3]*q[2], 2*q[0]*q[2] + 2*q[3]*q[1]],
                    [2*q[0]*q[1] + 2*q[3]*q[2], 1 - 2*q[0]**2 - 2*q[2]**2, 2*q[1]*q[2] - 2*q[3]*q[0]],
                    [2*q[0]*q[2] - 2*q[3]*q[1], 2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[0]**2 - 2*q[1]**2]
                ];

                const boxVertices = corners.map(corner => {{
                    const rotated = matrixMultiply(rotationMatrix, corner);
                    return rotated.map((val, idx) => val + center[idx]);
                }});

                const tiltAngle = 90.0;
                const viewRotationMatrix = [
                    [1, 0, 0],
                    [0, Math.cos(tiltAngle * Math.PI/180), -Math.sin(tiltAngle * Math.PI/180)],
                    [0, Math.sin(tiltAngle * Math.PI/180), Math.cos(tiltAngle * Math.PI/180)]
                ];

                const points = boxVertices;
                const rotatedPoints = points.map(p => matrixMultiply(viewRotationMatrix, p));
                const translatedPoints = rotatedPoints.map(p => p.map(v => v + 0));

                const vertexDistances = translatedPoints.map(p =>
                    Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
                );

                const minDist = Math.min(...vertexDistances);
                const maxDist = Math.max(...vertexDistances);
                const distRange = maxDist - minDist;

                const projectedPoints = translatedPoints.map(p => matrixMultiply(intrinsics, p));
                const vertices = projectedPoints.map(p => [p[0]/p[2], p[1]/p[2]]);

                const topVertices = vertices.slice(0,4);
                const bottomVertices = vertices.slice(4,8);
                const topDistances = vertexDistances.slice(0,4);
                const bottomDistances = vertexDistances.slice(4,8);

                for(let i = 0; i < 4; i++) {{
                    const lines = [
                        {{start: topVertices[i], end: topVertices[(i + 1) % 4],
                        dist: (topDistances[i] + topDistances[(i + 1) % 4]) / 2}},
                        {{start: bottomVertices[i], end: bottomVertices[(i + 1) % 4],
                        dist: (bottomDistances[i] + bottomDistances[(i + 1) % 4]) / 2}},
                        {{start: topVertices[i], end: bottomVertices[i],
                        dist: (topDistances[i] + bottomDistances[i]) / 2}}
                    ];

                    for(let {{start, end, dist}} of lines) {{
                        const line = document.createElement('div');
                        line.className = 'box-line';
                        line.dataset.label = boxData.label;

                        const dx = end[0] - start[0];
                        const dy = end[1] - start[1];
                        const length = Math.sqrt(dx*dx + dy*dy);
                        const angle = Math.atan2(dy, dx);

                        const normalizedDist = (dist - minDist) / distRange;

                        const maxWidth = 4;
                        const minWidth = 1;
                        const width = maxWidth - normalizedDist * (maxWidth - minWidth);

                        line.style.width = `${{length}}px`;
                        line.style.height = `${{width}}px`;
                        line.style.transform = `translate(${{start[0]}}px, ${{start[1]}}px) rotate(${{angle}}rad)`;

                        boxOverlay.appendChild(line);
                    }}
                }}

                const textPosition3d = points[0].map((val, idx) =>
                    points.reduce((sum, p) => sum + p[idx], 0) / points.length
                );
                textPosition3d[2] += 0.1;

                const textPoint = matrixMultiply(intrinsics,
                    matrixMultiply(viewRotationMatrix, textPosition3d.map(v => v + 0))
                );
                const textPos = [textPoint[0]/textPoint[2], textPoint[1]/textPoint[2]];

                const label = document.createElement('div');
                label.className = 'box-label';
                label.dataset.label = boxData.label;
                label.textContent = boxData.label;
                label.style.left = `${{textPos[0]}}px`;
                label.style.top = `${{textPos[1]}}px`;

                label.addEventListener('mouseenter', () => highlightBox(boxData.label, true));
                label.addEventListener('mouseleave', () => highlightBox(boxData.label, false));

                boxOverlay.appendChild(label);
            }});
        }}

        function matrixMultiply(m, v) {{
            return m.map(row =>
                row.reduce((sum, val, i) => sum + val * v[i], 0)
            );
        }}

        const frame = {{
            width: canvas.width,
            height: canvas.height
        }};

        const fovSlider = document.getElementById('fovSlider');
        const fovValue = document.getElementById('fovValue');
        const zoomSlider = document.getElementById('zoomSlider');
        const zoomValue = document.getElementById('zoomValue');

        fovSlider.addEventListener('input', (e) => {{
            const fov = parseFloat(e.target.value);
            fovValue.textContent = `${{fov}}°`;
            annotateFrame(frame, fov);
            drawTopView();
        }});

        zoomSlider.addEventListener('input', (e) => {{
            const zoom = parseFloat(e.target.value);
            zoomValue.textContent = `${{zoom}}x`;
            drawTopView();
        }});

        function handleMouseDown(e, view) {{
            isDragging[view] = true;
            lastX = e.clientX;
            lastY = e.clientY;
        }}

        function handleMouseMove(e, view) {{
            if (isDragging[view]) {{
                const deltaX = e.clientX - lastX;
                const deltaY = e.clientY - lastY;

                if (view === 'left') {{
                    boxesData = boxesData.map(boxData => {{
                        const newBox3d = [...boxData.box_3d];
                        newBox3d[0] += deltaX * 0.001;
                        newBox3d[2] -= deltaY * 0.001;
                        return {{...boxData, box_3d: newBox3d}};
                    }});
                }} else {{
                    panOffset.x += deltaX;
                    panOffset.y += deltaY;
                }}

                lastX = e.clientX;
                lastY = e.clientY;

                annotateFrame(frame, parseFloat(fovSlider.value));
                drawTopView();
            }}
        }}

        function handleMouseUp(view) {{
            isDragging[view] = false;
        }}

        canvas.addEventListener('mousedown', (e) => handleMouseDown(e, 'left'));
        canvas.addEventListener('mousemove', (e) => handleMouseMove(e, 'left'));
        canvas.addEventListener('mouseup', () => handleMouseUp('left'));
        canvas.addEventListener('mouseleave', () => handleMouseUp('left'));

        topView.addEventListener('mousedown', (e) => handleMouseDown(e, 'right'));
        topView.addEventListener('mousemove', (e) => handleMouseMove(e, 'right'));
        topView.addEventListener('mouseup', () => handleMouseUp('right'));
        topView.addEventListener('mouseleave', () => handleMouseUp('right'));

        annotateFrame(frame, 60);
        drawTopView();
    </script>
</body>
</html>
"""
        
        