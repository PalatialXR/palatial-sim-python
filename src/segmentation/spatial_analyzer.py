from typing import List, Dict, Union, Any
from PIL import Image
import json
from google import genai
from google.genai import types
import numpy as np

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
            image (PIL.Image): Input image
            max_items (int): Maximum number of items to detect
            temperature (float): Temperature parameter for generation
            
        Returns:
            List[Dict]: List of detected items with points and labels
        """
        resized_img = self._resize_image(image)
        
        prompt = f"""
        Point to no more than {max_items} items in the image.
        The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...]. 
        The points are in [y, x] format normalized to 0-1000.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[resized_img, prompt],
            config=types.GenerateContentConfig(temperature=temperature)
        )
        
        return json.loads(response.text)

    def detect_3d_boxes(self, 
                       image: Image.Image, 
                       max_items: int = 10, 
                       temperature: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect 3D bounding boxes of objects in the image
        
        Args:
            image (PIL.Image): Input image
            max_items (int): Maximum number of items to detect
            temperature (float): Temperature parameter for generation
            
        Returns:
            List[Dict]: List of detected items with 3D bounding box parameters
        """
        prompt = f"""
        Detect the 3D bounding boxes of no more than {max_items} items.
        Output a json list where each entry contains the object name in "label" and its 3D bounding box in "box_3d"
        The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image, prompt],
            config=types.GenerateContentConfig(temperature=temperature)
        )
        
        return json.loads(response.text)

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
        
        return json.loads(response.text) 