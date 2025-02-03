from openai import OpenAI
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64
import mimetypes
from models.scene_graph import SceneGraph, SpatialRelation

def _encode_image(image_path: str) -> Dict[str, str]:
    """Helper function to encode image and determine mime type"""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        mime_type = 'image/jpeg'  # Default to JPEG
        
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}"
            }
        }

def analyze_scene_image(image_paths: List[str], prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes a scene image using OpenAI's GPT-4o Structured Outputs to extract scene information.
    
    Args:
        image_paths: List of paths to the image files to analyze
        prompt: Optional custom prompt to use. If None, uses default prompt.
        
    Returns:
        Dict containing the model's response and structured scene analysis
    """
    # Validate image paths
    for image_path in image_paths:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Initialize OpenAI client
    client = OpenAI()
    
    # Prepare default prompt if none provided
    if prompt is None:
        prompt = ("Analyze this scene which contains multiple objects, and different images are more focused on different orientations of the scene, and provide a structured description of:\n"
                 "If there are multiple similar objects, name them as '<object_name>_<index>', and include their relationships. Make sure to include the LVIS/COCO based category, and detailed descriptions, including make, model, color, texture, and other properties of each of the objects in the scene.\n"
                 "1. Spatial relationships between ALL objects using the following relations: on, next_to, above, below, in_front_of, behind, left_of, right_of, inside, between, aligned_with. TWO TYPES OF RELATIONSHIPS SHOULD BE PRESENT: a) Parent-child relationships (e.g., objects on the desk) b) Peer relationships between objects at the same level\n"
                 "2. Hierarchical structure showing which objects support others\n"
                 "Format the output to match the following structure:\n"
                 "- List of ObjectNode entries with name, parent, children\n"
                 "- List of SubLinkage entries showing relationships between objects")

    # Encode all images
    encoded_images = [_encode_image(path) for path in image_paths]

    # Prepare message content with text prompt and all images
    message_content = [{"type": "text", "text": prompt}]
    message_content.extend(encoded_images)

    # Prepare the API request
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are a computer vision system that analyzes scenes and returns structured scene graphs in JSON format. Be precise and only return the JSON object with no additional text or explanations."
                },
                {
                    "role": "user", 
                    "content": message_content
                }
            ],
            response_format=SceneGraph
        )

        print(response)

        return {
            "success": True,
            "structured_analysis": response.choices[0].message.parsed,
            "response": response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "details": f"Failed to analyze images: {e}",
            "image_paths": image_paths
        }
