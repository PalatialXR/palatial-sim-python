from openai import OpenAI
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64
import mimetypes
from src.models.scene_graph import (
    SceneGraph,
    SpatialRelation,
    SemanticRelation,
    SceneObject,
    HierarchicalProperties,
    SPATIAL_RELATIONS
)

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
    Analyzes a scene image using OpenAI's GPT-4 Vision to extract scene information.
    
    Args:
        image_paths: List of paths to the image files to analyze
        prompt: Optional custom prompt to use. If None, uses default prompt.
        
    Returns:
        Dict containing:
            success: bool indicating if analysis was successful
            structured_analysis: SceneGraph object with scene structure
            response: Raw API response
            error: Error message if success is False
    """
    # Validate image paths
    for image_path in image_paths:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Initialize OpenAI client
    client = OpenAI()
    
    # Prepare default prompt if none provided
    if prompt is None:
        valid_categories = [
            "Bottle", "Box", "Bucket", "Cart", "Chair", "Clock", 
            "CoffeeMachine", "Dishwasher", "Dispenser", "Display/Monitor`", "Door", 
            "Eyeglasses", "Fan", "Faucet", "FoldingChair", "Globe", "Kettle", 
            "Keyboard", "KitchenPot", "Knife", "Lamp", "Laptop", "Lighter", 
            "Microwave", "Mouse", "Oven", "Pen", "Phone", "Pliers", "Printer", 
            "Refrigerator", "Remote", "Safe", "Scissors", "Stapler", 
            "StorageFurniture", "Suitcase", "Switch", "Table", "Toaster", 
            "Toilet", "TrashCan", "USB", "WashingMachine", "Window"
        ]
        
        prompt = (
            "Analyze this scene which contains multiple objects, and different images are more focused on different orientations of the scene. "
            "IMPORTANT: Only identify and describe objects that EXACTLY match one of these categories (case-sensitive):\n"
            f"{', '.join(valid_categories)}\n\n"
            "If an object doesn't match any of these categories exactly, DO NOT include it in the output at all, and don't mention it as a relationship with another object, "
            "and DO NOT create any relationships involving that object.\n\n"
            "For objects that DO match the valid categories:\n"
            "1. Name similar objects as '<object_name>_<index>'\n"
            "2. Include the exact category name from the list above\n"
            "3. Provide detailed descriptions including make, model, color, texture, and other properties\n"
            f"4. Define spatial relationships using ONLY these relations: {', '.join(SPATIAL_RELATIONS)}\n"
            "   Include TWO TYPES OF RELATIONSHIPS:\n"
            "   a) Parent-child relationships (e.g., objects on the table)\n"
            "   b) Peer relationships between objects at the same level\n"
            "5. Show hierarchical structure of which objects support others\n\n"
        )

    # Encode all images
    encoded_images = [_encode_image(path) for path in image_paths]

    # Prepare message content with text prompt and all images
    message_content = [{"type": "text", "text": prompt}]
    message_content.extend(encoded_images)

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

        # Convert semantic relations to spatial relations if needed
        scene_graph = response.choices[0].message.parsed
        if not isinstance(scene_graph, SceneGraph):
            raise ValueError("API response did not match SceneGraph format")

        return {
            "success": True,
            "structured_analysis": scene_graph,
            "response": response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "details": f"Failed to analyze images: {e}",
            "image_paths": image_paths
        }
