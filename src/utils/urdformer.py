# idea is that once we have the mesh, we can use trimesh to calculate the bounding box and dimensions. We then use the llm with the object, desciption and dimensions to generate the appropriate bounding box
# input will be the mesh file name, object name, and description
# output will be the saved urdf file location with object name and description


import trimesh
import openai
import os
import json
import logging
import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI

try:
    import chardet
except ImportError:
    logger.error("Missing required dependency 'chardet'. Please install it using: pip install chardet")
    raise ImportError("Please install chardet: pip install chardet")

logger = logging.getLogger(__name__)

urdf_schema = """
{
  "type": "object",
  "properties": {
    "robot": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "links": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "visual": {
                "type": "object",
                "properties": {
                  "geometry": {
                    "type": "object",
                    "properties": {
                      "shape": { "type": "string", "enum": ["box", "cylinder", "sphere", "mesh"] },
                      "size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                      "filename": { "type": "string" }
                    },
                    "required": ["shape"]
                  }
                },
                "required": ["geometry"]
              },
              "collision": {
                "type": "object",
                "properties": {
                  "geometry": {
                    "type": "object",
                    "properties": {
                      "shape": { "type": "string", "enum": ["box", "cylinder", "sphere", "mesh"] },
                      "size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                      "filename": { "type": "string" }
                    },
                    "required": ["shape"]
                  }
                },
                "required": ["geometry"]
              },
              "inertial": {
                "type": "object",
                "properties": {
                  "mass": { "type": "number" },
                  "inertia": {
                    "type": "object",
                    "properties": {
                      "ixx": { "type": "number" },
                      "iyy": { "type": "number" },
                      "izz": { "type": "number" }
                    },
                    "required": ["ixx", "iyy", "izz"]
                  }
                },
                "required": ["mass", "inertia"]
              }
            },
            "required": ["name", "visual", "collision", "inertial"]
          }
        },
        "joints": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "type": { "type": "string", "enum": ["fixed", "revolute", "prismatic", "continuous"] },
              "parent": { "type": "string" },
              "child": { "type": "string" },
              "axis": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
              "limit": {
                "type": "object",
                "properties": {
                  "lower": { "type": "number" },
                  "upper": { "type": "number" },
                  "effort": { "type": "number" },
                  "velocity": { "type": "number" }
                },
                "required": ["lower", "upper", "effort", "velocity"]
              }
            },
            "required": ["name", "type", "parent", "child"]
          }
        }
      },
      "required": ["name", "links"]
    }
  },
  "required": ["robot"]
}
"""

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

def generate_urdf(
    mesh_file_name: str,
    object_name: str,
    object_description: str,
    object_category: str,
    image_path: Optional[str] = None
) -> Optional[str]:
    """Generate URDF file with mesh and optional image context
    
    Args:
        mesh_file_name: Path to the mesh file
        object_name: Name of the object
        object_description: Description of the object
        object_category: LVIS/COCO category of the object
        image_path: Optional path to object image for visual context
        
    Returns:
        Path to the generated URDF file, or None if generation fails or mesh doesn't exist
    """
    # Skip if mesh file doesn't exist
    if not mesh_file_name or not os.path.exists(mesh_file_name):
        logger.warning(f"Mesh file not found for {object_name}, skipping URDF generation")
        return None

    try:
        # Load the mesh
        loaded_mesh = trimesh.load(mesh_file_name)
        if loaded_mesh is None:
            logger.error(f"Failed to load mesh for {object_name}: Mesh loading returned None")
            return None
        
        # Handle both Scene and Trimesh objects
        if isinstance(loaded_mesh, trimesh.Scene):
            logger.info(f"Loaded Scene object for {object_name}, extracting geometry...")
            
            # Debug scene contents
            logger.info(f"Scene contents for {object_name}:")
            logger.info(f"- Graph keys: {list(loaded_mesh.graph.keys())}")
            logger.info(f"- Geometry keys: {list(loaded_mesh.geometry.keys())}")
            
            # Try multiple methods to extract geometry
            geometry = []
            
            # Method 1: Direct geometry values
            if loaded_mesh.geometry:
                geometry = list(loaded_mesh.geometry.values())
                logger.info(f"Found {len(geometry)} geometries using direct extraction")
            
            # Method 2: Try to dump the scene to a single mesh
            if not geometry:
                try:
                    logger.info("Attempting to export scene to single mesh...")
                    temp_mesh = loaded_mesh.dump(concatenate=True)
                    if isinstance(temp_mesh, trimesh.Trimesh):
                        geometry = [temp_mesh]
                        logger.info("Successfully exported scene to single mesh")
                    else:
                        logger.warning(f"Scene dump returned unexpected type: {type(temp_mesh)}")
                except Exception as e:
                    logger.warning(f"Failed to export scene to single mesh: {e}")
            
            # Method 3: Try to get geometry from graph
            if not geometry:
                try:
                    logger.info("Attempting to extract geometry from scene graph...")
                    for node_name in loaded_mesh.graph.nodes:
                        if 'geometry' in loaded_mesh.graph.nodes[node_name]:
                            geom = loaded_mesh.graph.nodes[node_name]['geometry']
                            if isinstance(geom, trimesh.Trimesh):
                                geometry.append(geom)
                    if geometry:
                        logger.info(f"Found {len(geometry)} geometries in scene graph")
                except Exception as e:
                    logger.warning(f"Failed to extract geometry from scene graph: {e}")
            
            if not geometry:
                logger.error(f"No geometry found in Scene for {object_name} after trying multiple extraction methods")
                return None
            
            # Combine all meshes in the scene
            try:
                logger.info(f"Attempting to concatenate {len(geometry)} geometries...")
                mesh = trimesh.util.concatenate(geometry)
                if not isinstance(mesh, trimesh.Trimesh):
                    logger.error(f"Failed to combine scene geometry for {object_name}: result is {type(mesh)}")
                    return None
                logger.info("Successfully concatenated geometries")
            except Exception as e:
                logger.error(f"Error concatenating geometries: {e}")
                return None
                
        elif isinstance(loaded_mesh, trimesh.Trimesh):
            mesh = loaded_mesh
            logger.info("Loaded direct Trimesh object")
        else:
            logger.error(f"Unsupported mesh type for {object_name}: {type(loaded_mesh)}")
            return None

        # Calculate the bounding box
        try:
            # Ensure the mesh is valid
            if not mesh.is_valid:
                logger.info(f"Fixing invalid mesh for {object_name}...")
                mesh.fix_normals()
                mesh.fill_holes()
                mesh.remove_degenerate_faces()
                mesh.remove_duplicate_faces()
                
                if not mesh.is_valid:
                    logger.error(f"Failed to fix invalid mesh for {object_name}")
                    return None

            bounding_box = mesh.bounding_box
            if bounding_box is None:
                logger.error(f"Failed to calculate bounding box for {object_name}")
                return None
            
            bounding_box_dimensions = bounding_box.extents
            if bounding_box_dimensions is None or not all(isinstance(x, (int, float)) for x in bounding_box_dimensions):
                logger.error(f"Invalid bounding box dimensions for {object_name}: {bounding_box_dimensions}")
                return None
            
            # Calculate volume - handle case where volume calculation fails
            try:
                volume = mesh.volume
                if volume is None or not isinstance(volume, (int, float)):
                    logger.warning(f"Invalid volume for {object_name}, using bounding box volume as approximation")
                    volume = bounding_box_dimensions[0] * bounding_box_dimensions[1] * bounding_box_dimensions[2]
            except Exception as e:
                logger.warning(f"Failed to calculate exact volume for {object_name}, using bounding box volume: {e}")
                volume = bounding_box_dimensions[0] * bounding_box_dimensions[1] * bounding_box_dimensions[2]

        except Exception as e:
            logger.error(f"Error calculating mesh properties for {object_name}: {e}")
            return None

        logger.info(f"Mesh properties for {object_name}:")
        logger.info(f"- Bounding box dimensions: {bounding_box_dimensions}")
        logger.info(f"- Volume: {volume}")
        logger.info(f"- Number of vertices: {len(mesh.vertices)}")
        logger.info(f"- Number of faces: {len(mesh.faces)}")

        # Prepare the base prompt
        prompt = (
            f"Generate a complete structured URDF file for the object {object_name} (category: {object_category}).\n"
            f"The object has the following properties:\n"
            f"Description: {object_description}\n"
            f"Dimensions: {bounding_box_dimensions}\n"
            f"Volume: {volume}\n"
            f"Mesh file: {mesh_file_name}\n"
            f"Vertices: {len(mesh.vertices)}\n"
            f"Faces: {len(mesh.faces)}\n\n"
            f"Focus on articulation points and joints based on the object's appearance and category.\n"
            f"Use the following schema for the URDF structure:\n{urdf_schema}"
        )

        # Initialize OpenAI client
        client = OpenAI()

        # Prepare message content
        message_content = [{"type": "text", "text": prompt}]
        
        # Add image if provided
        if image_path and os.path.exists(image_path):
            try:
                encoded_image = _encode_image(image_path)
                message_content.append(encoded_image)
                logger.info("Added image context to URDF generation")
            except Exception as e:
                logger.error(f"Failed to encode image {image_path}: {e}")

        # Generate the URDF with OpenAI API structured output
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",  # Using same model as scene_analyzer
                messages=[
                    {
                        "role": "system",
                        "content": "You are a robotics expert that generates URDF files. Focus on identifying articulation points and creating accurate joint configurations based on the object's appearance and category."
                    },
                    {
                        "role": "user", 
                        "content": message_content
                    }
                ],
                response_format={"type": "json_object"}
            )

            logger.info("Successfully generated URDF response")
            urdf_data = json.loads(response.choices[0].message.content)

            # Save the URDF
            os.makedirs("urdfs", exist_ok=True)
            urdf_file_location = f"urdfs/{object_name}.urdf"
            with open(urdf_file_location, "w") as f:
                json.dump(urdf_data, f, indent=2)

            return urdf_file_location

        except Exception as e:
            logger.error(f"Failed to generate URDF for {object_name}: {e}")
            return None

    except Exception as e:
        logger.error(f"Failed to process mesh for {object_name}: {e}")
        return None

