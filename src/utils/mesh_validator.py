import os
import logging
import trimesh
import requests
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def validate_and_process_mesh(mesh_url: str, save_dir: str, object_name: str) -> Tuple[bool, Optional[str]]:
    """
    Downloads, validates and processes a mesh file.
    
    Args:
        mesh_url: URL to download the mesh from
        save_dir: Directory to save the processed mesh
        object_name: Name of the object (for naming the file)
        
    Returns:
        Tuple of (success: bool, file_path: Optional[str])
    """
    os.makedirs(save_dir, exist_ok=True)
    stl_path = os.path.join(save_dir, f"model.stl")
    
    try:
        # Download the mesh file
        logger.info(f"Downloading mesh for {object_name} from {mesh_url}")
        response = requests.get(mesh_url, stream=True)
        response.raise_for_status()
        
        # Check if the response is empty
        if int(response.headers.get('content-length', 0)) < 100:  # Less than 100 bytes is suspicious
            logger.warning(f"Downloaded file for {object_name} is too small, likely empty")
            return False, None
            
        # Save the downloaded file
        with open(stl_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Try to load with trimesh to validate
        try:
            mesh = trimesh.load(stl_path)
            if mesh is None:
                logger.warning(f"Trimesh couldn't load the file for {object_name}")
                return False, None
                
            # Check if it's a scene or direct mesh
            if isinstance(mesh, trimesh.Scene):
                if not mesh.geometry:
                    logger.warning(f"Scene has no geometry for {object_name}")
                    return False, None
                # Convert scene to single mesh if possible
                try:
                    mesh = mesh.dump(concatenate=True)
                except Exception as e:
                    logger.warning(f"Failed to convert scene to mesh for {object_name}: {e}")
                    return False, None
                    
            # Validate the mesh
            if not isinstance(mesh, trimesh.Trimesh):
                logger.warning(f"File is not a valid mesh for {object_name}")
                return False, None
                
            # Check if mesh has valid geometry
            if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
                logger.warning(f"Mesh has invalid geometry for {object_name}: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
                return False, None
                
            # Save the processed mesh
            processed_path = os.path.join(save_dir, f"processed_model.stl")
            mesh.export(processed_path)
            logger.info(f"Successfully processed and saved mesh for {object_name}")
            
            return True, processed_path
            
        except Exception as e:
            logger.warning(f"Failed to validate mesh for {object_name}: {e}")
            return False, None
            
    except Exception as e:
        logger.error(f"Failed to download or process mesh for {object_name}: {e}")
        return False, None 