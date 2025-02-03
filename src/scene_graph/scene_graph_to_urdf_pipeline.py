from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import logging
from scene_graph.scene_analyzer import analyze_scene_image
from utils.objaverse_downloader import ObjaverseDownloader
from utils.urdformer import generate_urdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneGraphToURDFPipeline:
    def __init__(self, download_dir: str = "downloaded_objects", urdf_dir: str = "generated_urdfs"):
        """Initialize the pipeline with directories for downloaded objects and generated URDFs.
        
        Args:
            download_dir: Directory to store downloaded 3D models
            urdf_dir: Directory to store generated URDF files
        """
        self.download_dir = Path(download_dir)
        self.urdf_dir = Path(urdf_dir)
        self.objaverse_downloader = ObjaverseDownloader()
        
        # Create directories if they don't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.urdf_dir.mkdir(parents=True, exist_ok=True)

    def process_scene(self, image_paths: List[str], custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a scene from images to generate URDFs for all objects.
        
        Args:
            image_paths: List of paths to scene images
            custom_prompt: Optional custom prompt for scene analysis
            
        Returns:
            Dictionary containing:
            - scene_graph: The analyzed scene graph
            - downloaded_objects: Info about downloaded 3D models
            - urdf_files: Paths to generated URDF files
        """
        try:
            # Step 1: Analyze scene images to get scene graph
            logger.info("Analyzing scene images...")
            scene_analysis = analyze_scene_image(image_paths, prompt=custom_prompt)
            
            if not scene_analysis["success"]:
                raise Exception(f"Scene analysis failed: {scene_analysis['error']}")
            
            scene_graph = scene_analysis["structured_analysis"]
            
            # Step 2: Create object descriptions dictionary using existing descriptions, and add LVIS/COCO category
            objects_dict = {}
            for obj in scene_graph.objects:
                # Use the exact description from the scene analyzer
                objects_dict[obj.name] = {
                    "description": obj.description,
                    "category": obj.category
                }
            # Step 3: Download 3D models using ObjaverseDownloader
            logger.info("Downloading 3D models from Objaverse...")
            download_results = self.objaverse_downloader.process_objects(
                objects_dict, 
                download_dir=str(self.download_dir)
            )
            
            # Step 4: Generate URDF files for each downloaded object
            logger.info("Generating URDF files...")
            urdf_files = {}
            
            for obj_name, obj_data in download_results.items():
                if obj_data["file_path"] is not None:
                    try:
                        # Generate URDF file using the original detailed description
                        urdf_path = generate_urdf(
                            mesh_file_name=obj_data["file_path"],
                            object_name=obj_name,
                            object_description=objects_dict[obj_name]["description"],
                            object_category=objects_dict[obj_name]["category"]
                        )
                        
                        # Move URDF to final location
                        final_urdf_path = self.urdf_dir / f"{obj_name}.urdf"
                        os.rename(urdf_path, final_urdf_path)
                        urdf_files[obj_name] = str(final_urdf_path)
                        
                        logger.info(f"Generated URDF for {obj_name}: {final_urdf_path}")
                    except Exception as e:
                        logger.error(f"Failed to generate URDF for {obj_name}: {str(e)}")
                        urdf_files[obj_name] = None
                else:
                    logger.warning(f"No 3D model downloaded for {obj_name}, skipping URDF generation")
                    urdf_files[obj_name] = None
            
            return {
                "success": True,
                "scene_graph": scene_graph,
                "downloaded_objects": download_results,
                "urdf_files": urdf_files
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def cleanup(self):
        """Clean up temporary files and resources."""
        # Add any cleanup code here if needed
        pass 