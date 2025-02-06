import os
import sys
import logging
from typing import Dict, List, Optional
import pybullet as p
from pathlib import Path
import traceback
import json

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.scene_graph.PartNet import PartNetManager, PartNetMatch, PartNetError
from src.scene_graph.scene_analyzer import analyze_scene_image
from src.scene_graph.scene_planner import ScenePlanner
from src.scene_graph.scene_generator import Scene3DGenerator
from src.models.scene_graph import SceneGraph
from src.models.scene_graph import ObjectState

class SceneDriverError(Exception):
    """Base exception for SceneDriver errors."""
    pass

class SceneDriver:
    def __init__(self, partnet_root: str, gui: bool = True):
        """Initialize scene generation pipeline components."""
        self.partnet_manager = PartNetManager(partnet_root)
        self.scene_planner = ScenePlanner()
        self.scene_generator = Scene3DGenerator()
        self.gui = gui

    def generate_scene(self, 
                      scene_images: List[str],
                      output_dir: Optional[str] = None) -> Dict:
        """Generate complete 3D scene from images using PartNet objects."""
        try:
            # 1. Analyze scene images to create initial semantic graph
            logger.info("Analyzing scene images...")
            analysis_result = analyze_scene_image(scene_images)

            logger.info(f"Analysis result: {analysis_result}")
            
            if not analysis_result["success"]:
                raise SceneDriverError(f"Scene analysis failed: {analysis_result['error']}")
            
            initial_graph = analysis_result["structured_analysis"]
            
            # 2. Match objects to PartNet dataset and get URDFs/bounding boxes
            logger.info("Finding matching PartNet objects...")
            matches, updated_graph = self.partnet_manager.find_matching_objects(initial_graph)
            
            if not matches:
                raise SceneDriverError("No valid PartNet objects found for scene")
            
            if not self.partnet_manager.validate_scene_graph(updated_graph):
                raise SceneDriverError("Scene graph validation failed after matching")
            
            # 3. Plan optimal object placement
            logger.info("Planning object placement...")
            
            # Create object states with bounding box information from matches
            object_states = {}
            for match in matches:
                if match.bbox_data:
                    dims = self.partnet_manager._calculate_object_dimensions(match.bbox_data)
                    object_states[match.scene_object.name] = ObjectState(
                        name=match.scene_object.name,
                        bbox=dims,
                        image_info={
                            'paths': scene_images,
                            'bbox_2d': match.scene_object.bbox_2d if hasattr(match.scene_object, 'bbox_2d') else None,
                            'depth': match.scene_object.depth if hasattr(match.scene_object, 'depth') else None
                        }
                    )
                    logger.debug(f"Added bbox for {match.scene_object.name}: {dims}")
                else:
                    logger.warning(f"No bounding box data for {match.scene_object.name}")
            
            # Use ScenePlanner to determine positions
            placement_order, object_states = self.scene_planner.precompute_placement(
                updated_graph.relationships,
                object_states=object_states
            )
            
            # Log objects without positions but continue with those that have positions
            missing_positions = [name for name, state in object_states.items() 
                               if state.position is None]
            if missing_positions:
                logger.warning(f"Objects without positions (will be skipped): {missing_positions}")
            
            # Filter matches to only include objects with positions
            valid_matches = [match for match in matches 
                           if match.scene_object.name in object_states 
                           and object_states[match.scene_object.name].position is not None]
            
            if not valid_matches:
                raise SceneDriverError("No objects have valid positions for placement")
            
            # 4. Generate physical scene in PyBullet using Scene3DGenerator
            logger.info("Generating physical scene...")
            logger.info(f"Attempting to place {len(valid_matches)} objects")
            
            self.scene_generator.init_physics(gui=self.gui)
            object_positions = self.scene_generator.place_objects(valid_matches, object_states)
            
            if not object_positions:
                raise SceneDriverError("Failed to place any objects in physics simulation")
            
            # Track successfully placed objects
            placed_objects = set(object_positions.keys())
            all_objects = set(obj.scene_object.name for obj in matches)
            missing_objects = all_objects - placed_objects
            
            if missing_objects:
                logger.warning(f"Objects not placed in final scene: {sorted(missing_objects)}")
            
            logger.info(f"Successfully placed {len(placed_objects)} objects: {sorted(placed_objects)}")
            
            # 5. Save results if requested
            if output_dir:
                self._save_results(output_dir, updated_graph, matches, object_positions)
            
            return {
                "success": True,
                "scene_graph": updated_graph,
                "object_matches": matches,
                "object_positions": object_positions,
                "placement_order": placement_order,
                "placed_objects": sorted(placed_objects),
                "missing_objects": sorted(missing_objects)
            }
            
        except Exception as e:
            logger.error(f"Scene generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        finally:
            # Ensure physics simulation is cleaned up
            if hasattr(self, 'scene_generator') and self.scene_generator.physics_client is not None:
                try:
                    p.disconnect(self.scene_generator.physics_client)
                    self.scene_generator.physics_client = None
                except:
                    pass
    
    def _save_results(self, output_dir: str, graph: SceneGraph, matches: List[PartNetMatch], positions: Dict[str, List[float]]):
        """Save scene generation results."""
        # Save scene graph
        graph_path = os.path.join(output_dir, "scene_graph.json")
        with open(graph_path, 'w') as f:
            json.dump(graph.model_dump(), f, indent=2)  # Using model_dump instead of dict
            
        # Save object matches and positions
        matches_path = os.path.join(output_dir, "object_matches.json")
        matches_data = []
        for match in matches:
            try:
                match_data = {
                    "scene_object": match.scene_object.model_dump(),  # Using model_dump instead of dict
                    "urdf_path": match.urdf_path,
                    "category": match.category,
                    "object_id": match.object_id,
                    "similarity_score": match.similarity_score,
                    "match_description": match.match_description,
                    "position": positions.get(match.scene_object.name, None)  # Use .get() with default None
                }
                matches_data.append(match_data)
            except Exception as e:
                logger.warning(f"Failed to save match data for {match.scene_object.name}: {str(e)}")
                continue
                
        with open(matches_path, 'w') as f:
            json.dump(matches_data, f, indent=2)
            
        # Log statistics
        placed_objects = [name for name in positions.keys()]
        unplaced_objects = [match.scene_object.name for match in matches if match.scene_object.name not in positions]
        
        if unplaced_objects:
            logger.warning(f"Objects not placed in final scene: {unplaced_objects}")
        if placed_objects:
            logger.info(f"Successfully placed {len(placed_objects)} objects: {placed_objects}")
            
        return placed_objects, unplaced_objects


def main():
    """Example usage of SceneDriver."""
    driver = SceneDriver(
        partnet_root="src/datasets/partnet-mobility-v0",
        gui=True
    )
    
    result = driver.generate_scene(
        scene_images=["src/utils/assets/desk1.jpeg", "src/utils/assets/desk2.jpeg", "src/utils/assets/desk3.jpeg", "src/utils/assets/desk4.jpeg"],
        output_dir="output"
    )
    
    if result["success"]:
        logger.info(f"Scene generation successful!")
        logger.info(f"Generated scene with {len(result['object_matches'])} objects")
    else:
        logger.error(f"Scene generation failed: {result['error']}")


if __name__ == "__main__":
    main()