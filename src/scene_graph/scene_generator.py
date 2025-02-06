from typing import List, Dict, Tuple
import pybullet as p
import pybullet_data
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scene_graph.PartNet import PartNetMatch
from src.models.scene_graph import ObjectState

class Scene3DGenerator:
    def __init__(self):
        self.physics_client = None
        self.object_ids = {}
        
    def init_physics(self, gui=True):
        """Initialize physics simulation with real-time disabled for stable placement."""
        try:
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
            
            self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")
            p.setRealTimeSimulation(0)  # Disable real-time for stable placement
            p.setTimeStep(1./240.)  # Standard 240Hz timestep
            
            if gui:
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0]
                )
        except Exception as e:
            print(f"Error initializing physics: {str(e)}")
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
            raise
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)
    
    def place_objects(self, matches: List[PartNetMatch], object_states: Dict[str, ObjectState]) -> Dict[str, Tuple[float, float, float]]:
        """Place objects at their predetermined positions and simulate physics."""
        print("\nPlacing objects in physics simulation...")
        final_positions = {}
        
        try:
            # Place objects in order
            for match in matches:
                obj_name = match.scene_object.name
                obj_state = object_states.get(obj_name)
                
                if not obj_state or not obj_state.position:
                    print(f"Warning: No position specified for {obj_name}, skipping")
                    continue
                    
                print(f"\nPlacing {obj_name}:")
                print(f"  URDF: {match.urdf_path}")
                print(f"  Position: {[f'{x:.3f}' for x in obj_state.position]}")
                
                try:
                    # Set upright orientation (no rotation)
                    orientation = p.getQuaternionFromEuler([0, 0, 0])
                    
                    # Load URDF with position and orientation
                    obj_id = p.loadURDF(
                        match.urdf_path,
                        obj_state.position,
                        orientation,
                        useFixedBase=obj_name.lower().startswith("table")  # Fix tables to ground
                    )
                    self.object_ids[obj_name] = obj_id
                    
                    # Set physics properties for stability
                    p.changeDynamics(
                        obj_id, 
                        -1,  # -1 means base link
                        linearDamping=0.9,
                        angularDamping=0.9,
                        jointDamping=0.9,
                        restitution=0.1,
                        lateralFriction=1.0,
                        spinningFriction=0.1,
                        rollingFriction=0.1,
                    )
                    
                    # Let object settle briefly
                    for _ in range(10):
                        p.stepSimulation()
                    
                    # Get and store position after settling
                    pos, _ = p.getBasePositionAndOrientation(obj_id)
                    final_positions[obj_name] = pos
                    print(f"  Successfully placed at: {[f'{x:.3f}' for x in pos]}")
                    
                except Exception as e:
                    print(f"Error placing {obj_name}: {str(e)}")
                    continue
            
            # Let all objects settle together
            print("\nLetting objects settle...")
            for _ in range(100):
                p.stepSimulation()
                
            # Update final positions after full settling
            print("\nFinal object positions:")
            for obj_name, obj_id in self.object_ids.items():
                try:
                    pos, _ = p.getBasePositionAndOrientation(obj_id)
                    final_positions[obj_name] = pos
                    print(f"{obj_name}: {[f'{x:.3f}' for x in pos]}")
                except Exception as e:
                    print(f"Error getting final position for {obj_name}: {str(e)}")
            
            return final_positions
            
        except Exception as e:
            print(f"Error in physics simulation: {str(e)}")
            return final_positions
        finally:
            # Clean up objects
            for obj_id in self.object_ids.values():
                try:
                    p.removeBody(obj_id)
                except:
                    pass
            self.object_ids.clear()