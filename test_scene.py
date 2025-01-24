import pybullet as p
import pybullet_data
import time
import os
from src.scene_graph.scene_generator import Scene3DGenerator, SemanticRelation, SpatialRelation

def test_monitor_on_table():
    """Test placing a monitor on a table with physics validation."""
    print("\nInitializing physics test...")
    
    # Initialize scene generator
    generator = Scene3DGenerator()
    
    # Set pybullet data path for plane.urdf
    plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    
    # Initialize physics with correct plane path
    generator.physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.loadURDF(plane_path)
    
    try:
        # Define the semantic relations for our scene
        relations = [
            SemanticRelation(
                source="monitor",
                relation=SpatialRelation.ON,
                target="table",
                confidence=1.0
            )
        ]
        
        print("\nPlacing objects...")
        object_positions = generator.place_objects(relations)
        
        # Basic validation
        monitor_pos = object_positions["monitor"]
        table_pos = object_positions["table"]
        assert monitor_pos[2] > table_pos[2], "Monitor should be above table"
        
        print("\nObject positions:")
        for obj_name, pos in object_positions.items():
            print(f"{obj_name}: {[f'{x:.3f}' for x in pos]}")
        
        # Run physics and keep window open
        print("\nRunning physics simulation. Press Ctrl+C to exit...")
        try:
            while True:
                p.stepSimulation()
                time.sleep(1/240.0)  # Run at ~240 Hz
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
    finally:
        print("\nTest complete - close the PyBullet window to exit completely")

if __name__ == "__main__":
    test_monitor_on_table()