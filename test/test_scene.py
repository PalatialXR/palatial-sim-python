import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.scene_graph.scene_analyzer import analyze_scene_image
from src.models.scene_graph import SceneGraph, SpatialRelation
def main():
    # Path to the desk image
    image_paths = ["src/utils/assets/desk1.jpeg", "src/utils/assets/desk2.jpeg", "src/utils/assets/desk3.jpeg", "src/utils/assets/desk4.jpeg"]
    print ("Analyzing images:")    
    print("=" * 50)
    
    # Run analysis
    try:
        result = analyze_scene_image(image_paths)
        
        if result["success"]:
            scene_graph = result["structured_analysis"]
            
            print("\nObjects detected:")
            print("-" * 20)
            for obj in scene_graph.objects:
                print(f"Name: {obj.name}")
                print(f"Category: {obj.category}")
                print(f"Level: {obj.hierarchy.level}")
                print(f"Is anchor: {obj.hierarchy.is_anchor}")
                print(f"Parent: {obj.hierarchy.parent}")
                print(f"Children: {obj.hierarchy.children}")
            
            print("\nSpatial Relationships:")
            print("-" * 20)
            for rel in scene_graph.relationships:
                print(f"{rel.source} {rel.relation_type} {rel.target}")
                print()
                
        else:
            print("\nAnalysis failed!")
            print(f"Error: {result['error']}")
            print(f"Details: {result['details']}")
            
    except Exception as e:
        print(f"\nError running analysis: {str(e)}")

if __name__ == "__main__":
    main()