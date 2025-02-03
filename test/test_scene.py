import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from scene_graph.scene_analyzer import analyze_scene_image

def main():
    # Path to the desk image
    image_paths = ["desk1.jpeg", "desk2.jpeg", "desk3.jpeg", "desk4.jpeg"]
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