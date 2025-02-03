from scene_graph.scene_graph_to_urdf_pipeline import SceneGraphToURDFPipeline
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the pipeline
    pipeline = SceneGraphToURDFPipeline(
        download_dir="downloaded_objects",
        urdf_dir="generated_urdfs"
    )
    
    # List of scene images to process
    image_paths = [
        "src/utils/assets/desk1.jpeg",
        "src/utils/assets/desk2.jpeg",
        "src/utils/assets/desk3.jpeg",
        "src/utils/assets/desk4.jpeg"
    ]
    
    try:
        # Process the scene
        logger.info("Starting scene processing pipeline...")
        results = pipeline.process_scene(image_paths)
        
        if results["success"]:
            # Print summary
            logger.info("\nPipeline completed successfully!")
            
            logger.info("\nScene Graph:")
            for obj in results["scene_graph"].objects:
                logger.info(f"- {obj.name}:")
                logger.info(f"  Category: {obj.category}")
                logger.info(f"  Description: {obj.description}")
                if obj.hierarchy:
                    logger.info(f"  Parent: {obj.hierarchy.parent}")
                    logger.info(f"  Children: {obj.hierarchy.children}")
            
            logger.info("\nSpatial Relationships:")
            for rel in results["scene_graph"].relationships:
                logger.info(f"- {rel.source} is {rel.relation_type} {rel.target}")
            
            logger.info("\nDownloaded Objects:")
            for obj_name, obj_data in results["downloaded_objects"].items():
                if obj_data["file_path"]:
                    logger.info(f"- {obj_name}: {obj_data['file_path']}")
                else:
                    logger.info(f"- {obj_name}: Failed to download")
            
            logger.info("\nGenerated URDFs:")
            for obj_name, urdf_path in results["urdf_files"].items():
                if urdf_path:
                    logger.info(f"- {obj_name}: {urdf_path}")
                else:
                    logger.info(f"- {obj_name}: Failed to generate URDF")
            
            # Save results to JSON for reference
            output_file = Path("scene_to_urdf_results.json")
            with open(output_file, "w") as f:
                # Convert scene graph to dict for JSON serialization
                results_json = {
                    "success": results["success"],
                    "scene_graph": {
                        "objects": [
                            {
                                "name": obj.name,
                                "category": obj.category,
                                "description": obj.description,
                                "hierarchy": {
                                    "level": obj.hierarchy.level if obj.hierarchy else None,
                                    "parent": obj.hierarchy.parent if obj.hierarchy else None,
                                    "children": obj.hierarchy.children if obj.hierarchy else []
                                } if obj.hierarchy else None
                            } for obj in results["scene_graph"].objects
                        ],
                        "relationships": [
                            {
                                "source": rel.source,
                                "target": rel.target,
                                "relation_type": rel.relation_type
                            } for rel in results["scene_graph"].relationships
                        ]
                    },
                    "downloaded_objects": results["downloaded_objects"],
                    "urdf_files": results["urdf_files"]
                }
                json.dump(results_json, f, indent=2)
            
            logger.info(f"\nResults saved to {output_file}")
            
        else:
            logger.error(f"Pipeline failed: {results['error']}")
    
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
    
    finally:
        # Clean up resources
        pipeline.cleanup()

if __name__ == "__main__":
    main() 