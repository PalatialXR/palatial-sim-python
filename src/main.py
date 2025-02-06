import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import logging
import datetime
import numpy as np
import json
from segmentation.segmenter import SemanticSegmenter
from segmentation.interactive_segmenter import InteractiveSegmenter
from segmentation.interactive_ui import InteractiveSegmentationUI
from segmentation.description_generator import DescriptionGenerator
from segmentation.spatial_analyzer import GeminiSpatialUnderstanding
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Segment objects and detect 3D bounding boxes in images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="debug", help="Output directory")
    parser.add_argument("--pipeline", type=str, choices=["auto", "gemini", "depth", "interactive"], default="auto",
                      help="Pipeline to use (auto/gemini/interactive segmentation or depth estimation)")
    parser.add_argument("--max-objects", type=int, default=10,
                      help="Maximum number of objects to detect")
    parser.add_argument("--temperature", type=float, default=0.5,
                      help="Temperature for Gemini generation")
    parser.add_argument("--mode", type=str, choices=["2d", "3d", "points"], default="2d",
                      help="Detection mode for Gemini pipeline (2d/3d boxes or direct points)")
    parser.add_argument("--segment", action="store_true",
                      help="Run segmentation after detection")
    parser.add_argument("--generate-descriptions", action="store_true",
                      help="Generate physical descriptions for segmented objects")
    parser.add_argument("--points", type=str,
                      help="Comma-separated list of x,y coordinates for interactive mode")
    parser.add_argument("--labels", type=str,
                      help="Comma-separated list of point labels (1 for foreground, 0 for background)")
    parser.add_argument("--box", type=str,
                      help="Comma-separated box coordinates x1,y1,x2,y2 for interactive mode")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and prepare image
    image = Image.open(args.image)
    
    if args.pipeline == "gemini":
        # Get Gemini API key
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize Gemini spatial analyzer
        spatial_analyzer = GeminiSpatialUnderstanding(gemini_api_key)
        
        # Create output directory
        output_dir = Path(args.output_dir) / f"gemini_debug_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.mode == "3d":
            # Detect 3D boxes
            boxes = spatial_analyzer.detect_3d_boxes(
                image=image,
                max_items=args.max_objects,
                temperature=args.temperature
            )
            
            # Convert image for visualization
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Visualize 3D boxes
            result_image = spatial_analyzer.visualize_analysis(
                image=cv_image,
                boxes=boxes,
                mode="3d",
                save_path=str(output_dir / "gemini_3d_boxes.jpg")
            )
            
            # Save boxes data
            with open(output_dir / "gemini_boxes.json", "w") as f:
                json.dump(boxes, f, indent=2)
                
            logger.info(f"Saved 3D box visualization and data to {output_dir}")
            
            # Run segmentation if requested
            if args.segment:
                segmenter = SemanticSegmenter()
                segmenter.set_image(image)
                masks, scores = segmenter.segment_with_gemini(
                    gemini_api_key=gemini_api_key,
                    max_items=args.max_objects,
                    temperature=args.temperature,
                    mode="3d"
                )
                logger.info(f"Generated {len(masks)} masks using Gemini segmentation")
                
                # Save segmentation results
                segmenter.save_results(
                    masks=masks,
                    scores=scores,
                    output_dir=output_dir,
                    generate_descriptions=args.generate_descriptions
                )
        
        elif args.mode == "2d":
            # Detect 2D boxes
            boxes = spatial_analyzer.detect_2d_boxes(
                pil_image=image,
                max_items=args.max_objects,
                temperature=args.temperature
            )
            
            # Convert image for visualization
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Visualize 2D boxes
            result_image = spatial_analyzer.visualize_analysis(
                image=cv_image,
                boxes=boxes,
                mode="2d",
                save_path=str(output_dir / "gemini_2d_boxes.jpg")
            )
            
            # Save boxes data
            with open(output_dir / "gemini_boxes.json", "w") as f:
                json.dump(boxes, f, indent=2)
                
            logger.info(f"Saved 2D box visualization and data to {output_dir}")
            
            # Run segmentation if requested
            if args.segment:
                segmenter = SemanticSegmenter()
                segmenter.set_image(image)
                masks, scores = segmenter.segment_with_gemini(
                    gemini_api_key=gemini_api_key,
                    max_items=args.max_objects,
                    temperature=args.temperature,
                    mode="2d"
                )
                logger.info(f"Generated {len(masks)} masks using Gemini segmentation")
                
                # Save segmentation results
                segmenter.save_results(
                    masks=masks,
                    scores=scores,
                    output_dir=output_dir,
                    generate_descriptions=args.generate_descriptions
                )
        
        elif args.mode == "points":
            # Get points
            points = spatial_analyzer.point_to_items(
                image=image,
                max_items=args.max_objects,
                temperature=args.temperature
            )
            
            # Convert image for visualization
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Visualize points
            result_image = spatial_analyzer.visualize_analysis(
                image=cv_image,
                boxes=points,
                mode="points",
                save_path=str(output_dir / "gemini_points.jpg")
            )
            
            # Save points data
            with open(output_dir / "gemini_points.json", "w") as f:
                json.dump(points, f, indent=2)
                
            logger.info(f"Saved points visualization and data to {output_dir}")
            
            # Run segmentation if requested
            if args.segment:
                segmenter = SemanticSegmenter()
                segmenter.set_image(image)
                masks, scores = segmenter.segment_with_points(
                    gemini_api_key=gemini_api_key,
                    max_items=args.max_objects,
                    temperature=args.temperature
                )
                logger.info(f"Generated {len(masks)} masks using point-based segmentation")
                
                # Save segmentation results
                segmenter.save_results(
                    masks=masks,
                    scores=scores,
                    output_dir=output_dir,
                    generate_descriptions=args.generate_descriptions
                )
    
    elif args.pipeline == "interactive":
        # Run interactive segmentation pipeline
        logger.info("Running interactive segmentation pipeline")
        
        # Initialize interactive segmenter
        segmenter = InteractiveSegmenter()
        segmenter.set_image(image)
        
        # Create and run interactive UI
        ui = InteractiveSegmentationUI(segmenter)
        masks, scores = ui.run()
        
        if masks is not None and scores is not None:
            # Create output directory
            output_dir = Path(args.output_dir) / "interactive" / f"output_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            segmenter.save_results(
                masks=masks,
                scores=scores,
                output_dir=output_dir,
                generate_descriptions=args.generate_descriptions
            )
    
    else:  # Auto pipeline
        # Run automatic segmentation
        segmenter = SemanticSegmenter()
        segmenter.set_image(image)
        masks, scores = segmenter.segment_automatic(max_objects=args.max_objects)
        logger.info(f"Generated {len(masks)} masks using automatic segmentation")
        
        # Create output directory
        output_dir = Path(args.output_dir) / "auto" / f"output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        segmenter.save_results(
            masks=masks,
            scores=scores,
            output_dir=output_dir,
            generate_descriptions=args.generate_descriptions
        )

if __name__ == "__main__":
    main() 