import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import logging
from segmentation.segmenter import SemanticSegmenter
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Segment objects in an image using SAM2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--pipeline", type=str, choices=["auto", "gemini"], default="auto",
                      help="Segmentation pipeline to use")
    parser.add_argument("--max-objects", type=int, default=10,
                      help="Maximum number of objects to detect")
    parser.add_argument("--point-offset", type=float, default=0.2,
                      help="Offset ratio for point prompts (0.2 = 20%)")
    parser.add_argument("--temperature", type=float, default=0.5,
                      help="Temperature for Gemini generation")
    parser.add_argument("--debug", action="store_true",
                      help="Save debug visualizations")
    parser.add_argument("--mode", type=str, choices=["2d", "3d", "points"], default="2d",
                      help="Detection mode for Gemini pipeline (2d/3d boxes or direct points)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize segmenter
    segmenter = SemanticSegmenter()
    
    # Load and set image
    image = Image.open(args.image)
    segmenter.set_image(image)
    
    # Run segmentation
    if args.pipeline == "auto":
        masks, scores = segmenter.segment_automatic(max_objects=args.max_objects)
        logger.info(f"Generated {len(masks)} masks using automatic segmentation")
    else:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        if args.mode == "points":
            masks, scores = segmenter.segment_with_points(
                gemini_api_key=gemini_api_key,
                max_items=args.max_objects,
                temperature=args.temperature,
                debug=args.debug
            )
            logger.info(f"Generated {len(masks)} masks using Gemini point guidance")
        else:
            masks, scores = segmenter.segment_with_gemini(
                gemini_api_key=gemini_api_key,
                max_items=args.max_objects,
                point_offset_ratio=args.point_offset,
                temperature=args.temperature,
                debug=args.debug,
                mode=args.mode
            )
            logger.info(f"Generated {len(masks)} masks using Gemini {args.mode} box guidance")
    
    # Save results
    metadata = segmenter.save_results(args.output_dir)
    logger.info(f"Saved segmentation results to {args.output_dir}")
    
if __name__ == "__main__":
    main() 