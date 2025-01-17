import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import logging
from segmentation.segmenter import SemanticSegmenter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Segment Annotate - Image Segmentation and Analysis Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt",
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                       help="Path to SAM2 model config")
    
    # Pipeline selection
    parser.add_argument("--pipeline", type=str, choices=["auto", "gemini"], default="auto",
                       help="Pipeline to use (auto: automatic segmentation, gemini: Gemini-guided)")
    
    # Automatic segmentation options
    parser.add_argument("--max-objects", type=int, default=20,
                       help="Maximum number of objects to detect")
    
    # Gemini pipeline options
    parser.add_argument("--point-offset", type=float, default=0.2,
                       help="Point offset ratio for Gemini pipeline (0.2 = 20%%)")
    parser.add_argument("--temperature", type=float, default=0.5,
                       help="Temperature for Gemini generation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with visualizations")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize segmenter
    logger.info("Initializing segmenter...")
    segmenter = SemanticSegmenter(args.checkpoint, args.model_cfg)
    
    # Load and set image
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image)
    segmenter.set_image(image)
    
    # Run selected pipeline
    if args.pipeline == "auto":
        logger.info("Running automatic segmentation pipeline")
        masks, scores = segmenter.segment_automatic(max_objects=args.max_objects)
        logger.info(f"Generated {len(masks)} masks")
        
    else:  # gemini pipeline
        logger.info("Running Gemini-guided segmentation pipeline")
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        masks, scores = segmenter.segment_with_gemini(
            gemini_api_key=gemini_api_key,
            max_items=args.max_objects,
            point_offset_ratio=args.point_offset,
            temperature=args.temperature,
            debug=args.debug
        )
        logger.info(f"Generated {len(masks)} masks using Gemini guidance")
    
    # Save results
    logger.info(f"Saving results to {args.output_dir}")
    metadata = segmenter.save_results(args.output_dir)
    logger.info(f"Results saved successfully. Visualization at: {metadata['visualization_path']}")
    
    # Print summary
    print("\nSegmentation Summary:")
    print(f"- Pipeline: {args.pipeline}")
    print(f"- Objects detected: {len(masks)}")
    print(f"- Average confidence: {scores.mean():.3f}")
    print(f"- Results saved to: {args.output_dir}")
    if args.debug:
        print(f"- Debug files saved to: debug/gemini_debug_*")

if __name__ == "__main__":
    main() 