import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import logging
from segmentation.segmenter import SemanticSegmenter
from segmentation.interactive_segmenter import InteractiveSegmenter
from segmentation.interactive_ui import InteractiveSegmentationUI
from segmentation.description_generator import DescriptionGenerator
# from depth.depth_estimator import DepthEstimator
import torch
import datetime
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Segment objects and estimate depth in images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--pipeline", type=str, choices=["auto", "gemini", "depth", "interactive"], default="auto",
                      help="Pipeline to use (auto/gemini/interactive segmentation or depth estimation)")
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
    parser.add_argument("--generate-descriptions", action="store_true",
                      help="Generate physical descriptions for segmented objects using GPT-4")
    # parser.add_argument("--prompt-depth", type=str,
    #                   help="Path to prompt depth image (e.g., LiDAR depth)")
    parser.add_argument("--points", type=str,
                      help="Comma-separated list of x,y coordinates for interactive mode (e.g., '100,100,200,200')")
    parser.add_argument("--labels", type=str,
                      help="Comma-separated list of point labels (1 for foreground, 0 for background)")
    parser.add_argument("--box", type=str,
                      help="Comma-separated box coordinates x1,y1,x2,y2 for interactive mode")
    parser.add_argument("--convert-3d", action="store_true",
                      help="Convert segments to 3D models")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load and prepare image
    image = Image.open(args.image)
    
    if args.pipeline == "interactive":
        logger.info("Running interactive segmentation pipeline")
        
        # Initialize interactive segmenter
        segmenter = InteractiveSegmenter()
        segmenter.set_image(image)
        
        # Create and run interactive UI
        ui = InteractiveSegmentationUI(segmenter)
        masks, scores = ui.run()
        
        if masks is not None and scores is not None:
            # Create output directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.output_dir) / "interactive" / f"output_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization of all masks
            vis_image = segmenter.visualize_masks(
                masks=masks,
                scores=scores,
                points=None,  # Don't show points in final visualization
                labels=None,
                box=None
            )
            vis_path = output_dir / "segmentation_visualization.png"
            Image.fromarray(vis_image).save(vis_path)
            logger.info(f"Saved visualization to {vis_path}")
            
            # Save individual masks and segments
            masks_dir = output_dir / "masks"
            segments_dir = output_dir / "segments"
            masks_dir.mkdir(exist_ok=True)
            segments_dir.mkdir(exist_ok=True)
            
            # Save metadata about the segmentation
            metadata = {
                "timestamp": timestamp,
                "num_masks": len(masks),
                "scores": scores.tolist()
            }
            
            # Process each mask
            segments = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Save binary mask
                mask_path = masks_dir / f"mask_{i:03d}.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                
                # Create and save segment
                segment = np.array(image)
                mask_bool = mask.astype(bool)
                segment[~mask_bool] = 0  # Zero out non-mask pixels
                segment_path = segments_dir / f"segment_{i:03d}.png"
                Image.fromarray(segment).save(segment_path)
                
                segments.append({
                    "mask_path": str(mask_path),
                    "segment_path": str(segment_path),
                    "score": float(score)
                })
                logger.info(f"Saved mask {i} with score {score:.3f}")
            
            # Save metadata
            metadata["segments"] = segments
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Generate descriptions using GPT-4 if requested
            if args.generate_descriptions:
                try:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    if not openai_api_key:
                        raise ValueError("OPENAI_API_KEY not found in environment")
                        
                    logger.info("Generating object descriptions with GPT-4...")
                    description_generator = DescriptionGenerator(api_key=openai_api_key)
                    
                    # Convert masks to format expected by description generator
                    mask_dicts = []
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        # Get mask dimensions
                        mask_bool = mask.astype(bool)
                        rows = np.any(mask_bool, axis=1)
                        cols = np.any(mask_bool, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        
                        mask_dicts.append({
                            'segmentation': mask_bool,
                            'area': int(np.sum(mask_bool)),
                            'bbox': [int(cmin), int(rmin), int(cmax), int(rmax)],  # [x1, y1, x2, y2]
                            'predicted_iou': 1.0,  # User-verified masks have perfect IoU
                            'stability_score': 1.0  # User-verified masks are perfectly stable
                        })
                    
                    descriptions = description_generator.generate_descriptions(
                        image=np.array(image),
                        masks=mask_dicts,
                        output_dir=str(output_dir),
                        iou_threshold=0.0,  # No threshold since these are user-verified
                        stability_threshold=0.0  # No threshold since these are user-verified
                    )
                    
                    logger.info(f"Generated descriptions for {len(descriptions)} objects")
                    
                except Exception as e:
                    logger.error(f"Failed to generate descriptions: {str(e)}")
            else:
                logger.info("Skipping description generation (use --generate-descriptions to enable)")
            
            # Convert segments to 3D if requested
            if args.convert_3d:
                logger.info("Converting segments to 3D models")
                try:
                    meshy_api_key = os.getenv("MESHY_API_KEY")
                    if not meshy_api_key:
                        raise ValueError("MESHY_API_KEY not found in environment")
                        
                    from conversion.meshy_converter import MeshyConverter
                    converter = MeshyConverter(api_key=meshy_api_key)
                    
                    # Convert segments to 3D
                    models_dir = output_dir / "3d_models"
                    models_dir.mkdir(exist_ok=True)
                    
                    conversion_results = converter.convert_segments_to_3d(
                        segmentation_dir=str(output_dir),
                        output_dir=str(models_dir)
                    )
                    
                    # Save conversion results
                    results_path = models_dir / "conversion_results.json"
                    with open(results_path, "w") as f:
                        json.dump(conversion_results, f, indent=2)
                        
                    logger.info(f"Saved 3D models and results to {models_dir}")
                    
                except Exception as e:
                    logger.error(f"Failed to convert segments to 3D: {str(e)}")
            
            logger.info(f"Completed interactive segmentation pipeline. Results saved to {output_dir}")
            
    elif args.pipeline == "depth":
        # Run depth estimation
        logger.info("Running depth estimation pipeline")
        
        # Initialize depth estimator
        depth_estimator = DepthEstimator()
        
        # Load prompt depth if provided
        prompt_depth = None
        if args.prompt_depth:
            logger.info(f"Using prompt depth from {args.prompt_depth}")
            prompt_depth = args.prompt_depth
        
        # Estimate depth
        depth = depth_estimator.estimate_depth(image, prompt_depth)
        
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / "depth" / f"output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save depth visualization
        vis_path = output_dir / "depth_visualization.png"
        depth_estimator.save_depth_visualization(
            depth,
            vis_path,
            prompt_depth=prompt_depth,
            image=image
        )
        logger.info(f"Saved depth visualization to {vis_path}")
        
    else:
        # Run segmentation pipeline
        segmenter = SemanticSegmenter()
        segmenter.set_image(image)
        
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
        
        # Save segmentation results
        metadata = segmenter.save_results(args.output_dir)
        logger.info(f"Saved segmentation results to {args.output_dir}")
    
if __name__ == "__main__":
    main() 