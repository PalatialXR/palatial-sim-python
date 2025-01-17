import os
import cv2
import datetime
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
from segmentation.segmenter import SemanticSegmenter
from segmentation.description_generator import DescriptionGenerator
from conversion.meshy_converter import MeshyConverter

def get_results_dir() -> Path:
    """Get the base results directory, creating it if necessary"""
    base_dir = Path(__file__).parent.parent / 'results'
    base_dir.mkdir(exist_ok=True)
    return base_dir

def create_output_dirs(timestamp: str) -> tuple[Path, Path]:
    """Create and return paths for segmentation and model outputs"""
    base_dir = get_results_dir()
    seg_dir = base_dir / 'segmentation' / f'output_{timestamp}'
    model_dir = base_dir / 'models' / f'output_{timestamp}'
    
    seg_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    return seg_dir, model_dir

def check_environment():
    """Check and validate environment setup"""
    # Find .env file
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        print("Warning: .env file not found. Creating template...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("MESHY_API_KEY=your_meshy_api_key_here\n")
        print("Created .env template. Please fill in your API keys.")
        return False
    
    # Load environment variables
    load_dotenv(env_path)
    
    # Check required keys
    api_key = os.getenv('OPENAI_API_KEY')
    meshy_api_key = os.getenv('MESHY_API_KEY')
    
    if not api_key or api_key == 'your_openai_api_key_here':
        print("Warning: OpenAI API key not set in .env file")
        return False
    
    if not meshy_api_key or meshy_api_key == 'your_meshy_api_key_here':
        print("Warning: Meshy API key not set in .env file")
        return False
    
    return True

def main():
    # Check environment setup
    env_ready = check_environment()
    if not env_ready:
        print("\nPlease set up your environment variables in .env file and try again.")
        return
    
    api_key = os.getenv('OPENAI_API_KEY')
    meshy_api_key = os.getenv('MESHY_API_KEY')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process image segmentation and 3D conversion')
    parser.add_argument('--image', default='shelves.jpg', help='Input image path')
    parser.add_argument('--segmentation-dir', help='Directory containing segmentation results')
    parser.add_argument('--output-dir', help='Directory for 3D model outputs')
    args = parser.parse_args()
    
    # Validate input image
    if not args.segmentation_dir and not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        return
    
    # Initialize segmenter
    try:
        segmenter = SemanticSegmenter(
            model_name="facebook/sam2.1-hiera-large",
            batch_size=32  
        )
    except Exception as e:
        print(f"Error initializing segmenter: {str(e)}")
        return
    
    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # If we're doing new segmentation
        if not args.segmentation_dir:
            # Load and process image
            image = cv2.imread(args.image)
            if image is None:
                raise ValueError(f"Could not load image from {args.image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create output directories
            seg_dir, model_dir = create_output_dirs(timestamp)
            segmentation_dir = str(seg_dir)
            output_dir = str(model_dir)
            
            print(f"\nProcessing image: {args.image}")
            print(f"Output directory: {segmentation_dir}")
            
            # Generate and save masks
            masks = segmenter.generate_masks(image)
            masks = segmenter.post_process_masks(masks, min_area=1000, max_overlap=0.3)
            segmenter.save_masks(masks, segmentation_dir)
            
            # Get descriptions if API key available
            if api_key:
                description_generator = DescriptionGenerator(api_key)
                descriptions = description_generator.generate_descriptions(
                    image, masks, segmentation_dir
                )
            
            # Save visualization
            segmenter.visualize_segmentation(image, masks)
            plt.savefig(str(seg_dir / 'segmentation_visualization.png'))
        else:
            # Validate segmentation directory
            if not os.path.exists(args.segmentation_dir):
                print(f"Error: Segmentation directory not found: {args.segmentation_dir}")
                return
                
            segmentation_dir = args.segmentation_dir
            output_dir = args.output_dir or str(get_results_dir() / 'models' / f'output_{timestamp}')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert to 3D if we have Meshy API key
        if meshy_api_key:
            print(f"\nStarting 3D conversion process...")
            print(f"Input segmentation: {segmentation_dir}")
            print(f"Output directory: {output_dir}")
            
            converter = MeshyConverter(meshy_api_key)
            conversion_results = converter.convert_segments_to_3d(
                segmentation_dir=segmentation_dir,
                output_dir=output_dir
            )
            
            print("\nConversion complete!")
            print(f"Results saved to: {output_dir}")
        else:
            print("\nSkipping 3D conversion: No Meshy API key provided")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 