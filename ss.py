import torch
import numpy as np
import cv2
from PIL import Image
import requests
import json
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from openai import OpenAI
import io
import base64
from memory_utils import clear_gpu_memory, optimize_memory_usage, print_memory_stats
import os 
import datetime

def resize_image(image, max_size=1024):
    """Resize image if larger than max_size while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height)), scale
    return image, 1.0

class SemanticSegmenter:
    def __init__(self, model_name="facebook/sam2-hiera-large", device="cuda", batch_size=16):
        self.device = device
        optimize_memory_usage(True)
        
        print("\nInitializing SAM2 model...")
        print_memory_stats()
        
        # Initialize with memory-optimized settings
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        self.mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
            model_name,
            points_per_side=16,          # Reduced from default
            points_per_batch=batch_size,  # Smaller batch size
            pred_iou_thresh=0.86,        # Slightly higher threshold
            stability_score_thresh=0.92,
            stability_score_offset=0.8,
            box_nms_thresh=0.7,
            crop_n_layers=1,             # Minimum number of layers
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,    # Increased minimum area
            use_m2m=False               # Disable mask-to-mask refinement
        )
        
        self.batch_size = batch_size
        clear_gpu_memory()
        
    def generate_masks(self, image, monitor_memory=True):
        """Generate masks with proper resizing handling"""
        if monitor_memory:
            print("\nMemory before preprocessing:")
            print_memory_stats()
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Resize image if needed and get scale
        resized_image, scale = resize_image(image)
        
        clear_gpu_memory()
        
        try:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                masks = self.mask_generator.generate(resized_image)
                
                # Resize masks back to original dimensions if needed
                if scale != 1.0:
                    for mask in masks:
                        mask['segmentation'] = cv2.resize(
                            mask['segmentation'].astype(np.uint8), 
                            (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        # Scale bounding box coordinates
                        mask['bbox'] = [
                            int(coord / scale) for coord in mask['bbox']
                        ]
                        mask['area'] = int(mask['area'] / (scale * scale))
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_gpu_memory(aggressive=True)
                print("\nRetrying with more aggressive memory optimization...")
                resized_image, scale = resize_image(image, max_size=768)
                with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                    masks = self.mask_generator.generate(resized_image)
                    
                    # Resize masks back to original dimensions
                    for mask in masks:
                        mask['segmentation'] = cv2.resize(
                            mask['segmentation'].astype(np.uint8), 
                            (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        # Scale bounding box coordinates
                        mask['bbox'] = [
                            int(coord / scale) for coord in mask['bbox']
                        ]
                        mask['area'] = int(mask['area'] / (scale * scale))
        
        if monitor_memory:
            print("\nMemory after mask generation:")
            print_memory_stats()
        
        return masks

    def get_gpt4v_descriptions(self, image, masks, api_key, output_file="segmentation_responses.json"):
        """Get GPT-4V descriptions with memory management and save responses"""
        if not api_key:
            print("Warning: No OpenAI API key provided. Skipping GPT-4V descriptions.")
            return [{"error": "No API key provided"}] * len(masks)
            
        client = OpenAI(api_key=api_key)
        descriptions = []
        responses_log = []
        
        # Process in smaller batches
        batch_size = 5
        for i in range(0, len(masks), batch_size):
            batch_masks = masks[i:i + batch_size]
            
            for mask_data in batch_masks:
                try:
                    # Get bounding box coordinates
                    x, y, w, h = mask_data['bbox']
                    
                    # Create masked image with transparent background
                    masked_img = np.zeros((*image.shape[:2], 4), dtype=np.uint8)  # RGBA
                    mask = mask_data['segmentation']
                    
                    # Handle the RGB to RGBA conversion properly
                    masked_region = image[mask]
                    masked_img[mask] = np.column_stack([
                        masked_region,
                        np.full(len(masked_region), 255, dtype=np.uint8)
                    ])
                    
                    # Crop to bounding box
                    cropped_img = masked_img[y:y+h, x:x+w]
                    
                    # Ensure the cropped image is not empty
                    if cropped_img.size == 0:
                        print(f"Warning: Empty mask detected for mask {i + len(descriptions)}")
                        continue
                    
                    # Convert to PIL and save
                    try:
                        masked_pil = Image.fromarray(cropped_img, 'RGBA')
                    except Exception as e:
                        print(f"Error converting to PIL: {str(e)}")
                        print(f"Cropped image shape: {cropped_img.shape}")
                        print(f"Unique values: {np.unique(cropped_img)}")
                        continue
                    
                    # Save individual mask
                    mask_filename = f"masks/mask_{i + len(descriptions)}.png"
                    os.makedirs("masks", exist_ok=True)
                    masked_pil.save(mask_filename)
                    
                    # Convert to base64
                    buffered = io.BytesIO()
                    masked_pil.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    print(f"\nProcessing mask {i + len(descriptions) + 1}/{len(masks)}")
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",  # Changed from gpt-4o-mini
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """
                                    Analyze this image segment and provide:
                                    {
                                        "category": "object class or region type",
                                        "physical_properties": {
                                            "material": "primary material composition",
                                            "estimated_size": "approximate dimensions"
                                        },
                                        "confidence": "confidence level (0-1)"
                                    }
                                """},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                            ]
                        }],
                        max_tokens=200
                    )
                    
                    try:
                        description = json.loads(response.choices[0].message.content)
                    except json.JSONDecodeError:
                        description = response.choices[0].message.content
                    
                    descriptions.append(description)
                    
                    # Log response with metadata
                    response_entry = {
                        'mask_id': i + len(descriptions),
                        'bbox': mask_data['bbox'].tolist() if isinstance(mask_data['bbox'], np.ndarray) else mask_data['bbox'],
                        'area': int(mask_data['area']),
                        'mask_file': mask_filename,
                        'description': description,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    responses_log.append(response_entry)
                    
                    # Save responses periodically
                    if len(responses_log) % batch_size == 0:
                        with open(output_file, 'w') as f:
                            json.dump(responses_log, f, indent=2)
                    
                except Exception as e:
                    print(f"Error processing mask: {str(e)}")
                    print(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
                    print(f"Bbox: {mask_data['bbox']}")
                    descriptions.append({"error": str(e)})
                
                finally:
                    # Clean up
                    try:
                        del masked_img, cropped_img
                        if 'masked_pil' in locals():
                            del masked_pil
                        if 'buffered' in locals():
                            del buffered
                        if 'img_str' in locals():
                            del img_str
                    except Exception:
                        pass
                    clear_gpu_memory()
        
        # Save final responses
        with open(output_file, 'w') as f:
            json.dump(responses_log, f, indent=2)
        
        return descriptions

    def visualize_segmentation(self, image, masks):
        """Visualize segmentation with proper dimension handling"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        overlay = np.zeros((*image.shape[:2], 4))
        
        for mask in sorted_masks:
            color = np.concatenate([np.random.random(3), [0.35]])
            overlay[mask['segmentation']] = color
        
        plt.imshow(overlay)
        plt.axis('off')
        plt.tight_layout()

if __name__ == "__main__":
    # Set environment variable for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("\nInitial memory status:")
    print_memory_stats()
    
    # Initialize segmenter with conservative settings
    segmenter = SemanticSegmenter(
        model_name="facebook/sam2-hiera-large",
        batch_size=16  # Reduced batch size
    )
    
    # Load and process image
    image_path = "shelves.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    try:
        masks = segmenter.generate_masks(image)
        print(f"\nGenerated {len(masks)} masks successfully")
        
        # Get GPT-4V descriptions if API key is available
        if api_key:
            descriptions = segmenter.get_gpt4v_descriptions(
                image, 
                masks, 
                api_key,
                output_file=f"segmentation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Print some sample descriptions
            print("\nSample descriptions:")
            for i, desc in enumerate(descriptions[:3]):
                print(f"\nMask {i + 1}:", desc)
        
        # Visualize results
        segmenter.visualize_segmentation(image, masks)
        plt.savefig('segmentation_result.png', bbox_inches='tight', pad_inches=0)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print_memory_stats()