import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os
import datetime
import json
from typing import List, Dict, Optional

class SemanticSegmenter:
    def __init__(self, model_name="facebook/sam2.1-hiera-large", device="cuda", batch_size=64):
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        
        print("\nInitializing SAM2 model...")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        
        # Initialize with default settings
        self.mask_generator = self._create_mask_generator()
        
    def _create_mask_generator(self, config=None):
        """Create mask generator with settings optimized for complete object segmentation"""
        # Optimized settings for larger, complete objects
        default_config = {
            'points_per_side': 48,          # Higher density for better detail
            'points_per_batch': self.batch_size,  # Balanced batch size
            'pred_iou_thresh': 0.97,        # Very high to ensure accurate boundaries
            'stability_score_thresh': 0.98,  # High stability to prevent over-segmentation
            'stability_score_offset': 0.8,   # Higher offset for more stable masks
            'crop_n_layers': 1,             # Single layer since objects are of similar scale
            'box_nms_thresh': 0.45,         # Lower NMS to treat shelf as one unit
            'min_mask_region_area': 500.0,  # Higher minimum area to avoid small segments
            'use_m2m': True                 # Enable mask refinement for better boundaries
        }
        
        # Use provided config or default
        config = config or default_config
        
        print("\nInitializing mask generator with settings:")
        for key, value in config.items():
            print(f"- {key}: {value}")
        
        return SAM2AutomaticMaskGenerator.from_pretrained(
            self.model_name,
            **config
        )

    def generate_masks(self, image, monitor_memory=True):
        print("\nStarting mask generation...")
        print(f"Input image shape: {image.shape}")
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Resize image if needed and get scale
        resized_image, scale = self._resize_image(image)
        if scale != 1.0:
            print(f"Resized image from {(original_height, original_width)} to {resized_image.shape[:2]}")
        
        try:
            print("Generating masks with SAM2...")
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                masks = self.mask_generator.generate(resized_image)
                print(f"Generated {len(masks)} initial masks")
                
                # Resize masks back to original dimensions if needed
                if scale != 1.0:
                    print("Rescaling masks to original dimensions...")
                    for i, mask in enumerate(masks):
                        mask['segmentation'] = cv2.resize(
                            mask['segmentation'].astype(np.uint8), 
                            (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        mask['bbox'] = [
                            int(coord / scale) for coord in mask['bbox']
                        ]
                        mask['area'] = int(mask['area'] / (scale * scale))
                    print("Mask rescaling complete")
                
        except RuntimeError as e:
            print(f"Error during mask generation: {str(e)}")
            raise e
        
        print("Mask generation completed successfully")
        return masks

    def _resize_image(self, image, max_size=1024):
        """Resize image if larger than max_size while maintaining aspect ratio"""
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height)), scale
        return image, 1.0

    def post_process_masks(self, masks, min_area=1000, max_overlap=0.3):
        print("\nStarting mask post-processing...")
        print(f"Input masks: {len(masks)}")
        print(f"Parameters: min_area={min_area}, max_overlap={max_overlap}")
        
        processed_masks = []
        skip_indices = set()
        
        # Sort masks by area (largest first)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        merged_count = 0
        for i, mask1 in enumerate(sorted_masks):
            if i in skip_indices:
                continue
            
            current_mask = mask1['segmentation'].copy()
            current_area = mask1['area']
            
            # Merge small overlapping masks
            if current_area < min_area:
                for j, mask2 in enumerate(sorted_masks[i+1:], i+1):
                    if j in skip_indices:
                        continue
                    
                    overlap = np.logical_and(current_mask, mask2['segmentation']).sum()
                    overlap_ratio = overlap / min(current_area, mask2['area'])
                    
                    if overlap_ratio > max_overlap:
                        current_mask = np.logical_or(current_mask, mask2['segmentation'])
                        current_area = current_mask.sum()
                        skip_indices.add(j)
            
            # Create new mask entry
            new_mask = mask1.copy()
            new_mask['segmentation'] = current_mask
            new_mask['area'] = current_area
            processed_masks.append(new_mask)
        
        print(f"Post-processing complete:")
        print(f"- Original masks: {len(masks)}")
        print(f"- Processed masks: {len(processed_masks)}")
        print(f"- Merged masks: {merged_count}")
        
        return processed_masks

    def save_masks(self, masks: List[Dict], output_dir: str = "segmentation_output"):
        """Save masks and their metadata"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        
        # Save individual masks and create metadata
        metadata = []
        for idx, mask in enumerate(masks):
            # Save binary mask as PNG
            mask_filename = f"mask_{idx:04d}.png"
            mask_path = os.path.join(output_dir, "masks", mask_filename)
            
            # Convert boolean mask to uint8 (0 or 255)
            mask_img = mask['segmentation'].astype(np.uint8) * 255
            cv2.imwrite(mask_path, mask_img)
            
            # Store metadata
            metadata.append({
                'mask_id': idx,
                'filename': os.path.join("masks", mask_filename),
                'bbox': mask['bbox'],
                'area': int(mask['area']),
                'predicted_iou': float(mask['predicted_iou']),
                'stability_score': float(mask['stability_score'])
            })
        
        # Save metadata
        with open(os.path.join(output_dir, 'masks_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

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