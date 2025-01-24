import os
import base64
import json
import datetime
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from openai import OpenAI
from models.physical_properties import ObjectDescription

class DescriptionGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_descriptions(
        self,
        image,
        masks: List[Dict],
        output_dir: str,
        iou_threshold: float = 0.95,
        stability_threshold: float = 0.97
    ) -> List[ObjectDescription]:
        """Get detailed physical descriptions for high-confidence masks"""
        print("\nStarting GPT-4o description generation...")
        
        high_confidence_masks = self._get_high_confidence_masks(
            masks,
            iou_threshold,
            stability_threshold
        )
        
        if not high_confidence_masks:
            print("No high-confidence masks found. Skipping description generation.")
            return []
        
        metadata = self._save_transparent_masks(image, high_confidence_masks, output_dir)
        descriptions = []
        responses_log = []
        
        total_segments = len(metadata)
        successful_descriptions = 0
        failed_descriptions = 0
        skipped_unknowns = 0
        
        for mask_meta in metadata:
            try:
                segment_path = os.path.join(output_dir, mask_meta['filename'])
                with open(segment_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                print(f"\nProcessing segment {mask_meta['mask_id'] + 1}/{total_segments}")
                
                completion = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[{
                        "role": "system",
                        "content": """You are a computer vision system that analyzes objects and their physical properties.
                        Provide detailed physical analysis only when highly confident. Use 'unknown' for uncertain cases or objects with unclear images. For objects that are recognizable, try estimating properties based on the image and the real world product.
                        All numeric values should be within reasonable physical bounds:
                        - Friction, roughness, and elasticity should be between 0 and 1
                        - Dimensions should be in meters
                        - Mass should be in kilograms
                        - Density should be in kg/m³"""
                    }, {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                        ]
                    }],
                    response_format=ObjectDescription
                )
                
                description = completion.choices[0].message.parsed
                
                if description.category.lower() == 'unknown':
                    print(f"Skipping segment {mask_meta['mask_id'] + 1} - insufficient information")
                    skipped_unknowns += 1
                    continue
                
                descriptions.append(description)
                response_entry = {
                    **mask_meta,
                    'description': description.model_dump(),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                responses_log.append(response_entry)
                
                successful_descriptions += 1
                print(f"Successfully processed segment {mask_meta['mask_id'] + 1}")
                
            except Exception as e:
                failed_descriptions += 1
                print(f"Error processing segment {mask_meta['mask_id'] + 1}: {str(e)}")
        
        print(f"\nDescription generation complete:")
        print(f"- Successfully processed: {successful_descriptions}")
        print(f"- Skipped unknown: {skipped_unknowns}")
        print(f"- Failed: {failed_descriptions}")
        
        if responses_log:
            response_path = os.path.join(output_dir, 'segment_descriptions.json')
            with open(response_path, 'w') as f:
                json.dump(responses_log, f, indent=2, default=str)
            print(f"Saved detailed descriptions to {response_path}")
        
        return descriptions

    def _get_high_confidence_masks(
        self,
        masks: List[Dict],
        iou_threshold: float,
        stability_threshold: float
    ) -> List[Dict]:
        """Filter masks to keep only those with very high confidence scores"""
        print("\nFiltering for high-confidence masks...")
        high_confidence_masks = []
        
        for mask in masks:
            if (mask['predicted_iou'] >= iou_threshold and 
                mask['stability_score'] >= stability_threshold):
                high_confidence_masks.append(mask)
        
        print(f"Found {len(high_confidence_masks)} high-confidence masks out of {len(masks)} total")
        print(f"Using thresholds: IoU >= {iou_threshold}, Stability >= {stability_threshold}")
        
        return high_confidence_masks

    def _save_transparent_masks(
        self,
        image,
        masks: List[Dict],
        output_dir: str
    ) -> List[Dict]:
        """Save transparent versions of the masks"""
        print(f"\nSaving transparent masks to {output_dir}")
        print(f"Processing {len(masks)} masks...")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "segments"), exist_ok=True)
        
        metadata = []
        for idx, mask in enumerate(masks):
            try:
                # Create masked image with transparent background
                masked_img = np.zeros((*image.shape[:2], 4), dtype=np.uint8)  # RGBA
                
                # Handle the RGB to RGBA conversion properly
                mask_bool = mask['segmentation']
                masked_region = image[mask_bool]
                
                # Create RGBA array with proper shape
                rgba_region = np.zeros((len(masked_region), 4), dtype=np.uint8)
                rgba_region[:, :3] = masked_region  # Copy RGB values
                rgba_region[:, 3] = 255  # Set alpha channel to fully opaque
                
                # Assign the RGBA values to the masked region
                masked_img[mask_bool] = rgba_region
                
                # Crop to bounding box
                x, y, w, h = mask['bbox']
                cropped_img = masked_img[y:y+h, x:x+w]
                
                if cropped_img.size == 0:
                    print(f"Warning: Empty segment detected for mask {idx}")
                    continue
                
                # Save as transparent PNG
                segment_filename = f"segment_{idx:04d}.png"
                segment_path = os.path.join(output_dir, "segments", segment_filename)
                
                # Convert to PIL and save
                segment_pil = Image.fromarray(cropped_img, 'RGBA')
                segment_pil.save(segment_path)
                
                # Store metadata
                metadata.append({
                    'mask_id': idx,
                    'filename': os.path.join("segments", segment_filename),
                    'bbox': mask['bbox'],
                    'area': int(mask['area']),
                    'predicted_iou': float(mask['predicted_iou']),
                    'stability_score': float(mask['stability_score'])
                })
                
            except Exception as e:
                print(f"Error saving segment {idx}: {str(e)}")
                continue
        
        print(f"Successfully saved {len(metadata)} segments")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'segments_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
        
        return metadata 