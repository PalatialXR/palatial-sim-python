import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
from meshy import MeshyAPIClient, ImageTo3DRequest, TopologyType, SymmetryMode, PollingConfig

class MeshyConverter:
    def __init__(self, api_key: str):
        self.client = MeshyAPIClient(api_key=api_key)
        
    def convert_segments_to_3d(
        self,
        segmentation_dir: Union[str, Path],
        output_dir: Union[str, Path],
        polling_config: Optional[PollingConfig] = None
    ):
        """Convert segmented objects to 3D models using Meshy API"""
        # Convert to Path objects
        segmentation_dir = Path(segmentation_dir)
        output_dir = Path(output_dir)
        
        print("\nStarting 3D conversion process...")
        print(f"Input directory: {segmentation_dir}")
        print(f"Output directory: {output_dir}")
        
        # Load segment descriptions
        descriptions_path = segmentation_dir / 'segment_descriptions.json'
        if not descriptions_path.exists():
            raise FileNotFoundError(f"No segment descriptions found at {descriptions_path}")
        
        with open(descriptions_path, 'r') as f:
            segment_descriptions = json.load(f)
        print(f"Loaded {len(segment_descriptions)} segment descriptions")
        
        # Create output directory
        models_dir = output_dir / "3d_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure polling
        if polling_config is None:
            polling_config = PollingConfig(
                interval=5.0,
                timeout=1200.0,
                on_progress=self._default_progress_callback
            )
        
        # Process segments
        conversion_results = []
        for segment in segment_descriptions:
            try:
                result = self._process_segment(
                    segment,
                    segmentation_dir,
                    models_dir,
                    polling_config
                )
                if result:
                    conversion_results.append(result)
                    time.sleep(30)  # Rate limiting between successful requests
                    
            except Exception as e:
                print(f"Error processing segment: {str(e)}")
                conversion_results.append({
                    "segment_id": segment.get('mask_id'),
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save results
        results_path = output_dir / "3d_conversion_results.json"
        with open(results_path, 'w') as f:
            json.dump(conversion_results, f, indent=2)
        
        successful = sum(1 for r in conversion_results if r.get('status') == 'success')
        failed = sum(1 for r in conversion_results if r.get('status') == 'failed')
        
        print("\n3D conversion complete!")
        print(f"- Total segments processed: {len(segment_descriptions)}")
        print(f"- Successful conversions: {successful}")
        print(f"- Failed conversions: {failed}")
        print(f"Results saved to {results_path}")
        
        return conversion_results

    def _process_segment(
        self,
        segment: Dict,
        segmentation_dir: Path,
        models_dir: Path,
        polling_config: PollingConfig
    ) -> Optional[Dict]:
        """Process a single segment for 3D conversion"""
        idx = segment.get('mask_id')
        description = segment.get('description', {})
        filename = segment.get('filename', '')
        
        if not all([idx is not None, description, filename]):
            print(f"Skipping invalid segment data: {segment}")
            return None
        
        segment_path = segmentation_dir / "segments" / Path(filename).name
        if not segment_path.exists():
            print(f"Skipping segment {idx}: Image not found at {segment_path}")
            return None
        
        print(f"\nProcessing segment {idx}:")
        print(f"- Category: {description.get('category', 'unknown')}")
        print(f"- File: {segment_path.name}")
        
        # Configure conversion parameters
        params = self._get_conversion_params(description)
        
        try:
            request = ImageTo3DRequest.from_file(
                file_path=str(segment_path),
                topology=params['topology'],
                target_polycount=params['polycount'],
                should_remesh=True,
                enable_pbr=True,
                should_texture=True,
                symmetry_mode=params['symmetry']
            )
            
            response = self.client.create_image_to_3d_task(request)
            task_id = response["result"]["id"]
            print(f"Created task {task_id}")
            
            final_status = self.client.wait_for_completion(task_id, polling_config)
            
            if 'obj' in final_status["model_urls"]:
                obj_url = final_status["model_urls"]["obj"]
                category_name = description.get('category', 'unknown').lower()
                obj_path = models_dir / f"{category_name}_{idx:04d}.obj"
                
                self._download_file(obj_url, str(obj_path))
                print(f"Successfully saved OBJ model to {obj_path}")
                
                return {
                    "segment_id": idx,
                    "category": category_name,
                    "task_id": task_id,
                    "status": "success",
                    "obj_file": str(obj_path),
                    "polycount": params['polycount'],
                    "topology": params['topology'].value,
                    "symmetry": params['symmetry'].value,
                    "volume": params.get('volume', 0.0),
                    "original_segment": str(segment_path)
                }
            else:
                raise ValueError(f"No OBJ file in response for task {task_id}")
                
        except Exception as e:
            print(f"Error in 3D conversion for segment {idx}: {str(e)}")
            return {
                "segment_id": idx,
                "category": description.get('category', 'unknown'),
                "status": "failed",
                "error": str(e)
            }

    def _get_conversion_params(self, description: Dict) -> Dict:
        """Calculate conversion parameters based on object properties"""
        properties = description.get('properties', {})
        geometric_props = properties.get('geometry', {})
        
        is_symmetrical = geometric_props.get('symmetrical', False)
        topology = TopologyType.QUAD if is_symmetrical else TopologyType.TRIANGLE
        symmetry = SymmetryMode.ON if is_symmetrical else SymmetryMode.AUTO
        
        # Calculate polycount based on object complexity
        volume = geometric_props.get('volume', 0.0)
        base_polycount = 30000
        
        category = description.get('category', '').lower()
        if category in ['box', 'container', 'simple_object']:
            base_polycount = max(15000, int(volume * 100000))
        elif category in ['detailed_object', 'complex_shape']:
            base_polycount = max(45000, int(volume * 200000))
        
        base_polycount = min(max(base_polycount, 10000), 100000)
        
        return {
            'topology': topology,
            'symmetry': symmetry,
            'polycount': base_polycount,
            'volume': volume
        }

    def _download_file(self, url: str, destination: str):
        """Download a file from URL to local destination"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _default_progress_callback(self, status: Dict):
        """Default callback for progress updates"""
        progress = status.get("progress", 0)
        task_id = status.get("id", "unknown")
        print(f"Task {task_id} progress: {progress}%") 