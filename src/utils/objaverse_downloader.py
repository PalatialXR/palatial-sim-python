import os
from typing import List, Dict, Any, Set, Tuple, Optional
import pandas as pd
import logging
import traceback
import requests
from pathlib import Path
import urllib.parse
import re
import objaverse
import pickle
import numpy as np
import trimesh
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import shutil
import tempfile
import json
import multiprocessing
from utils.category_manager import CategoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectViewer:
    def __init__(self, matches: List[Tuple[str, Dict, str]], object_name: str):
        """
        Initialize viewer with matches
        matches: List of tuples (uid, metadata, temp_file_path)
        """
        self.matches = matches
        self.object_name = object_name
        self.selected_match = None
        self.temp_dir = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"Select match for {object_name}")
        self.root.geometry("1200x800")
        
        # Create frames
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create split view for 2D and 3D
        self.viewer_frame = ttk.Frame(self.root)
        self.viewer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_frame = ttk.Frame(self.viewer_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.model_frame = ttk.Frame(self.viewer_frame)
        self.model_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add widgets
        self.setup_widgets()
        
        # Load first object
        self.current_index = 0
        self.load_current_object()
    
    def setup_widgets(self):
        # Info label with scrollbar
        info_frame = ttk.Frame(self.info_frame)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(info_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_label = tk.Text(info_frame, wrap=tk.WORD, height=10, yscrollcommand=scrollbar.set)
        self.info_label.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_label.yview)
        
        # Navigation buttons
        ttk.Button(self.button_frame, text="Previous", command=self.prev_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Next", command=self.next_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Select", command=self.select_current).pack(side=tk.RIGHT, padx=5)
        ttk.Button(self.button_frame, text="Skip", command=self.skip_current).pack(side=tk.RIGHT, padx=5)
    
    def load_current_object(self):
        uid, obj_data, temp_path = self.matches[self.current_index]
        
        # Update info with detailed metadata
        info_text = f"Match {self.current_index + 1}/{len(self.matches)}\n\n"
        
        # Basic information
        info_text += "Basic Information:\n"
        info_text += f"- Name: {obj_data.get('name', 'No name')}\n"
        info_text += f"- Category: {obj_data.get('category', 'Unknown')}\n"
        info_text += f"- Description: {obj_data.get('description', 'No description')}\n\n"
        
        # Mesh information if available
        if 'vertex_count' in obj_data:
            info_text += "Mesh Details:\n"
            info_text += f"- Vertices: {obj_data['vertex_count']:,}\n"
            info_text += f"- Faces: {obj_data['face_count']:,}\n"
            if 'volume' in obj_data:
                info_text += f"- Volume: {obj_data['volume']:.2f}\n"
            if 'bounding_box' in obj_data:
                dims = obj_data['bounding_box']
                info_text += f"- Dimensions (xyz): {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}\n\n"
        
        # Source information
        info_text += "Source Information:\n"
        info_text += f"- UID: {uid}\n"
        if 'mesh_url' in obj_data:
            info_text += f"- Mesh URL: {obj_data['mesh_url']}\n"
        if 'thumbnail_url' in obj_data:
            info_text += f"- Thumbnail URL: {obj_data['thumbnail_url']}\n"
        
        # Additional metadata if available
        if 'metadata' in obj_data:
            info_text += "\nAdditional Metadata:\n"
            for key, value in obj_data['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    info_text += f"- {key}: {value}\n"
        
        self.info_label.config(text=info_text)
        
        # Try to load and display thumbnail
        if 'thumbnail_url' in obj_data:
            try:
                response = requests.get(obj_data['thumbnail_url'])
                img = Image.open(io.BytesIO(response.content))
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                
                if hasattr(self, 'image_label'):
                    self.image_label.destroy()
                
                self.image_label = ttk.Label(self.image_frame, image=photo)
                self.image_label.image = photo
                self.image_label.pack(pady=10)
            except Exception as e:
                logger.error(f"Error loading thumbnail: {e}")
        
        # Try to load and display 3D model
        try:
            if hasattr(self, 'model_canvas'):
                self.model_canvas.destroy()
            
            # Load the model using trimesh
            mesh = trimesh.load(temp_path)
            
            # Create a simple OpenGL viewer
            scene = trimesh.Scene(mesh)
            
            # Create a canvas for the 3D viewer
            from trimesh.viewer import SceneViewer
            self.model_canvas = SceneViewer(scene, resolution=(400, 400), background=[1, 1, 1, 0])
            self.model_canvas.pack(in_=self.model_frame, pady=10)
            
        except Exception as e:
            logger.error(f"Error loading 3D model: {e}")
    
    def prev_object(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_object()
    
    def next_object(self):
        if self.current_index < len(self.matches) - 1:
            self.current_index += 1
            self.load_current_object()
    
    def select_current(self):
        self.selected_match = self.matches[self.current_index]
        self.root.quit()
    
    def skip_current(self):
        self.selected_match = None
        self.root.quit()
    
    def run(self) -> Tuple[str, Dict, str]:
        self.root.mainloop()
        self.root.destroy()
        return self.selected_match

class ObjaverseDownloader:
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """Initialize the ObjaverseDownloader."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.categories_cache_file = self.cache_dir / "lvis_categories.json"
        self.cached_categories = self._load_cached_categories()
        logger.info("Initialized ObjaverseDownloader")

    def _load_cached_categories(self) -> Dict[str, List[str]]:
        """Load cached LVIS categories if they exist"""
        if self.categories_cache_file.exists():
            try:
                with open(self.categories_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load categories cache: {e}")
        return {}

    def _save_categories_cache(self, categories: Dict[str, List[str]]):
        """Save LVIS categories to cache"""
        try:
            with open(self.categories_cache_file, 'w') as f:
                json.dump(categories, f)
            logger.info(f"Saved {len(categories)} categories to cache")
        except Exception as e:
            logger.error(f"Failed to save categories cache: {e}")

    def get_lvis_annotations(self) -> Dict[str, List[str]]:
        """Get LVIS annotations, using cache if available"""
        if self.cached_categories:
            logger.info("Using cached LVIS categories")
            return self.cached_categories
        
        logger.info("Loading LVIS annotations from Objaverse...")
        categories = objaverse.load_lvis_annotations()
        self._save_categories_cache(categories)
        self.cached_categories = categories
        return categories

    def validate_object_name(self, object_name: str) -> bool:
        if object_name.endswith('_'):
            logger.error(f"Invalid object name '{object_name}': Cannot end with bare underscore")
            return False
        return True

    def strip_number_suffix(self, name: str) -> str:
        match = re.match(r'^(.+?)_\d+$', name)
        if match:
            base_name = match.group(1)
            logger.info(f"Stripped number suffix from '{name}' to get '{base_name}'")
            return base_name
        return name

    def download_preview_objects(self, category: str, uids: List[str], temp_dir: str) -> List[Tuple[str, Dict, str]]:
        """Download objects for preview and return their paths"""
        preview_objects = []
        
        # Load annotations for the category
        annotations = objaverse.load_annotations(uids)
        
        # Randomly sample from uids to get more variety
        if len(uids) > 20:
            uids = np.random.choice(uids, 20, replace=False).tolist()
        
        # Download and process each object
        for uid in uids:
            obj_data = annotations.get(uid)
            if not obj_data:
                continue
            
            # Get full metadata for better information
            full_metadata = objaverse.get_metadata([uid])[uid]
            if full_metadata:
                obj_data.update(full_metadata)
            
            # Check for required files
            has_mesh = 'mesh' in obj_data
            has_images = (
                'thumbnails' in obj_data 
                and 'images' in obj_data['thumbnails'] 
                and len(obj_data['thumbnails']['images']) > 0
            )
            
            if has_mesh and has_images:
                try:
                    # Verify mesh URL is accessible
                    mesh_url = obj_data['mesh']
                    mesh_response = requests.head(mesh_url)
                    if mesh_response.status_code != 200:
                        logger.warning(f"Mesh URL not accessible for {uid}")
                        continue
                    
                    # Download the object
                    downloaded_paths = objaverse.load_objects(
                        uids=[uid],
                        download_processes=1
                    )
                    
                    if downloaded_paths and uid in downloaded_paths:
                        # Validate downloaded mesh
                        try:
                            mesh = trimesh.load(downloaded_paths[uid])
                            if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
                                logger.warning(f"Invalid mesh format for {uid}")
                                continue
                            if isinstance(mesh, trimesh.Trimesh) and (len(mesh.vertices) < 3 or len(mesh.faces) < 1):
                                logger.warning(f"Mesh has no geometry for {uid}")
                                continue
                        except Exception as e:
                            logger.warning(f"Failed to validate mesh for {uid}: {e}")
                            continue
                        
                        # Save 3D model
                        temp_path = os.path.join(temp_dir, f"{uid}.stl")
                        shutil.move(downloaded_paths[uid], temp_path)
                        
                        # Download and save thumbnail
                        thumbnail_url = obj_data['thumbnails']['images'][0].get('url', '')
                        if thumbnail_url:
                            try:
                                response = requests.get(thumbnail_url)
                                if response.status_code == 200:
                                    image_path = os.path.join(temp_dir, f"{uid}_thumb.jpg")
                                    with open(image_path, 'wb') as f:
                                        f.write(response.content)
                                    obj_data['local_thumbnail'] = image_path
                            except Exception as e:
                                logger.error(f"Error downloading thumbnail for {uid}: {e}")
                        
                        # Add additional metadata for display
                        obj_data['category'] = category
                        obj_data['thumbnail_url'] = thumbnail_url
                        obj_data['mesh_url'] = mesh_url
                        if isinstance(mesh, trimesh.Trimesh):
                            obj_data['vertex_count'] = len(mesh.vertices)
                            obj_data['face_count'] = len(mesh.faces)
                            obj_data['volume'] = mesh.volume
                            obj_data['bounding_box'] = mesh.bounding_box.extents.tolist()
                        
                        preview_objects.append((uid, obj_data, temp_path))
                        
                except Exception as e:
                    logger.error(f"Error downloading preview for {uid}: {e}")
        
        return preview_objects

    def find_matches(self, object_name: str, object_description: Dict[str, str]) -> List[Tuple[str, Dict, str]]:
        """Find matching 3D objects and download previews using LVIS category"""
        try:
            if not self.validate_object_name(object_name):
                return []
                
            logger.info(f"\nSearching for matches for object: {object_name}")
            logger.info(f"Description: {object_description}")
            
            # Create temporary directory for preview objects
            temp_dir = tempfile.mkdtemp()
            
            # Get LVIS annotations
            lvis_annotations = objaverse.load_lvis_annotations()
            
            # Direct category mapping for common objects
            direct_lvis_mapping = {
                'desk': ['desk', 'table', 'office_desk'],
                'table': ['table', 'desk', 'coffee_table', 'dining_table'],
                'chair': ['chair', 'office_chair', 'armchair'],
                'monitor': ['monitor', 'computer_monitor', 'display'],
                'keyboard': ['keyboard', 'computer_keyboard'],
                'mouse': ['mouse', 'computer_mouse'],
                'computer': ['computer', 'desktop_computer', 'laptop'],
                'backpack': ['backpack', 'bag'],
                'case': ['case', 'container', 'box']
            }
            
            # Get base object name without number suffix
            base_name = self.strip_number_suffix(object_name)
            
            # Try direct category match first
            candidate_uids = []
            
            # Check direct mapping first
            if base_name in direct_lvis_mapping:
                for category in direct_lvis_mapping[base_name]:
                    if category in lvis_annotations:
                        candidate_uids.extend(lvis_annotations[category])
                        logger.info(f"Found matches in LVIS category: {category}")
            
            # If no direct matches, try the category field
            if not candidate_uids and 'category' in object_description:
                category = object_description['category'].lower()
                if category in direct_lvis_mapping:
                    for mapped_category in direct_lvis_mapping[category]:
                        if mapped_category in lvis_annotations:
                            candidate_uids.extend(lvis_annotations[mapped_category])
                            logger.info(f"Found matches in category: {mapped_category}")
                elif category in lvis_annotations:
                    candidate_uids.extend(lvis_annotations[category])
                    logger.info(f"Found matches in category: {category}")
            
            if not candidate_uids:
                logger.warning(f"No category matches found for {object_name}")
                return []
            
            logger.info(f"Found {len(candidate_uids)} initial candidates")
            
            # Get annotations for all candidates
            annotations = objaverse.load_annotations(candidate_uids)
            
            # Filter and analyze candidates
            analyzed_candidates = []
            for uid in candidate_uids:
                try:
                    # Load object metadata and mesh
                    obj_data = self._load_and_analyze_object(uid, annotations.get(uid, {}))
                    if obj_data:
                        # Score the match based on properties
                        match_score = self.category_manager.score_object_match(
                            obj_data, 
                            object_description.get('properties', {})
                        )
                        analyzed_candidates.append((uid, obj_data, match_score))
                except Exception as e:
                    logger.warning(f"Error processing candidate {uid}: {e}")
            
            if not analyzed_candidates:
                logger.warning("No valid candidates found after analysis")
                return []
                
            # Sort by match score
            analyzed_candidates.sort(key=lambda x: x[2], reverse=True)
            logger.info(f"Found {len(analyzed_candidates)} valid candidates after analysis")
            
            # Take top candidates for preview
            top_candidates = analyzed_candidates[:20]
            
            # Download and prepare preview objects
            preview_objects = []
            for uid, obj_data, score in top_candidates:
                try:
                    # Download the object
                    downloaded_paths = objaverse.load_objects([uid])
                    
                    if not downloaded_paths or uid not in downloaded_paths:
                        continue
                        
                    mesh_path = downloaded_paths[uid]
                    
                    # Copy to temp directory
                    temp_path = os.path.join(temp_dir, f"{uid}.glb")
                    shutil.copy2(mesh_path, temp_path)
                    
                    # Get thumbnail
                    if 'thumbnails' in obj_data and 'images' in obj_data['thumbnails']:
                        thumbnail_url = obj_data['thumbnails']['images'][0]['url']
                        try:
                            response = requests.get(thumbnail_url)
                            if response.status_code == 200:
                                thumb_path = os.path.join(temp_dir, f"{uid}_thumb.jpg")
                                with open(thumb_path, 'wb') as f:
                                    f.write(response.content)
                                obj_data['local_thumbnail'] = thumb_path
                        except Exception as e:
                            logger.warning(f"Failed to download thumbnail for {uid}: {e}")
                    
                    # Add match score to metadata
                    obj_data['match_score'] = score
                    preview_objects.append((uid, obj_data, temp_path))
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare preview for {uid}: {e}")
                    continue
            
            if preview_objects:
                logger.info(f"Prepared {len(preview_objects)} objects for preview")
                return preview_objects
            else:
                logger.warning("No preview objects could be prepared")
                return []
                
        except Exception as e:
            logger.error(f"Error in find_matches: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
            return []

    def _check_available_formats(self, uid: str) -> Dict[str, str]:
        """Check which formats are available for a given object."""
        try:
            # Get object metadata using standard Objaverse
            metadata = objaverse.get_metadata([uid])[uid]
            if not metadata:
                logger.warning(f"No metadata found for object {uid}")
                return {}
                
            available_formats = {}
            
            # Check for mesh formats
            if 'mesh' in metadata:
                mesh_url = metadata['mesh']
                if mesh_url and isinstance(mesh_url, str):
                    format_type = mesh_url.split('.')[-1].lower()
                    if format_type in ['stl', 'obj', 'glb', 'gltf']:
                        available_formats[format_type] = mesh_url
                        
            # Check for thumbnail
            if 'thumbnails' in metadata and metadata['thumbnails']:
                if isinstance(metadata['thumbnails'], list) and metadata['thumbnails']:
                    available_formats['thumbnail'] = metadata['thumbnails'][0]
                elif isinstance(metadata['thumbnails'], str):
                    available_formats['thumbnail'] = metadata['thumbnails']
                    
            logger.info(f"Available formats for {uid}: {list(available_formats.keys())}")
            return available_formats
            
        except Exception as e:
            logger.error(f"Error checking formats for {uid}: {e}")
            return {}

    def _download_file(self, url: str, save_path: str) -> bool:
        """Download a file from a URL.
        
        Args:
            url: URL to download from
            save_path: Path to save the file to
            
        Returns:
            bool: Whether the download was successful
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Check if response is empty
            content_length = int(response.headers.get('content-length', 0))
            if content_length < 100:  # Less than 100 bytes is suspicious
                logger.warning(f"File at {url} is too small ({content_length} bytes)")
                return False
                
            # Save the file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    def process_objects(self, objects_dict: Dict[str, Dict[str, str]], download_dir: str) -> Dict[str, Dict[str, Any]]:
        """Process and download objects from Objaverse.
        
        Args:
            objects_dict: Dictionary of objects with their descriptions and categories
            download_dir: Directory to save downloaded files
            
        Returns:
            Dictionary containing download results for each object
        """
        results = {}
        
        for obj_name, obj_info in objects_dict.items():
            logger.info(f"\nProcessing {obj_name}...")
            
            # Find matches using LVIS categories and preview selection
            matches = self.find_matches(obj_name, obj_info)
            
            if not matches:
                logger.warning(f"No matches found for {obj_name}")
                results[obj_name] = {
                    "success": False,
                    "file_path": None,
                    "error": "No matches found"
                }
                continue
            
            # Show preview and get user selection
            viewer = ObjectViewer(matches, obj_name)
            selected_match = viewer.run()
            
            if not selected_match:
                logger.warning(f"No selection made for {obj_name}")
                results[obj_name] = {
                    "success": False,
                    "file_path": None,
                    "error": "No selection made"
                }
                continue
            
            # Process selected match
            uid, obj_data, temp_path = selected_match
            
            # Create object directory
            obj_dir = os.path.join(download_dir, f"{obj_name}_stl")
            os.makedirs(obj_dir, exist_ok=True)
            
            # Move files to final location
            final_mesh_path = os.path.join(obj_dir, "model.stl")
            shutil.copy2(temp_path, final_mesh_path)
            
            # Copy thumbnail if available
            thumbnail_path = None
            if 'local_thumbnail' in obj_data and os.path.exists(obj_data['local_thumbnail']):
                thumbnail_path = os.path.join(obj_dir, "thumbnail.jpg")
                shutil.copy2(obj_data['local_thumbnail'], thumbnail_path)
            
            results[obj_name] = {
                "success": True,
                "file_path": final_mesh_path,
                "thumbnail_path": thumbnail_path,
                "uid": uid,
                "category": obj_data.get('category', '')
            }
            
            logger.info(f"Successfully processed {obj_name}")
            
        return results 

    def _load_and_analyze_object(self, uid: str, annotation: Dict) -> Optional[Dict]:
        """Load an object and analyze its properties."""
        try:
            # Start with the annotation data
            obj_data = annotation.copy()
            
            # Check for required files
            if not ('mesh' in obj_data and 
                   'thumbnails' in obj_data and 
                   'images' in obj_data['thumbnails'] and 
                   obj_data['thumbnails']['images']):
                return None
            
            # Download and load mesh for analysis
            downloaded = objaverse.load_objects([uid])
            if not downloaded or uid not in downloaded:
                return None
                
            mesh_path = downloaded[uid]
            mesh = trimesh.load(mesh_path)
            
            if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
                return None
                
            # Analyze properties
            properties = self.category_manager.analyze_object_properties(mesh)
            
            # Add basic mesh properties
            if isinstance(mesh, trimesh.Trimesh):
                properties.update({
                    'vertex_count': len(mesh.vertices),
                    'face_count': len(mesh.faces),
                    'volume': mesh.volume if hasattr(mesh, 'volume') else 0,
                    'bounding_box': mesh.bounding_box.extents.tolist()
                })
            
            # Combine everything
            obj_data.update(properties)
            obj_data['mesh_path'] = mesh_path
            
            return obj_data
            
        except Exception as e:
            logger.warning(f"Error loading and analyzing object {uid}: {e}")
            return None 