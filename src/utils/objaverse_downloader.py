import os
from typing import List, Dict, Any, Set, Tuple
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
        # Info label
        self.info_label = ttk.Label(self.info_frame, text="", wraplength=1000)
        self.info_label.pack(fill=tk.X)
        
        # Navigation buttons
        ttk.Button(self.button_frame, text="Previous", command=self.prev_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Next", command=self.next_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Select", command=self.select_current).pack(side=tk.RIGHT, padx=5)
        ttk.Button(self.button_frame, text="Skip", command=self.skip_current).pack(side=tk.RIGHT, padx=5)
    
    def load_current_object(self):
        uid, obj_data, temp_path = self.matches[self.current_index]
        
        # Update info
        info_text = f"Match {self.current_index + 1}/{len(self.matches)}\n"
        info_text += f"Name: {obj_data.get('name', 'No name')}\n"
        info_text += f"Category: {obj_data.get('category', 'Unknown')}\n"
        info_text += f"Description: {obj_data.get('description', 'No description')}"
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
        
        # Download and process each object
        for uid in uids[:10]:  # Limit to top 10 per category
            obj_data = annotations.get(uid)
            if not obj_data:
                continue
                
            # Check for thumbnail images and STL file
            has_images = (
                'thumbnails' in obj_data 
                and 'images' in obj_data['thumbnails'] 
                and len(obj_data['thumbnails']['images']) > 0
            )
            
            if has_images:
                try:
                    # Download the object
                    downloaded_paths = objaverse.load_objects(
                        uids=[uid],
                        download_processes=1
                    )
                    
                    if downloaded_paths and uid in downloaded_paths:
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
                        
                        obj_data['category'] = category
                        obj_data['thumbnail_url'] = thumbnail_url
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
            logger.info(f"Category: {object_description.get('category', 'Unknown')}")
            
            # Create temporary directory for preview objects
            temp_dir = tempfile.mkdtemp()
            
            # Get LVIS annotations
            lvis_annotations = self.get_lvis_annotations()
            
            all_preview_objects = []
            
            # First try exact category match if provided
            if object_description.get('category'):
                category = object_description['category']
                if category in lvis_annotations:
                    logger.info(f"Found exact LVIS category match: {category}")
                    category_uids = lvis_annotations[category]
                    if category_uids:
                        preview_objects = self.download_preview_objects(category, category_uids, temp_dir)
                        all_preview_objects.extend(preview_objects)
            
            # If no matches found with exact category or no category provided, fall back to name-based search
            if not all_preview_objects:
                logger.info("No matches found with exact category, falling back to name search...")
                search_name = self.strip_number_suffix(object_name)
                
                # Prepare search terms
                search_terms = search_name.lower().split()
                for term in search_terms.copy():
                    if term.endswith('s'):
                        search_terms.append(term[:-1])
                    else:
                        search_terms.append(term + 's')
                
                # Find matching categories
                matching_categories = []
                for category in lvis_annotations.keys():
                    category_lower = category.lower().replace('_', ' ')
                    if any(term in category_lower for term in search_terms):
                        matching_categories.append(category)
                
                # Process matching categories
                for category in matching_categories:
                    if len(all_preview_objects) >= 10:
                        break
                        
                    category_uids = lvis_annotations[category]
                    if category_uids:
                        preview_objects = self.download_preview_objects(
                            category, 
                            category_uids, 
                            temp_dir
                        )
                        all_preview_objects.extend(preview_objects)
            
            if not all_preview_objects:
                logger.warning(f"No matching objects found for {object_name}")
                shutil.rmtree(temp_dir)
                return []
            
            return all_preview_objects[:10]
            
        except Exception as e:
            logger.error(f"Error finding matches for {object_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
            return []

    def process_objects(self, objects_dict: Dict[str, Dict[str, Any]], download_dir: str = "downloaded_objects") -> Dict[str, Any]:
        """Process objects with user selection"""
        results = {}
        
        try:
            logger.info(f"\nProcessing {len(objects_dict)} objects...")
            logger.info(f"Download directory: {download_dir}")
            
            for obj_name, obj_desc in objects_dict.items():
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing object: {obj_name}")
                
                # Find and download preview matches
                matches = self.find_matches(obj_name, obj_desc)
                
                if not matches:
                    logger.warning(f"No matches found for {obj_name}, skipping")
                    results[obj_name] = {
                        "name": obj_name,
                        "description": obj_desc.get("description", ""),
                        "category": obj_desc.get("category", ""),
                        "file_path": None,
                        "image_path": None
                    }
                    continue
                
                # Show GUI for selection
                viewer = ObjectViewer(matches, obj_name)
                selected_match = viewer.run()
                
                # Clean up temporary files
                temp_dir = os.path.dirname(matches[0][2])
                
                if not selected_match:
                    logger.warning(f"No selection made for {obj_name}, skipping")
                    shutil.rmtree(temp_dir)
                    results[obj_name] = {
                        "name": obj_name,
                        "description": obj_desc.get("description", ""),
                        "category": obj_desc.get("category", ""),
                        "file_path": None,
                        "image_path": None
                    }
                    continue
                
                uid, obj_data, temp_path = selected_match
                
                # Create object directory
                object_dir = os.path.join(download_dir, f"{obj_name}_stl")
                os.makedirs(object_dir, exist_ok=True)
                
                # Move 3D model to final location
                model_path = os.path.join(object_dir, "model.stl")
                shutil.copy2(temp_path, model_path)
                
                # Move thumbnail if it exists
                image_path = None
                if 'local_thumbnail' in obj_data and os.path.exists(obj_data['local_thumbnail']):
                    image_path = os.path.join(object_dir, "thumbnail.jpg")
                    shutil.copy2(obj_data['local_thumbnail'], image_path)
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)
                
                results[obj_name] = {
                    "name": obj_name,
                    "description": obj_desc.get("description", ""),
                    "category": obj_desc.get("category", ""),
                    "file_path": model_path,
                    "image_path": image_path
                }
                logger.info(f"Successfully processed {obj_name}")
            
            logger.info("\nProcessing completed!")
            logger.info(f"Successfully downloaded: {sum(1 for r in results.values() if r['file_path'] is not None)}/{len(objects_dict)} objects")
            return results
            
        except Exception as e:
            logger.error(f"Error processing objects: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                obj_name: {
                    "name": obj_name,
                    "description": obj_desc.get("description", ""),
                    "category": obj_desc.get("category", ""),
                    "file_path": None,
                    "image_path": None
                } for obj_name, obj_desc in objects_dict.items()
            } 