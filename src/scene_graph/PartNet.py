import os
import sys
import random
import pickle
import xml.etree.ElementTree as ET
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from src.models.scene_graph import SceneGraph, SceneObject, SpatialRelation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PartNetError(Exception):
    """Base exception for PartNet-related errors."""
    pass

class CategoryMatchError(PartNetError):
    """Error in category matching process."""
    pass

class SceneGraphError(PartNetError):
    """Error in scene graph operations."""
    pass

class LayoutError(PartNetError):
    """Error in scene layout calculations."""
    pass

@dataclass
class PipelineStats:
    """Tracks statistics and status of the pipeline process."""
    total_objects: int = 0
    matched_objects: int = 0
    failed_matches: int = 0
    semantic_matches: int = 0
    exact_matches: int = 0
    fuzzy_matches: int = 0
    removed_relationships: int = 0
    layout_conflicts: int = 0

@dataclass
class CategoryMatch:
    """Stores category matching information."""
    category: str
    similarity_score: float
    description: str = None
    match_type: str = None  # 'exact', 'semantic', or 'fuzzy'

@dataclass
class PartNetMatch:
    """Stores matching information for a scene object."""
    scene_object: SceneObject
    urdf_path: str
    category: str
    object_id: str = None
    joint_data: List[Tuple] = None
    hierarchy_data: Dict = None
    spatial_data: Dict = None
    bbox_data: Dict = None
    similarity_score: float = None
    match_description: str = None

class PartNetManager:
    """Manages PartNet object matching and scene graph updates."""
    
    VALID_CATEGORIES = {
        "Bottle": 57, "Box": 28, "Bucket": 36, "Camera": 37, "Cart": 61,
        "Chair": 81, "Clock": 31, "CoffeeMachine": 54, "Dishwasher": 48,
        "Dispenser": 57, "Display": 37, "Door": 36, "Eyeglasses": 65,
        "Fan": 81, "Faucet": 84, "FoldingChair": 26, "Globe": 61,
        "Kettle": 29, "Keyboard": 37, "KitchenPot": 25, "Knife": 44,
        "Lamp": 45, "Laptop": 55, "Lighter": 28, "Microwave": 16,
        "Mouse": 14, "Oven": 30, "Pen": 48, "Phone": 18, "Pliers": 25,
        "Printer": 29, "Refrigerator": 44, "Remote": 49, "Safe": 30,
        "Scissors": 47, "Stapler": 23, "StorageFurniture": 346,
        "Suitcase": 24, "Switch": 70, "Table": 101, "Toaster": 25,
        "Toilet": 69, "TrashCan": 70, "USB": 51, "WashingMachine": 17,
        "Window": 58
    }

    def __init__(self, partnet_root: str):
        """Initialize with path to PartNet dataset."""
        self.partnet_root = Path(partnet_root)
        self.dataset_dir = self.partnet_root / "dataset"  # Dataset is directly in the root
        self.stats = PipelineStats()
        
        # Add category mapping for common aliases
        self.category_mapping = {
            "Monitor": "Display",
            "Screen": "Display",
            "ComputerMonitor": "Display",
            "LCD": "Display",
            "LED": "Display"
        }
        
        logger.info(f"Initializing PartNetManager with root: {partnet_root}")
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._cache_valid_objects()
            self._load_or_create_object_types()
            self._initialize_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize PartNetManager: {str(e)}")
            raise PartNetError(f"Initialization failed: {str(e)}")
        
    def _load_or_create_object_types(self):
        """Load or create object type mappings."""
        obj_types_path = self.partnet_root / "obj_types.json"
        if obj_types_path.exists():
            with open(obj_types_path, 'r') as f:
                self.obj_types = json.load(f)
        else:
            self.obj_types = self._track_obj_types()
            
    def _track_obj_types(self) -> Dict[str, List[str]]:
        """Track object types and their IDs."""
        obj_types = defaultdict(list)
        obj_ids = self._get_obj_ids()
        
        for obj_id in obj_ids:
            obj_input_dir = os.path.join(self.dataset_dir, obj_id)
            meta_path = os.path.join(obj_input_dir, "meta.json")
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                obj_type = meta.get("model_cat")
                if obj_type:
                    obj_types[obj_type].append(obj_id)
            except Exception as e:
                print(f"Error processing {meta_path}: {e}")
                
        # Save for future use
        with open(self.partnet_root / "obj_types.json", 'w') as f:
            json.dump(obj_types, f)
            
        return obj_types

    def _get_obj_ids(self) -> List[str]:
        """Get all object IDs from dataset."""
        return [d for d in os.listdir(self.dataset_dir) 
                if os.path.isdir(os.path.join(self.dataset_dir, d))]

    def _cache_valid_objects(self):
        """Cache all valid object URDFs and metadata by category."""
        self.object_cache: Dict[str, List[Tuple[str, str]]] = {}  # category -> [(obj_id, urdf_path)]
        
        logger.info("Caching PartNet objects...")
        
        # Iterate through all object directories
        for obj_dir in self.dataset_dir.iterdir():
            if not obj_dir.is_dir():
                continue
                
            obj_id = obj_dir.name
            urdf_path = obj_dir / "mobility.urdf"
            meta_path = obj_dir / "meta.json"
            
            # Check if required files exist
            if not (urdf_path.exists() and meta_path.exists()):
                logger.warning(f"Missing required files for object {obj_id}")
                continue
            
            try:
                # Load category from meta.json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                category = meta.get("model_cat")
                
                if not category:
                    logger.warning(f"No category found for object {obj_id}")
                    continue
                
                # Store object information
                if category not in self.object_cache:
                    self.object_cache[category] = []
                self.object_cache[category].append((obj_id, str(urdf_path)))
                
            except Exception as e:
                logger.error(f"Error processing object {obj_id}: {str(e)}")
                continue

    def _initialize_embeddings(self):
        """Initialize category embeddings cache."""
        cache_path = self.partnet_root / "category_embeddings.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.category_embeddings = pickle.load(f)
        else:
            print("Generating category embeddings...")
            # Generate embeddings for all categories and their variations
            categories = list(self.VALID_CATEGORIES.keys())
            category_texts = []
            for cat in categories:
                # Add variations and descriptions
                category_texts.extend([
                    cat,
                    f"a {cat.lower()}",
                    f"the {cat.lower()}",
                    f"{cat.lower()} object",
                ])
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(category_texts)
            
            # Store with mapping
            self.category_embeddings = {
                'texts': category_texts,
                'embeddings': embeddings,
                'categories': categories
            }
            
            # Cache for future use
            with open(cache_path, 'wb') as f:
                pickle.dump(self.category_embeddings, f)

    def _semantic_category_match(self, query: str, threshold: float = 0.6) -> Optional[CategoryMatch]:
        """Enhanced semantic matching with better error handling and logging."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(
                self.category_embeddings['embeddings'], 
                query_embedding
            )
            
            # Find best matches
            top_k = 3  # Get top 3 matches for logging
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            logger.debug(f"Top {top_k} matches for '{query}':")
            for idx, score in zip(top_indices, top_scores):
                logger.debug(f"  {self.category_embeddings['texts'][idx]}: {score:.3f}")
            
            if top_scores[0] < threshold:
                logger.info(f"No matches above threshold ({threshold}) for '{query}'")
                return None
                
            # Map back to original category
            matched_text = self.category_embeddings['texts'][top_indices[0]]
            for cat in self.category_embeddings['categories']:
                if cat.lower() in matched_text.lower():
                    return CategoryMatch(
                        category=cat,
                        similarity_score=float(top_scores[0]),
                        description=f"Matched '{query}' to '{cat}' (score: {top_scores[0]:.3f})",
                        match_type='semantic'
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic matching failed for query '{query}': {str(e)}")
            raise CategoryMatchError(f"Semantic matching failed: {str(e)}")

    def _extract_joint_data(self, urdf_path: str, include_semantic_names: bool = True) -> List[Tuple]:
        """Extract joint data from URDF file with better error handling."""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            joints_data = []
            
            for joint in root.findall("joint"):
                try:
                    # Required attributes
                    if not all(key in joint.attrib for key in ["type", "name"]):
                        logger.warning(f"Joint missing required attributes in {urdf_path}")
                        continue
                        
                    joint_type = joint.attrib["type"]
                    joint_name = joint.attrib["name"]
                    
                    # Required child elements
                    parent_elem = joint.find("parent")
                    child_elem = joint.find("child")
                    if parent_elem is None or child_elem is None:
                        logger.warning(f"Joint missing parent/child elements in {urdf_path}")
                        continue
                        
                    if "link" not in parent_elem.attrib or "link" not in child_elem.attrib:
                        logger.warning(f"Joint parent/child missing link attribute in {urdf_path}")
                        continue
                        
                    parent_link = parent_elem.attrib["link"]
                    child_link = child_elem.attrib["link"]
                    
                    if include_semantic_names:
                        obj_id = Path(urdf_path).parent.name
                        semantic_file = Path(urdf_path).parent / "semantics.txt"
                        if semantic_file.exists():
                            try:
                                with open(semantic_file, 'r') as f:
                                    semantics = {}
                                    for line in f:
                                        if line.strip():
                                            parts = line.strip().split()
                                            if len(parts) == 2:  # Only process valid lines
                                                semantics[parts[0]] = parts[1]
                                            
                                p_link = semantics.get(parent_link, parent_link)
                                c_link = semantics.get(child_link, child_link)
                                semantic_joint = f"{p_link}_to_{c_link}"
                                joints_data.append((joint_type, joint_name, parent_link, child_link, semantic_joint))
                            except Exception as e:
                                logger.warning(f"Error processing semantics for {urdf_path}: {e}")
                                joints_data.append((joint_type, joint_name, parent_link, child_link))
                        else:
                            joints_data.append((joint_type, joint_name, parent_link, child_link))
                    else:
                        joints_data.append((joint_type, joint_name, parent_link, child_link))
                        
                except Exception as e:
                    logger.warning(f"Error processing joint in {urdf_path}: {e}")
                    continue
                    
            return joints_data
            
        except Exception as e:
            logger.error(f"Error processing URDF {urdf_path}: {e}")
            return []

    def _load_bounding_box(self, obj_id: str) -> Optional[Dict]:
        """Load bounding box information from JSON file."""
        bbox_path = os.path.join(self.dataset_dir, obj_id, "bounding_box.json")
        try:
            with open(bbox_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading bounding box for {obj_id}: {e}")
            return None

    def _calculate_object_dimensions(self, bbox_data: Dict) -> Tuple[float, float, float]:
        """Calculate object dimensions from bounding box data."""
        if not bbox_data:
            return (1.0, 1.0, 1.0)  # Default dimensions if no data
            
        min_coords = bbox_data["min"]
        max_coords = bbox_data["max"]
        
        width = abs(max_coords[0] - min_coords[0])
        length = abs(max_coords[1] - min_coords[1])
        height = abs(max_coords[2] - min_coords[2])
        
        return (width, length, height)

    def find_matching_objects(self, scene_graph: SceneGraph) -> Tuple[List[PartNetMatch], SceneGraph]:
        """Find matching PartNet objects and update scene graph."""
        matches: List[PartNetMatch] = []
        unmatched_objects: List[SceneObject] = []
        
        print("\nMatching scene objects to PartNet categories...")
        
        # Track object type counts for unique naming
        type_counts = defaultdict(int)
        
        for obj in scene_graph.objects:
            match = self._find_match_for_object(obj)
            if match:
                # Update object name to ensure uniqueness
                base_name = match.scene_object.name
                type_counts[base_name] += 1
                if type_counts[base_name] > 1:
                    match.scene_object.name = f"{base_name}_{type_counts[base_name]}"
                
                # Extract joint data
                joint_data = self._extract_joint_data(match.urdf_path)
                match.joint_data = joint_data
                
                # Load bounding box data
                bbox_data = self._load_bounding_box(match.object_id)
                match.bbox_data = bbox_data
                
                # Store hierarchical data
                match.hierarchy_data = {
                    "level": obj.hierarchy.level,
                    "is_anchor": obj.hierarchy.is_anchor,
                    "parent": obj.hierarchy.parent,
                    "children": obj.hierarchy.children,
                    "dimensions": self._calculate_object_dimensions(bbox_data)
                }
                
                # Update spatial relationships with new object name if it was changed
                if match.scene_object.name != base_name:
                    updated_relations = []
                    for rel in scene_graph.relationships:
                        if rel.source == base_name:
                            rel.source = match.scene_object.name
                        if rel.target == base_name:
                            rel.target = match.scene_object.name
                        updated_relations.append(rel)
                    scene_graph.relationships = updated_relations
                
                # Store spatial relationships
                match.spatial_data = {
                    "relations": [rel for rel in scene_graph.relationships 
                                if rel.source == match.scene_object.name or 
                                rel.target == match.scene_object.name]
                }
                
                matches.append(match)
                print(f"Found match for {match.scene_object.name}: {match.category} ({match.urdf_path})")
                print(f"  Match confidence: {match.similarity_score:.3f}")
                print(f"  Match details: {match.match_description}")
                if bbox_data:
                    dims = self._calculate_object_dimensions(bbox_data)
                    print(f"  Dimensions (w,l,h): {[f'{x:.3f}' for x in dims]}")
            else:
                unmatched_objects.append(obj)
                print(f"No match found for {obj.name} ({obj.category})")
        
        # Create updated scene graph without unmatched objects
        updated_graph = self._create_updated_scene_graph(scene_graph, unmatched_objects)
        
        # Sort matches by hierarchy level for proper placement
        matches.sort(key=lambda x: x.hierarchy_data["level"])
        
        return matches, updated_graph

    def _find_match_for_object(self, obj: SceneObject) -> Optional[PartNetMatch]:
        """Enhanced object matching with better logging and stats tracking."""
        logger.info(f"Finding match for object: {obj.name} ({obj.category})")
        
        try:
            # Map category if it has an alias
            mapped_category = self.category_mapping.get(obj.category, obj.category)
            logger.info(f"Category mapping: {obj.category} -> {mapped_category}")
            
            # Try exact category match first
            category = self._clean_category_name(mapped_category)
            if category in self.object_cache:
                self.stats.exact_matches += 1
                logger.info(f"Found exact match: {category}")
                
                # For displays, try to get a unique one if possible
                if category == "Display":
                    used_displays = getattr(self, '_used_displays', set())
                    available_displays = [(obj_id, path) for obj_id, path in self.object_cache[category] 
                                       if obj_id not in used_displays]
                    
                    if available_displays:
                        obj_id, urdf_path = random.choice(available_displays)
                        if not hasattr(self, '_used_displays'):
                            self._used_displays = set()
                        self._used_displays.add(obj_id)
                    else:
                        # If all displays are used, just pick a random one
                        obj_id, urdf_path = random.choice(self.object_cache[category])
                else:
                    obj_id, urdf_path = random.choice(self.object_cache[category])
                
                return PartNetMatch(
                    obj, urdf_path, category, obj_id, 
                    similarity_score=1.0,
                    match_description="Exact category match"
                )
            
            # Try semantic matching
            query = f"{obj.category} {obj.description if obj.description else ''}"
            category_match = self._semantic_category_match(query)
            
            if category_match and category_match.category in self.object_cache:
                self.stats.semantic_matches += 1
                logger.info(f"Found semantic match: {category_match.category} ({category_match.similarity_score:.3f})")
                obj_id, urdf_path = random.choice(self.object_cache[category_match.category])
                return PartNetMatch(
                    obj, urdf_path, category_match.category, obj_id,
                    similarity_score=category_match.similarity_score,
                    match_description=category_match.description
                )
            
            # Fallback to fuzzy matching
            matched_category = self._fuzzy_match_category(obj.category)
            if matched_category and matched_category in self.object_cache:
                self.stats.fuzzy_matches += 1
                logger.info(f"Found fuzzy match: {matched_category}")
                obj_id, urdf_path = random.choice(self.object_cache[matched_category])
                return PartNetMatch(
                    obj, urdf_path, matched_category, obj_id,
                    similarity_score=0.5,
                    match_description=f"Fuzzy matched '{obj.category}' to '{matched_category}'"
                )
            
            self.stats.failed_matches += 1
            logger.warning(f"No match found for {obj.name} ({obj.category})")
            return None
            
        except Exception as e:
            logger.error(f"Error matching object {obj.name}: {str(e)}")
            self.stats.failed_matches += 1
            raise CategoryMatchError(f"Object matching failed: {str(e)}")

    def _clean_category_name(self, category: str) -> str:
        """Clean category name by removing instance count and whitespace."""
        import re
        clean = re.sub(r'\s*\(\d+\)\s*', '', category.strip())
        return clean

    def _fuzzy_match_category(self, category: str) -> Optional[str]:
        """Try to find a matching category using fuzzy matching."""
        category_lower = category.lower()
        
        # Simple substring matching
        for valid_category in self.VALID_CATEGORIES:
            if (category_lower in valid_category.lower() or 
                valid_category.lower() in category_lower):
                return valid_category
                
        # Try matching against obj_types
        for obj_type in self.obj_types:
            if (category_lower in obj_type.lower() or 
                obj_type.lower() in category_lower):
                return obj_type
        
        return None

    def _create_updated_scene_graph(self, 
                                  original_graph: SceneGraph, 
                                  unmatched_objects: List[SceneObject]) -> SceneGraph:
        """Create new scene graph excluding unmatched objects and their relationships."""
        unmatched_names = {obj.name for obj in unmatched_objects}
        
        # Filter objects
        new_objects = []
        for obj in original_graph.objects:
            if obj.name not in unmatched_names:
                # Update hierarchy to remove unmatched children and parents
                if obj.hierarchy:
                    new_children = [child for child in obj.hierarchy.children 
                                  if child not in unmatched_names]
                    new_parent = obj.hierarchy.parent if (obj.hierarchy.parent and 
                                                        obj.hierarchy.parent not in unmatched_names) else None
                    obj.hierarchy.children = new_children
                    obj.hierarchy.parent = new_parent
                new_objects.append(obj)
        
        # Filter relationships
        new_relationships = []
        for rel in original_graph.relationships:
            if (rel.source not in unmatched_names and rel.target not in unmatched_names):
                new_relationships.append(rel)
        
        # Create new graph
        return SceneGraph(
            objects=new_objects,
            relationships=new_relationships
        )

    def validate_scene_graph(self, graph: SceneGraph) -> bool:
        """Validate scene graph consistency after updates."""
        object_names = {obj.name for obj in graph.objects}
        
        # Check relationship consistency
        for rel in graph.relationships:
            if rel.source not in object_names or rel.target not in object_names:
                return False
        
        # Check hierarchy consistency
        for obj in graph.objects:
            if obj.hierarchy:
                # Check parent exists if specified
                if obj.hierarchy.parent and obj.hierarchy.parent not in object_names:
                    return False
                # Check all children exist
                for child in obj.hierarchy.children:
                    if child not in object_names:
                        return False
                
        return True

    def plan_scene_layout(self, matches: List[PartNetMatch]) -> Dict[str, Dict]:
        """Enhanced scene layout planning with conflict detection."""
        layout = {}
        collision_checks = defaultdict(list)
        
        logger.info("Planning scene layout")
        
        try:
            # Process objects level by level
            for match in matches:
                obj_name = match.scene_object.name
                bbox_data = match.bbox_data
                dims = self._calculate_object_dimensions(bbox_data) if bbox_data else (1.0, 1.0, 1.0)
                
                # Get parent information
                parent_name = match.hierarchy_data.get("parent")
                parent_info = layout.get(parent_name) if parent_name else None
                
                # Calculate initial position
                position = self._calculate_object_position(
                    match, dims, parent_info, layout
                )
                
                # Check for collisions
                while self._check_collisions(position, dims, layout):
                    self.stats.layout_conflicts += 1
                    # Adjust position to resolve collision
                    position = self._adjust_position(position, dims, layout)
                
                # Store layout information
                layout[obj_name] = {
                    "dimensions": dims,
                    "position": position,
                    "bbox_data": bbox_data,
                    "urdf_path": match.urdf_path,
                    "collisions_resolved": self.stats.layout_conflicts
                }
                
                logger.info(f"Placed {obj_name} at position {[f'{x:.3f}' for x in position]}")
                
            return layout
            
        except Exception as e:
            logger.error(f"Layout planning failed: {str(e)}")
            raise LayoutError(f"Layout planning failed: {str(e)}")

    def _calculate_object_position(
        self, 
        match: PartNetMatch, 
        dims: Tuple[float, float, float],
        parent_info: Optional[Dict],
        current_layout: Dict
    ) -> Tuple[float, float, float]:
        """Calculate object position with complex spatial relationships."""
        if not parent_info:  # Root object or no parent
            return (0.0, 0.0, dims[2]/2)  # Place on ground
            
        parent_pos = parent_info["position"]
        parent_dims = parent_info["dimensions"]
        
        # Find all relevant spatial relations
        spatial_rels = [rel for rel in match.spatial_data["relations"] 
                       if rel.target in current_layout or rel.source in current_layout]
        
        # Initialize with default position on parent
        x = parent_pos[0]
        y = parent_pos[1]
        z = parent_pos[2] + parent_dims[2]/2 + dims[2]/2
        
        # Group related objects for complex relationships
        related_objects = self._get_related_objects(match.scene_object.name, current_layout)
        
        for rel in spatial_rels:
            target_info = current_layout.get(rel.target)
            if not target_info:
                continue
                
            if rel.relation_type == "on":
                # Place on surface with proper alignment
                z = target_info["position"][2] + target_info["dimensions"][2]/2 + dims[2]/2
                # Center on the surface by default
                x = target_info["position"][0]
                y = target_info["position"][1]
                
            elif rel.relation_type == "inside":
                # Place inside with clearance
                clearance = 0.02  # 2cm clearance
                z = target_info["position"][2]  # Place at container bottom
                # Ensure object fits inside container
                if any(dim_obj > dim_container - 2*clearance 
                      for dim_obj, dim_container in zip(dims, target_info["dimensions"])):
                    logger.warning(f"Object too large to fit inside {rel.target}")
                    continue
                    
            elif rel.relation_type == "between":
                # Find the two reference objects
                ref_objects = [obj for obj in related_objects 
                             if obj != match.scene_object.name][:2]
                if len(ref_objects) == 2:
                    pos1 = current_layout[ref_objects[0]]["position"]
                    pos2 = current_layout[ref_objects[1]]["position"]
                    # Calculate midpoint
                    x = (pos1[0] + pos2[0]) / 2
                    y = (pos1[1] + pos2[1]) / 2
                    
            elif rel.relation_type == "aligned_with":
                # Align along the specified axis
                if rel.axis == "horizontal":
                    z = target_info["position"][2]
                elif rel.axis == "vertical":
                    x = target_info["position"][0]
                    y = target_info["position"][1]
                    
            elif rel.relation_type in ["left_of", "right_of"]:
                offset = dims[0]/2 + target_info["dimensions"][0]/2
                if rel.relation_type == "left_of":
                    x = target_info["position"][0] - offset
                else:
                    x = target_info["position"][0] + offset
                    
            elif rel.relation_type in ["in_front_of", "behind"]:
                offset = dims[1]/2 + target_info["dimensions"][1]/2
                if rel.relation_type == "in_front_of":
                    y = target_info["position"][1] + offset
                else:
                    y = target_info["position"][1] - offset
                    
            # Apply any specific distance constraints
            if hasattr(rel, "distance") and rel.distance is not None:
                self._apply_distance_constraint(
                    (x, y, z), rel.distance, rel.relation_type,
                    target_info["position"]
                )
        
        return (x, y, z)

    def _get_related_objects(self, obj_name: str, layout: Dict) -> List[str]:
        """Get all objects related to the given object."""
        related = set()
        for rel in self.scene_graph.relationships:
            if rel.source == obj_name:
                related.add(rel.target)
            elif rel.target == obj_name:
                related.add(rel.source)
        return [obj for obj in related if obj in layout]

    def _apply_distance_constraint(
        self,
        current_pos: Tuple[float, float, float],
        distance: float,
        relation_type: str,
        reference_pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Apply distance constraint while maintaining relationship type."""
        x, y, z = current_pos
        ref_x, ref_y, ref_z = reference_pos
        
        if relation_type in ["left_of", "right_of"]:
            # Adjust x while maintaining y,z
            direction = 1 if relation_type == "right_of" else -1
            x = ref_x + direction * distance
        elif relation_type in ["in_front_of", "behind"]:
            # Adjust y while maintaining x,z
            direction = 1 if relation_type == "in_front_of" else -1
            y = ref_y + direction * distance
        elif relation_type in ["above", "below"]:
            # Adjust z while maintaining x,y
            direction = 1 if relation_type == "above" else -1
            z = ref_z + direction * distance
            
        return (x, y, z)

    def _check_collisions(self, position: Tuple[float, float, float], 
                         dims: Tuple[float, float, float],
                         layout: Dict[str, Dict]) -> bool:
        """Check if object collides with any existing objects."""
        for other_obj, other_info in layout.items():
            if self._boxes_intersect(
                position, dims,
                other_info["position"],
                other_info["dimensions"]
            ):
                logger.warning(f"Collision detected at position {position}")
                return True
        return False

    def _boxes_intersect(self, pos1, dims1, pos2, dims2) -> bool:
        """Check if two boxes intersect."""
        for i in range(3):
            if abs(pos1[i] - pos2[i]) > (dims1[i] + dims2[i])/2:
                return False
        return True

    def _adjust_position(self, position: Tuple[float, float, float],
                        dims: Tuple[float, float, float],
                        layout: Dict[str, Dict]) -> Tuple[float, float, float]:
        """Adjust position to resolve collisions."""
        x, y, z = position
        step = 0.1  # Adjustment step size
        
        # Try adjusting x and y while maintaining z
        new_pos = (x + step, y, z)
        if not self._check_collisions(new_pos, dims, layout):
            return new_pos
            
        new_pos = (x, y + step, z)
        if not self._check_collisions(new_pos, dims, layout):
            return new_pos
            
        # If still colliding, try increasing z
        return (x, y, z + step)

    def process_scene(self, scene_graph: SceneGraph) -> Tuple[List[PartNetMatch], SceneGraph, PipelineStats]:
        """Main pipeline for processing a scene."""
        logger.info("Starting scene processing pipeline")
        self.stats = PipelineStats()  # Reset stats
        
        try:
            # 1. Match objects
            matches, updated_graph = self.find_matching_objects(scene_graph)
            
            # 2. Validate updated graph
            if not self.validate_scene_graph(updated_graph):
                raise SceneGraphError("Scene graph validation failed after matching")
            
            # 3. Plan layout
            layout = self.plan_scene_layout(matches)
            
            # 4. Generate scene URDF
            # TODO: Implement URDF generation
            
            logger.info(f"Pipeline completed successfully. Stats: {self.stats}")
            
            # 5. Clean up by restoring original files
            logger.info("Cleaning up scaled files...")
            if not self.cleanup_scaled_files():
                logger.warning("Some files could not be restored to original state")
            
            return matches, updated_graph, self.stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            # Try to clean up even if pipeline fails
            self.cleanup_scaled_files()
            raise

    def generate_scene_urdf(self, layout: Dict[str, Dict], output_path: str):
        """Generate a combined URDF file for the entire scene."""
        # Implementation for generating combined URDF
        # This would merge individual URDFs with proper transformations
        pass

    def set_object_dimensions(self, obj_id: str, dimensions: Tuple[float, float, float], image_info: Optional[Dict] = None) -> bool:
        """Set exact dimensions for an object by modifying URDF and mesh files.
        
        Args:
            obj_id: PartNet object ID
            dimensions: (width, length, height) in meters
            image_info: Optional dict containing image paths and bounding box info
        """
        try:
            urdf_path = os.path.join(self.dataset_dir, obj_id, "mobility.urdf")
            if not os.path.exists(urdf_path):
                logger.error(f"URDF file not found for object {obj_id}")
                return False

            # Backup original files
            backup_path = urdf_path + ".backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(urdf_path, backup_path)

            # Get current dimensions
            current_dims = self._get_mesh_dimensions_for_id(obj_id)
            if not current_dims:
                logger.error(f"Could not get current dimensions for {obj_id}")
                return False

            # Calculate transform ratios
            transform = (
                dimensions[0] / current_dims[0],
                dimensions[1] / current_dims[1],
                dimensions[2] / current_dims[2]
            )

            # Update URDF
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # Update all links
            for link in root.findall(".//link"):
                # Update visual geometries
                for visual in link.findall(".//visual"):
                    self._transform_geometry(visual, transform, dimensions)

                # Update collision geometries
                for collision in link.findall(".//collision"):
                    self._transform_geometry(collision, transform, dimensions)

                # Update inertial properties
                inertial = link.find("inertial")
                if inertial is None:
                    inertial = ET.SubElement(link, "inertial")
                self._update_inertial(inertial, dimensions)

            # Update joint origins
            for joint in root.findall(".//joint"):
                origin = joint.find("origin")
                if origin is not None:
                    xyz = origin.get("xyz", "0 0 0").split()
                    new_xyz = [float(x) * t for x, t in zip(xyz, transform)]

            # Save modified URDF
            tree.write(urdf_path, xml_declaration=True, encoding='utf-8')

            # Update mesh files
            mesh_dir = os.path.join(self.dataset_dir, obj_id, "textured_objs")
            if os.path.exists(mesh_dir):
                for mesh_file in os.listdir(mesh_dir):
                    if mesh_file.endswith(".obj"):
                        self._transform_mesh(
                            os.path.join(mesh_dir, mesh_file),
                            transform,
                            dimensions
                        )

            logger.info(f"Successfully updated dimensions for object {obj_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating dimensions for object {obj_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _transform_geometry(self, geom_elem: ET.Element, transform: Tuple[float, float, float], dimensions: Tuple[float, float, float]):
        """Transform geometry element in URDF with exact dimensions."""
        geometry = geom_elem.find("geometry")
        if geometry is None:
            return

        if geometry.find("box") is not None:
            box = geometry.find("box")
            box.set("size", f"{dimensions[0]} {dimensions[1]} {dimensions[2]}")

        elif geometry.find("cylinder") is not None:
            cylinder = geometry.find("cylinder")
            # Use average of width and length for radius
            radius = (dimensions[0] + dimensions[1]) / 4
            cylinder.set("radius", str(radius))
            cylinder.set("length", str(dimensions[2]))

        elif geometry.find("sphere") is not None:
            sphere = geometry.find("sphere")
            # Use average dimension for radius
            radius = sum(dimensions) / 6
            sphere.set("radius", str(radius))

        elif geometry.find("mesh") is not None:
            mesh = geometry.find("mesh")
            mesh.set("scale", f"{transform[0]} {transform[1]} {transform[2]}")

    def _update_inertial(self, inertial_elem: ET.Element, dimensions: Tuple[float, float, float]):
        """Update inertial properties based on new dimensions."""
        # Set origin to center of mass
        origin = inertial_elem.find("origin")
        if origin is None:
            origin = ET.SubElement(inertial_elem, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")

        # Calculate mass based on volume (assuming uniform density)
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        density = 1000  # kg/m³ (default density)
        mass = volume * density

        # Set mass
        mass_elem = inertial_elem.find("mass")
        if mass_elem is None:
            mass_elem = ET.SubElement(inertial_elem, "mass")
        mass_elem.set("value", str(mass))

        # Calculate inertia tensor for a box
        width, length, height = dimensions
        ixx = (mass / 12.0) * (length * length + height * height)
        iyy = (mass / 12.0) * (width * width + height * height)
        izz = (mass / 12.0) * (width * width + length * length)

        # Set inertia values
        inertia = inertial_elem.find("inertia")
        if inertia is None:
            inertia = ET.SubElement(inertial_elem, "inertia")

        inertia.set("ixx", str(max(0.001, ixx)))
        inertia.set("iyy", str(max(0.001, iyy)))
        inertia.set("izz", str(max(0.001, izz)))
        inertia.set("ixy", "0")
        inertia.set("ixz", "0")
        inertia.set("iyz", "0")

    def _transform_mesh(self, mesh_path: str, transform: Tuple[float, float, float], dimensions: Tuple[float, float, float]):
        """Transform mesh vertices to match exact dimensions."""
        try:
            import trimesh

            # Backup original mesh if not already backed up
            backup_path = mesh_path + ".backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(mesh_path, backup_path)

            # Load mesh
            mesh = trimesh.load(mesh_path)
            if not isinstance(mesh, trimesh.Trimesh):
                logger.warning(f"Could not load {mesh_path} as a trimesh object")
                return False

            # Create transformation matrix
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = transform[0]
            scale_matrix[1, 1] = transform[1]
            scale_matrix[2, 2] = transform[2]

            # Apply transformation
            mesh.apply_transform(scale_matrix)

            # Export transformed mesh
            export_options = {}
            if mesh_path.lower().endswith('.obj'):
                export_options = {
                    'vertex_normal': True,
                    'include_texture': True,
                    'include_materials': True,
                    'resolver': trimesh.resolvers.FilePathResolver()
                }

            mesh.export(mesh_path, **export_options)
            logger.info(f"Successfully transformed mesh: {mesh_path}")
            logger.info(f"  New dimensions: {[f'{x:.3f}' for x in dimensions]}")
            return True

        except Exception as e:
            logger.error(f"Error transforming mesh {mesh_path}: {str(e)}")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, mesh_path)
            return False

    def restore_original_files(self, obj_id: str) -> bool:
        """Restore original URDF and mesh files from backups.
        
        Args:
            obj_id: PartNet object ID
            
        Returns:
            bool: True if restoration was successful
        """
        try:
            # Restore URDF file
            urdf_path = os.path.join(self.dataset_dir, obj_id, "mobility.urdf")
            backup_path = urdf_path + ".backup"
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, urdf_path)
                os.remove(backup_path)
                logger.info(f"Restored original URDF for object {obj_id}")
            
            # Restore mesh files
            mesh_dir = os.path.join(self.dataset_dir, obj_id, "textured_objs")
            if os.path.exists(mesh_dir):
                restored_count = 0
                for mesh_file in os.listdir(mesh_dir):
                    if mesh_file.endswith(".obj.backup"):
                        original_path = mesh_file[:-7]  # Remove .backup
                        backup_path = os.path.join(mesh_dir, mesh_file)
                        original_path = os.path.join(mesh_dir, original_path)
                        shutil.copy2(backup_path, original_path)
                        os.remove(backup_path)
                        restored_count += 1
                if restored_count > 0:
                    logger.info(f"Restored {restored_count} original mesh files for object {obj_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring original files for object {obj_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def cleanup_scaled_files(self) -> bool:
        """Reset all scaled files back to their original state by restoring from backups.
        This should be called after scene generation is complete.
        
        Returns:
            bool: True if cleanup was successful
        """
        try:
            cleanup_successful = True
            files_restored = 0
            
            # Iterate through all object directories
            for obj_dir in self.dataset_dir.iterdir():
                if not obj_dir.is_dir():
                    continue
                    
                obj_id = obj_dir.name
                urdf_backup = obj_dir / "mobility.urdf.backup"
                
                # Check if this object was modified (has backup files)
                if urdf_backup.exists():
                    logger.info(f"Cleaning up scaled files for object {obj_id}")
                    if not self.restore_original_files(obj_id):
                        cleanup_successful = False
                        logger.error(f"Failed to restore files for object {obj_id}")
                    else:
                        files_restored += 1
            
            if files_restored > 0:
                logger.info(f"Successfully restored {files_restored} objects to original state")
            else:
                logger.info("No scaled files found to restore")
            
            return cleanup_successful
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False