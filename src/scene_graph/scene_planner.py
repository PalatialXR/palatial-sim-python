import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
from enum import Enum
import pybullet as p
from collections import defaultdict, deque
import time
import networkx as nx
import matplotlib.pyplot as plt
import json
from openai import OpenAI
import traceback
import trimesh  # For mesh manipulation
import shutil   # For file operations
import cv2
from PIL import Image

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.models.scene_graph import (
    SpatialRelation,
    SemanticRelation,
    ObjectState,
    SPATIAL_RELATIONS
)
from src.scene_graph.PartNet import PartNetManager

class PlacementError(Exception):
    """Base exception for placement-related errors."""
    pass

class ScenePlanner:
    def __init__(self):
        self.objects: Dict[str, ObjectState] = {}
        self.placement_order: List[str] = []
        self.placement_stats = {
            "hierarchy_stats": defaultdict(int),
            "placement_failures": defaultdict(list),
            "surface_utilization": defaultdict(float),
            "stability_checks": defaultdict(bool),
            "collision_counts": defaultdict(int)
        }
        self.llm_client = OpenAI()
        
        # Placement parameters
        self.min_spacing = 0.05  # 5cm minimum spacing between objects
        self.default_grid_resolution = 0.02  # 2cm grid resolution
        
        # Map semantic names to PartNet IDs
        self.object_id_map = {
            "Table": ["26652", "45262", "45671"],  # Table IDs
            "Monitor": ["4627", "102401", "102694"],  # Monitor IDs
            "Keyboard": ["10238", "10305", "10707"],  # Keyboard IDs
            "Mouse": ["101416", "101425", "101511"]  # Mouse IDs
        }
    
    def precompute_placement(self, 
                          relations: List[SpatialRelation],
                          object_states: Optional[Dict[str, ObjectState]] = None) -> Tuple[List[str], Dict[str, ObjectState]]:
        """Precompute optimal object placement order and hierarchy."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting placement planning for {len(relations)} relations")
        logger.info(f"Initial object states: {len(object_states) if object_states else 0} objects")
        
        try:
            # First pass: Build object hierarchy
            logger.info("\n1. Building object hierarchy...")
            hierarchy = self._build_hierarchy(relations)
            self._log_hierarchy_stats(hierarchy)
            
            # Initialize object states based on hierarchy and provided states
            logger.info("\n2. Initializing object states...")
            self._initialize_object_states(hierarchy, object_states)
            
            # Optimize object selection based on bounding box compatibility
            logger.info("\n3. Optimizing object selection...")
            selection_changes = self._optimize_object_selection(hierarchy)
            
            # Optimize object scales if needed
            logger.info("\n4. Optimizing object scales...")
            scale_changes = self._optimize_object_scales(hierarchy)
            
            if selection_changes or scale_changes:
                logger.info("\n5. Reinitializing placement grids due to dimension changes...")
                self._reinitialize_placement_grids()
            
            # Calculate placement order
            logger.info("\n6. Calculating placement order...")
            self._calculate_placement_order(hierarchy)
            logger.info(f"Placement order determined: {self.placement_order}")
            
            # Optimize surface allocation and determine positions
            logger.info("\n7. Optimizing surface allocation...")
            self._optimize_surface_allocation()
            
            # Log final positions and scales
            logger.info("\n8. Final object configurations:")
            for obj_name, obj_state in self.objects.items():
                logger.info(f"  • {obj_name}:")
                if obj_state.position:
                    logger.info(f"    - Position: {[f'{x:.3f}' for x in obj_state.position]}")
                else:
                    logger.warning(f"    - Position: Not assigned")
                logger.info(f"    - Scale: {[f'{x:.3f}' for x in obj_state.bbox]}")
            
            self._log_placement_stats()
            logger.info(f"{'='*50}\n")
            
            return self.placement_order, self.objects
            
        except Exception as e:
            logger.error(f"Placement planning failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise PlacementError(f"Failed to plan placement: {str(e)}")
    
    def _log_hierarchy_stats(self, hierarchy: Dict[str, Dict[str, Set[str]]]):
        """Log statistics about the object hierarchy."""
        stats = self.placement_stats["hierarchy_stats"]
        stats["total_objects"] = len(hierarchy) - 1  # Excluding ground
        stats["root_objects"] = len(hierarchy["ground"]["children"])
        stats["leaf_objects"] = sum(1 for obj in hierarchy if not hierarchy[obj]["children"])
        stats["floating_objects"] = sum(1 for obj in hierarchy if not hierarchy[obj]["parents"] and obj != "ground")
        
        logger.info("Hierarchy Statistics:")
        logger.info(f"  Total objects: {stats['total_objects']}")
        logger.info(f"  Root objects: {stats['root_objects']}")
        logger.info(f"  Leaf objects: {stats['leaf_objects']}")
        logger.info(f"  Floating objects: {stats['floating_objects']}")
    
    def _build_hierarchy(self, relations: List[SpatialRelation]) -> Dict[str, Dict[str, Set[str]]]:
        """Build object hierarchy using graph analysis."""
        # Create directed graph for hierarchy analysis
        G = nx.DiGraph()
        G.add_node("ground")  # Add ground node
        
        # First add all relations to graph
        for rel in relations:
            G.add_edge(rel.source, rel.target, 
                      relation_type=rel.relation_type)
            
            # Add nodes to ensure they exist even without relationships
            G.add_node(rel.source)
            G.add_node(rel.target)
        
        # Initialize hierarchy structure
        hierarchy = {"ground": {"children": set(), "parents": set()}}
        for node in G.nodes():
            if node != "ground":
                hierarchy[node] = {"children": set(), "parents": set()}
        
        # Process support relationships first (on, above)
        support_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d["relation_type"] in ["on", "above"]]
        
        # Build support hierarchy
        for source, target in support_edges:
            hierarchy[target]["children"].add(source)
            hierarchy[source]["parents"].add(target)
            logger.debug(f"Added support relationship: {source} supported by {target}")
        
        # Find furniture pieces (degree analysis and name matching)
        furniture = {node for node in G.nodes() 
                    if self._is_furniture(node) or 
                    (G.in_degree(node) == 0 and G.out_degree(node) > 0)}
        
        # Connect furniture to ground
        for node in furniture:
            if not hierarchy[node]["parents"]:
                hierarchy["ground"]["children"].add(node)
                hierarchy[node]["parents"].add("ground")
                logger.debug(f"Connected furniture to ground: {node}")
        
        # Verify hierarchy is complete
        orphans = [node for node in G.nodes() 
                  if node != "ground" and not hierarchy[node]["parents"]]
        
        if orphans:
            logger.warning(f"Found orphaned objects: {orphans}")
            # Try to find best parent based on spatial relationships
            for orphan in orphans:
                best_parent = self._find_best_parent(orphan, G, hierarchy)
                if best_parent:
                    hierarchy[best_parent]["children"].add(orphan)
                    hierarchy[orphan]["parents"].add(best_parent)
                    logger.info(f"Assigned orphan {orphan} to parent {best_parent}")
        
        return hierarchy

    def _find_best_parent(self, node: str, G: nx.DiGraph, hierarchy: Dict) -> Optional[str]:
        """Find best parent for an orphaned node based on spatial relationships."""
        # Check spatial relationships
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return None
        
        # Prefer furniture pieces as parents
        furniture_neighbors = [n for n in neighbors if self._is_furniture(n)]
        if furniture_neighbors:
            return furniture_neighbors[0]
        
        # Otherwise, choose node with most children as parent
        return max(neighbors, key=lambda n: len(hierarchy[n]["children"]), default=None)

    def _get_scene_parent(self, obj_name: str) -> Optional[str]:
        """Get parent from scene graph if available."""
        if obj_name in self.objects:
            return self.objects[obj_name].parent
        return None

    def _is_furniture(self, obj_name: str) -> bool:
        """Check if object is likely to be a major furniture piece."""
        furniture_keywords = ["Table", "Desk", "Cabinet", "Shelf", "Storage"]
        return any(keyword in obj_name for keyword in furniture_keywords)

    def _calculate_placement_order(self, hierarchy: Dict[str, Dict[str, Set[str]]]):
        """Calculate optimal placement order ensuring parents are placed before children."""
        visited = set()
        self.placement_order = []
        
        def visit(node: str, depth: int = 0):
            if node in visited or node == "ground":
                return
            visited.add(node)
            
            logger.debug(f"{'  ' * depth}Processing node: {node}")
            
            # First ensure all parents are placed
            for parent in hierarchy[node]["parents"]:
                if parent != "ground":
                    visit(parent, depth + 1)
            
            # Then place this object
            if node != "ground":
                self.placement_order.append(node)
                logger.debug(f"{'  ' * depth}Added to placement order: {node}")
            
            # Then place all children
            for child in sorted(hierarchy[node]["children"]):
                visit(child, depth + 1)
        
        # Start with furniture pieces (connected to ground)
        for root in sorted(hierarchy["ground"]["children"]):
            visit(root)
            
        # Then handle any remaining objects
        for node in hierarchy:
            visit(node)
        
        logger.info("Placement order determined:")
        for i, obj in enumerate(self.placement_order):
            parent = next(iter(hierarchy[obj]["parents"]), "none")
            children = hierarchy[obj]["children"]
            logger.info(f"  {i+1}. {obj}")
            logger.info(f"     Parent: {parent}")
            logger.info(f"     Children: {sorted(children)}")
    
    def _optimize_surface_allocation(self):
        """Optimize surface allocation for all objects."""
        logger.info("\n=== Starting Surface Allocation ===")
        
        # Track placement statistics
        placement_stats = {
            "total_objects": len(self.placement_order),
            "placed_objects": 0,
            "failed_placements": 0,
            "rotated_placements": 0,
            "surface_utilization": {}
        }
        
        # Process objects in hierarchy order
        for obj_name in self.placement_order:
            obj_state = self.objects[obj_name]
            if not obj_state.parent:  # Root object
                # Place at origin
                obj_state.position = (0.0, 0.0, obj_state.bbox[2]/2)
                placement_stats["placed_objects"] += 1
                logger.info(f"\n📦 Placed root object {obj_name} at origin")
                continue
                
            parent_state = self.objects[obj_state.parent]
            logger.info(f"\n=== Processing {obj_name} ===")
            
            # Log object properties
            logger.info("📦 Object Properties:")
            logger.info(f"   - Dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
            logger.info(f"   - Parent: {obj_state.parent}")
            logger.info(f"   - Position: {obj_state.position if obj_state.position else 'Not placed'}")
            logger.info(f"   - Children to place: {sorted(obj_state.children_on)}")
            
            # Sort children by size for better packing
            children = sorted(
                [(name, self.objects[name]) for name in obj_state.children_on],
                key=lambda x: x[1].bbox[0] * x[1].bbox[1],
                reverse=True
            )
            
            # Log placement order
            if children:
                logger.info("\n📋 Placement order by size:")
                for child_name, child_state in children:
                    area = child_state.bbox[0] * child_state.bbox[1]
                    logger.info(f"   - {child_name}: {area:.3f}m² ({child_state.bbox[0]:.3f}m × {child_state.bbox[1]:.3f}m)")
            
                # Calculate total surface area needed
                total_area_needed = sum(c[1].bbox[0] * c[1].bbox[1] for c in children)
                available_area = parent_state.bbox[0] * parent_state.bbox[1]
                logger.info(f"\n📐 Surface Area Analysis:")
                logger.info(f"   - Total area needed: {total_area_needed:.3f}m²")
                logger.info(f"   - Available area: {available_area:.3f}m²")
                logger.info(f"   - Required/Available ratio: {(total_area_needed/available_area)*100:.1f}%")
            
            # Try to place each child
            successful_placements = []
            failed_placements = []
            
            for child_name, child_state in children:
                logger.info(f"\n🎯 Attempting to place: {child_name}")
                
                # Find stable position
                position = self._find_stable_position(parent_state, child_state)
                
                if position:
                    # Update object state and occupancy grid
                    child_state.position = position
                    self._update_occupancy_grid(parent_state, child_state)
                    successful_placements.append(child_name)
                    placement_stats["placed_objects"] += 1
                else:
                    # If placement fails, try with rotated orientation
                    logger.info("   ↻ Trying rotated orientation...")
                    # Swap width and length
                    original_bbox = child_state.bbox
                    child_state.bbox = (original_bbox[1], original_bbox[0], original_bbox[2])
                    position = self._find_stable_position(parent_state, child_state)
                    
                    if position:
                        child_state.position = position
                        self._update_occupancy_grid(parent_state, child_state)
                        successful_placements.append(child_name)
                        placement_stats["placed_objects"] += 1
                        placement_stats["rotated_placements"] += 1
                        logger.info(f"   ✅ Successfully placed {child_name} with rotation")
                    else:
                        # Try with reduced spacing
                        logger.info("   ↓ Trying with reduced spacing...")
                        original_spacing = self.min_spacing
                        self.min_spacing = max(0.01, self.min_spacing / 2)  # Reduce spacing but keep minimum
                        position = self._find_stable_position(parent_state, child_state)
                        self.min_spacing = original_spacing  # Restore original spacing
                        
                        if position:
                            child_state.position = position
                            self._update_occupancy_grid(parent_state, child_state)
                            successful_placements.append(child_name)
                            placement_stats["placed_objects"] += 1
                            logger.info(f"   ✅ Successfully placed {child_name} with reduced spacing")
                        else:
                            # Restore original dimensions if placement failed
                            child_state.bbox = original_bbox
                            failed_placements.append(child_name)
                            placement_stats["failed_placements"] += 1
                            logger.warning(f"   ❌ Failed to place {child_name} after all attempts")
            
            # Log placement summary for this object
            if children:
                placed_area = sum(
                    self.objects[name].bbox[0] * self.objects[name].bbox[1]
                    for name in successful_placements
                )
                parent_area = parent_state.bbox[0] * parent_state.bbox[1]
                utilization = (placed_area / parent_area) * 100 if parent_area > 0 else 0
                
                placement_stats["surface_utilization"][obj_name] = utilization
                
                logger.info(f"\n📊 Placement Summary for {obj_name}:")
                logger.info(f"   - Successfully placed: {len(successful_placements)}/{len(children)} objects")
                logger.info(f"   - Surface utilization: {utilization:.1f}%")
                logger.info(f"   - Used area: {placed_area:.3f}m² / {parent_area:.3f}m²")
                if failed_placements:
                    logger.warning(f"   - Failed to place: {failed_placements}")
        
        # Log overall placement statistics
        logger.info("\n=== Final Placement Statistics ===")
        logger.info(f"Total objects processed: {placement_stats['total_objects']}")
        logger.info(f"Successfully placed: {placement_stats['placed_objects']}")
        logger.info(f"Failed placements: {placement_stats['failed_placements']}")
        logger.info(f"Objects placed with rotation: {placement_stats['rotated_placements']}")
        logger.info("\nSurface Utilization by Parent:")
        for obj, util in placement_stats["surface_utilization"].items():
            logger.info(f"  • {obj}: {util:.1f}%")
        
        if placement_stats["failed_placements"] > 0:
            logger.warning(f"\n⚠️ {placement_stats['failed_placements']} objects could not be placed")
    
    def _find_stable_position(self, parent: ObjectState, child: ObjectState) -> Optional[Tuple[float, float, float]]:
        """Find a stable position for child object on parent's surface."""
        try:
            logger.info(f"\n🔍 Finding stable position for {child.name} on {parent.name}:")
            logger.info(f"  📦 Child properties:")
            logger.info(f"     - Original dimensions (w,l,h): {[f'{x:.3f}' for x in child.bbox]}")
            logger.info(f"     - Required area: {(child.bbox[0] * child.bbox[1]):.3f}m²")
            
            logger.info(f"  📦 Parent properties:")
            logger.info(f"     - Dimensions (w,l,h): {[f'{x:.3f}' for x in parent.bbox]}")
            logger.info(f"     - Available area: {(parent.bbox[0] * parent.bbox[1]):.3f}m²")
            
            # Validate size compatibility
            if child.bbox[0] > parent.bbox[0] or child.bbox[1] > parent.bbox[1]:
                logger.warning(f"  ⚠️ Child {child.name} is too large for parent {parent.name}")
                logger.warning(f"     Child needs: {child.bbox[0]:.3f}m × {child.bbox[1]:.3f}m")
                logger.warning(f"     Parent has: {parent.bbox[0]:.3f}m × {parent.bbox[1]:.3f}m")
                return None
            
            # Initialize parent's grid if not already done
            if not parent.available_surface or not parent.grid_resolution:
                parent.init_grid()
                logger.info(f"     - Initialized grid for {parent.name} ({parent.grid_resolution}m resolution)")
            
            grid_size = len(parent.available_surface)
            child_w = max(1, int(child.bbox[0] / parent.grid_resolution))
            child_l = max(1, int(child.bbox[1] / parent.grid_resolution))
            
            logger.info(f"  🔲 Grid properties:")
            logger.info(f"     • Grid size: {grid_size}×{grid_size} cells")
            logger.info(f"     • Child size in grid: {child_w}×{child_l} cells")
            logger.info(f"     • Resolution: {parent.grid_resolution}m per cell")
            
            if child_w > grid_size or child_l > grid_size:
                logger.warning(f"  ⚠️ Child {child.name} too large for parent grid")
                logger.warning(f"     Child needs: {child_w}×{child_l} cells")
                logger.warning(f"     Available: {grid_size}×{grid_size} cells")
                
                # Try rotating the object 90 degrees if it doesn't fit
                if child_l <= grid_size and child_w <= grid_size:
                    logger.info("     Trying 90-degree rotation...")
                    child_w, child_l = child_l, child_w
                    logger.info(f"     New grid size needed: {child_w}×{child_l} cells")
                else:
                    return None
            
            # Find valid position in grid
            valid_pos = None
            attempts = 0
            start_time = time.time()
            
            # Define search patterns (center-out spiral)
            center_i = (grid_size - child_w) // 2
            center_j = (grid_size - child_l) // 2
            
            # Try center position first
            if all(parent.available_surface[center_i+di][center_j+dj] 
                  for di in range(child_w) 
                  for dj in range(child_l)):
                valid_pos = (center_i, center_j)
                logger.info(f"  ✅ Found center position at grid ({center_i}, {center_j})")
            else:
                logger.info("     Center position not available, trying spiral search...")
                # Spiral out from center
                max_radius = max(grid_size - child_w, grid_size - child_l)
                found_position = False
                
                for r in range(1, max_radius + 1):
                    if found_position:
                        break
                        
                    # Try positions in a square pattern around center
                    for i in range(-r, r+1):
                        for j in [-r, r]:  # Top and bottom edges
                            gi = center_i + i
                            gj = center_j + j
                            attempts += 1
                            
                            if (0 <= gi < grid_size - child_w + 1 and 
                                0 <= gj < grid_size - child_l + 1):
                                if all(parent.available_surface[gi+di][gj+dj] 
                                      for di in range(child_w) 
                                      for dj in range(child_l)):
                                    valid_pos = (gi, gj)
                                    found_position = True
                                    logger.info(f"  ✅ Found position at grid ({gi}, {gj}) after {attempts} attempts")
                                    break
                        if found_position:
                            break
                            
                    if found_position:
                        break
                        
                    # Try left and right edges
                    for i in [-r, r]:  # Left and right edges
                        for j in range(-r+1, r):
                            gi = center_i + i
                            gj = center_j + j
                            attempts += 1
                            
                            if (0 <= gi < grid_size - child_w + 1 and 
                                0 <= gj < grid_size - child_l + 1):
                                if all(parent.available_surface[gi+di][gj+dj] 
                                      for di in range(child_w) 
                                      for dj in range(child_l)):
                                    valid_pos = (gi, gj)
                                    found_position = True
                                    logger.info(f"  ✅ Found position at grid ({gi}, {gj}) after {attempts} attempts")
                                    break
                        if found_position:
                            break
            
            search_time = time.time() - start_time
            
            if valid_pos is None:
                logger.warning(f"  ❌ No valid position found after {attempts} attempts ({search_time:.3f}s)")
                logger.warning(f"     Placement failed for {child.name} on {parent.name}")
                return None
            
            # Convert grid position to world coordinates
            grid_x, grid_y = valid_pos
            
            # Calculate world position relative to parent's position
            if parent.position is None:
                # If parent has no position yet (e.g., it's the table), place at origin
                parent.position = (0.0, 0.0, parent.bbox[2]/2)
                logger.info(f"  📍 Set parent {parent.name} position to origin")
            
            # Calculate offset from parent center
            offset_x = (grid_x * parent.grid_resolution) - (parent.bbox[0] / 2)
            offset_y = (grid_y * parent.grid_resolution) - (parent.bbox[1] / 2)
            
            # Add small height offset for physics stability
            height_offset = 0.001  # 1mm offset
            
            # Calculate final world position
            world_x = parent.position[0] + offset_x
            world_y = parent.position[1] + offset_y
            world_z = parent.position[2] + parent.bbox[2]/2 + child.bbox[2]/2 + height_offset
            
            logger.info(f"  📍 Position calculation:")
            logger.info(f"     • Parent position: {[f'{x:.3f}' for x in parent.position]}")
            logger.info(f"     • Grid offsets: ({offset_x:.3f}, {offset_y:.3f})")
            logger.info(f"     • Final position: ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")
            
            return (world_x, world_y, world_z)
            
        except Exception as e:
            logger.error(f"  ❌ Error finding stable position for {child.name}: {str(e)}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return None
    
    def _update_occupancy_grid(self, parent: ObjectState, child: ObjectState):
        """Update parent's occupancy grid after placing child object."""
        if child.position is None or not parent.available_surface or not parent.grid_resolution:
            return
            
        # Convert world position to grid coordinates
        grid_x = int((child.position[0] - parent.position[0] + parent.bbox[0]/2) / parent.grid_resolution)
        grid_y = int((child.position[1] - parent.position[1] + parent.bbox[1]/2) / parent.grid_resolution)
        
        # Mark occupied cells
        child_w = int(child.bbox[0] / parent.grid_resolution)
        child_l = int(child.bbox[1] / parent.grid_resolution)
        
        cells_marked = 0
        for i in range(child_w):
            for j in range(child_l):
                if 0 <= grid_x + i < len(parent.available_surface) and \
                   0 <= grid_y + j < len(parent.available_surface[0]):
                    parent.available_surface[grid_x + i][grid_y + j] = False
                    cells_marked += 1
        
        logger.debug(f"Marked {cells_marked} cells as occupied for {child.name}")
    
    def _log_placement_stats(self):
        """Log comprehensive placement statistics."""
        stats = self.placement_stats
        
        logger.info("\nPlacement Statistics:")
        logger.info("-------------------")
        logger.info(f"Total objects placed: {len(self.placement_order)}")
        
        if stats["placement_failures"]:
            logger.warning("\nPlacement Failures:")
            for parent, children in stats["placement_failures"].items():
                logger.warning(f"  {parent}: Failed to place {len(children)} children")
                for child in children:
                    logger.warning(f"    - {child}")
        
        logger.info("\nSurface Utilization:")
        for obj, utilization in stats["surface_utilization"].items():
            logger.info(f"  {obj}: {utilization:.2%}")
        
        logger.info("\nHierarchy Statistics:")
        for stat, value in stats["hierarchy_stats"].items():
            logger.info(f"  {stat}: {value}")

    def _get_partnet_id(self, obj_name: str) -> Optional[str]:
        """Map semantic object name to PartNet ID."""
        try:
            # Extract base name without number (e.g., "Monitor_0" -> "Monitor")
            base_name = obj_name.split('_')[0]
            
            # Get list of possible IDs for this object type
            possible_ids = self.object_id_map.get(base_name)
            if not possible_ids:
                logger.warning(f"No PartNet IDs found for object type: {base_name}")
                return None
            
            # Use consistent ID for same numbered objects (e.g., all Monitor_0 use first ID)
            if '_' in obj_name:
                obj_num = int(obj_name.split('_')[1])
                id_idx = obj_num % len(possible_ids)
                return possible_ids[id_idx]
            
            # If no number, use first ID
            return possible_ids[0]
            
        except Exception as e:
            logger.error(f"Error mapping object name to PartNet ID: {str(e)}")
            return None

    def _get_mesh_dimensions(self, obj_name: str) -> Optional[Tuple[float, float, float]]:
        """Extract dimensions from mesh files and URDF transforms in the PartNet dataset."""
        try:
            # Get PartNet ID for this object
            obj_id = self._get_partnet_id(obj_name)
            if not obj_id:
                return None
            
            # Find the object directory in the dataset
            dataset_path = Path(project_root) / "src" / "datasets" / "partnet-mobility-v0" / "dataset"
            obj_dir = dataset_path / obj_id
            
            if not obj_dir.exists():
                logger.warning(f"Could not find mesh directory for object {obj_name} (ID: {obj_id})")
                return None
            
            # First read URDF file to get transforms
            urdf_file = obj_dir / "mobility.urdf"
            if not urdf_file.exists():
                logger.warning(f"No URDF file found for object {obj_name} (ID: {obj_id})")
                return None
            
            import xml.etree.ElementTree as ET
            tree = ET.parse(urdf_file)
            root = tree.getroot()
            
            # Extract transforms from URDF
            transforms = []
            for link in root.findall(".//link"):
                for visual in link.findall(".//visual"):
                    origin = visual.find("origin")
                    if origin is not None:
                        # Get position and rotation
                        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                        rpy = [float(r) for r in origin.get("rpy", "0 0 0").split()]
                        transforms.append((xyz, rpy))
            
            # Process mesh files with transforms
            min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
            max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
            
            # Get all OBJ files in the textured_objs directory
            obj_files = list((obj_dir / "textured_objs").glob("*.obj"))
            if not obj_files:
                logger.warning(f"No mesh files found for object {obj_name} (ID: {obj_id})")
                return None
            
            import numpy as np
            from scipy.spatial.transform import Rotation
            
            vertex_count = 0
            for obj_file in obj_files:
                vertices = []
                with open(obj_file, 'r') as f:
                    for line in f:
                        if line.startswith('v '):  # Vertex line
                            _, x, y, z = line.split()
                            vertices.append([float(x), float(y), float(z)])
                
                if not vertices:
                    continue
                
                vertices = np.array(vertices)
                vertex_count += len(vertices)
                
                # Apply URDF transforms to vertices
                for xyz, rpy in transforms:
                    # Create rotation matrix from RPY angles
                    R = Rotation.from_euler('xyz', rpy).as_matrix()
                    
                    # Apply rotation and translation
                    vertices = vertices @ R.T + xyz
                
                # Update bounds
                min_x = min(min_x, vertices[:, 0].min())
                min_y = min(min_y, vertices[:, 1].min())
                min_z = min(min_z, vertices[:, 2].min())
                max_x = max(max_x, vertices[:, 0].max())
                max_y = max(max_y, vertices[:, 1].max())
                max_z = max(max_z, vertices[:, 2].max())
            
            if min_x == float('inf'):
                logger.warning(f"No vertices found in mesh files for object {obj_name} (ID: {obj_id})")
                return None
            
            # Calculate dimensions
            width = max_x - min_x
            length = max_y - min_y
            height = max_z - min_z
            
            # Add small padding for physics stability
            padding = max(0.01, min(width, length, height) * 0.05)  # 1cm or 5% of smallest dimension
            width += padding
            length += padding
            height += padding
            
            logger.info(f"Extracted mesh dimensions for {obj_name} (ID: {obj_id}):")
            logger.info(f"  Width: {width:.3f}m")
            logger.info(f"  Length: {length:.3f}m")
            logger.info(f"  Height: {height:.3f}m")
            logger.info(f"  Vertices processed: {vertex_count}")
            logger.info(f"  Transforms applied: {len(transforms)}")
            
            return (abs(width), abs(length), abs(height))
        except Exception as e:
            logger.error(f"Error extracting mesh dimensions for {obj_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _initialize_object_states(self, 
                                hierarchy: Dict[str, Dict[str, Set[str]]], 
                                object_states: Optional[Dict[str, ObjectState]] = None):
        """Initialize object states with hierarchy information and provided states."""
        logger.info("Initializing object states:")
        
        # First pass: Create basic states
        for obj_name in hierarchy:
            if obj_name == "ground":
                continue
            
            logger.info(f"\nProcessing {obj_name}:")
            
            # Create or update object state
            if obj_name not in self.objects:
                # Try to get dimensions from mesh first
                mesh_dims = self._get_mesh_dimensions(obj_name)
                
                if object_states and obj_name in object_states:
                    provided_state = object_states[obj_name]
                    logger.info(f"  Using provided state:")
                    logger.info(f"  - Bbox: {provided_state.bbox}")
                    logger.info(f"  - Position: {provided_state.position if hasattr(provided_state, 'position') else None}")
                    
                    # If we have mesh dimensions, use them to validate/adjust provided dimensions
                    if mesh_dims:
                        scale_factor = 1.0
                        if provided_state.bbox:
                            # Calculate scale factor more robustly
                            ratios = []
                            for i in range(3):
                                if mesh_dims[i] > 1e-6:  # Avoid division by very small numbers
                                    ratios.append(provided_state.bbox[i] / mesh_dims[i])
                            if ratios:
                                # Use median ratio to avoid outliers
                                scale_factor = sorted(ratios)[len(ratios)//2]
                            logger.info(f"  Adjusting provided dimensions by scale factor: {scale_factor:.3f}")
                        
                        bbox = tuple(d * scale_factor for d in mesh_dims)
                    else:
                        bbox = provided_state.bbox
                    
                    self.objects[obj_name] = ObjectState(
                        name=obj_name,
                        bbox=bbox,
                        position=provided_state.position if hasattr(provided_state, 'position') else None
                    )
                else:
                    if mesh_dims:
                        logger.info(f"  Using mesh dimensions for {obj_name}")
                        self.objects[obj_name] = ObjectState(
                            name=obj_name,
                            bbox=mesh_dims
                        )
                    else:
                        # Use category-specific default sizes
                        base_name = obj_name.split('_')[0]
                        default_sizes = {
                            "Table": (1.5, 0.75, 0.75),
                            "Monitor": (0.6, 0.4, 0.4),
                            "Keyboard": (0.45, 0.15, 0.03),
                            "Mouse": (0.12, 0.07, 0.04)
                        }
                        default_size = default_sizes.get(base_name, (1.0, 1.0, 1.0))
                        logger.warning(f"  Using default values for {obj_name}: {default_size}")
                        self.objects[obj_name] = ObjectState(
                            name=obj_name,
                            bbox=default_size
                        )
            
            obj_state = self.objects[obj_name]
            
            # Set hierarchy information
            parent = next(iter(hierarchy[obj_name]["parents"]), None)
            if parent == "ground":
                parent = None
            obj_state.parent = parent
            obj_state.children_on = hierarchy[obj_name]["children"]
            
            logger.info(f"  Final state:")
            logger.info(f"  - Parent: {obj_state.parent}")
            logger.info(f"  - Children: {sorted(obj_state.children_on)}")
            logger.info(f"  - Bbox: {[f'{x:.3f}' for x in obj_state.bbox]}")
            if obj_state.position:
                logger.info(f"  - Position: {[f'{x:.3f}' for x in obj_state.position]}")
        
        # Second pass: Initialize placement grids for objects that will have children
        for obj_name, obj_state in self.objects.items():
            if obj_state.children_on:
                if not obj_state.grid_resolution:
                    # Initialize grid with resolution based on smallest child
                    min_child_dim = float('inf')
                    for child in obj_state.children_on:
                        if child in self.objects:
                            child_state = self.objects[child]
                            min_dim = min(child_state.bbox[0], child_state.bbox[1])
                            min_child_dim = min(min_child_dim, min_dim)
                    
                    # Use grid resolution of 1/10th of smallest child or 0.1m, whichever is smaller
                    resolution = min(0.1, min_child_dim / 10)
                    obj_state.init_grid(resolution)
                    logger.info(f"Initialized placement grid for {obj_name} with resolution {resolution:.3f}m")
        
        logger.info(f"\nInitialized states for {len(self.objects)} objects")
        for obj_name, obj_state in self.objects.items():
            logger.info(f"  • {obj_name}:")
            logger.info(f"    - Bbox: {[f'{x:.3f}' for x in obj_state.bbox]}")
            logger.info(f"    - Parent: {obj_state.parent}")
            logger.info(f"    - Children: {sorted(obj_state.children_on)}")
            if obj_state.grid_resolution:
                logger.info(f"    - Grid resolution: {obj_state.grid_resolution:.3f}m")

    def _optimize_object_scales(self, hierarchy: Dict[str, Dict[str, Set[str]]]) -> bool:
        """Optimize object scales by modifying URDF and mesh files directly."""
        logger.info("Starting scale optimization...")
        
        # Store original scales for comparison
        original_scales = {name: state.bbox for name, state in self.objects.items()}
        
        # Prepare context for LLM
        context = []
        for obj_name, obj_state in self.objects.items():
            parent = obj_state.parent if obj_state.parent else "ground"
            children = sorted(obj_state.children_on) if obj_state.children_on else []
            
            # Include image information if available
            image_info = getattr(obj_state, 'image_info', None)
            
            context.append({
                "name": obj_name,
                "category": ''.join([c for c in obj_name if not c.isdigit()]).rstrip('_'),
                "current_dimensions": {
                    "width": obj_state.bbox[0],
                    "length": obj_state.bbox[1],
                    "height": obj_state.bbox[2]
                },
                "parent": parent,
                "children": children,
                "image_info": image_info
            })
        
        # Get scaling suggestions from LLM
        scaling_suggestions = self._get_scale_suggestions(context)
        
        # Track if any scales changed
        scales_changed = False
        
        # Initialize PartNet manager for file updates
        dataset_path = Path(project_root) / "src" / "datasets" / "partnet-mobility-v0"
        partnet_manager = PartNetManager(str(dataset_path))
        
        # Apply scaling suggestions by updating URDF and mesh files
        for obj_name, scale_info in scaling_suggestions.items():
            if obj_name in self.objects:
                obj_state = self.objects[obj_name]
                old_bbox = obj_state.bbox
                
                # Calculate scale factors
                scale_factors = (
                    scale_info["scale_x"],
                    scale_info["scale_y"],
                    scale_info["scale_z"]
                )
                
                # Get PartNet ID for this object
                obj_id = self._get_partnet_id(obj_name)
                if not obj_id:
                    logger.warning(f"Could not find PartNet ID for {obj_name}, skipping scale update")
                    continue
                
                # Get image info if available
                image_info = getattr(obj_state, 'image_info', None)
                
                # Update URDF and mesh files
                if partnet_manager.update_object_scale(obj_id, scale_factors, image_info=image_info):
                    # Update object state with new dimensions
                    new_bbox = (
                        old_bbox[0] * scale_factors[0],
                        old_bbox[1] * scale_factors[1],
                        old_bbox[2] * scale_factors[2]
                    )
                    
                    # Check if scale actually changed
                    if new_bbox != old_bbox:
                        scales_changed = True
                        logger.info(f"\nScaling {obj_name}:")
                        logger.info(f"  Reasoning: {scale_info['reasoning']}")
                        logger.info(f"  Old dimensions: {[f'{x:.3f}' for x in old_bbox]}")
                        logger.info(f"  New dimensions: {[f'{x:.3f}' for x in new_bbox]}")
                        logger.info(f"  Scale factors: x={scale_factors[0]:.3f}, y={scale_factors[1]:.3f}, z={scale_factors[2]:.3f}")
                        
                        if image_info:
                            logger.info("  Scale validated against image information")
                            if image_info.get('depth'):
                                logger.info("  Depth information used for scaling")
                        
                        obj_state.bbox = new_bbox
                else:
                    logger.error(f"Failed to update scale for {obj_name}")
        
        # Clean up by restoring original files
        logger.info("\nRestoring original files...")
        if not partnet_manager.cleanup_scaled_files():
            logger.warning("Some files could not be restored to original state")
        
        if scales_changed:
            logger.info("\nScale changes were temporarily applied and then reverted")
        else:
            logger.info("\nNo scale changes were needed")
        
        return scales_changed

    def _get_scale_suggestions(self, context: List[Dict]) -> Dict:
        """Get scaling suggestions from LLM."""
        try:
            # Create prompt for LLM
            prompt = f"""Given a scene with objects and their current dimensions, analyze the provided images and suggest scaling adjustments to make the scene realistic.

Each object has:
- Current dimensions (width, length, height) in meters
- Parent-child relationships
- Image information (if available):
  - 2D bounding box coordinates
  - Scene images showing the object
  - Depth information (if available)

Current scene objects:
{json.dumps(context, indent=2)}

For each object that has image information:
1. Use the 2D bounding box and image dimensions to validate object size
2. If depth information is available, use it to better estimate real-world scale
3. Consider the object's relative size in the image compared to other objects
4. Check if the current dimensions match what's visible in the image

For all objects:
1. Verify if the size is realistic for its category
2. Check if the size is appropriate relative to its parent
3. Ensure it can accommodate its children (if any)
4. Maintain proper proportions with sibling objects

Return a JSON object with scaling factors for each object in this format:
{{
    "object_name": {{
        "scale_x": float,  # Multiply current width by this factor
        "scale_y": float,  # Multiply current length by this factor
        "scale_z": float,  # Multiply current height by this factor
        "reasoning": "string"  # Explanation for the scaling decision, including image-based validation
    }},
    ...
}}

Consider both the image evidence and common sense object sizes for the simulation.
Focus on making the scene physically plausible and visually consistent with the provided images.
Explain your reasoning for each scaling decision, especially when using image information.

IMPORTANT: Only return the JSON object with no additional text."""

            # Call LLM
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a computer vision expert specializing in 3D scene understanding and physical object relationships. Your task is to analyze object dimensions using both image evidence and common sense knowledge to suggest realistic scaling adjustments."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )

            # Parse response
            scaling_suggestions = json.loads(response.choices[0].message.content)
            
            # Validate response format
            for obj_name, scale_info in scaling_suggestions.items():
                required_keys = {"scale_x", "scale_y", "scale_z", "reasoning"}
                if not all(key in scale_info for key in required_keys):
                    raise ValueError(f"Invalid scale info format for {obj_name}")
                if not all(isinstance(scale_info[f"scale_{axis}"], (int, float)) 
                          for axis in ['x', 'y', 'z']):
                    raise ValueError(f"Invalid scale factors for {obj_name}")

            return scaling_suggestions

        except Exception as e:
            logger.error(f"Failed to get scaling suggestions from LLM: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def _reinitialize_placement_grids(self):
        """Reinitialize placement grids for all objects after scale changes."""
        for obj_name, obj_state in self.objects.items():
            if obj_state.children_on:
                # Reset existing grid
                obj_state.available_surface = None
                obj_state.grid_resolution = None
                
                # Calculate new grid resolution based on updated child sizes
                min_child_dim = float('inf')
                for child_name in obj_state.children_on:
                    if child_name in self.objects:
                        child_state = self.objects[child_name]
                        min_dim = min(child_state.bbox[0], child_state.bbox[1])
                        min_child_dim = min(min_child_dim, min_dim)
                
                # Use grid resolution of 1/10th of smallest child or 0.1m, whichever is smaller
                resolution = min(0.1, min_child_dim / 10)
                obj_state.init_grid(resolution)
                logger.info(f"Reinitialized grid for {obj_name} with resolution {resolution:.3f}m")
                logger.info(f"  - Parent surface: {obj_state.bbox[0]:.3f}m × {obj_state.bbox[1]:.3f}m")
                logger.info(f"  - Grid size: {len(obj_state.available_surface)}×{len(obj_state.available_surface)} cells")

    def _scale_mesh_file(self, mesh_path: str, scale_factors: Tuple[float, float, float], image_info: Optional[Dict] = None):
        """Scale mesh using trimesh for accurate transformations and image-based validation.
        
        Args:
            mesh_path: Path to the mesh file
            scale_factors: (scale_x, scale_y, scale_z) scaling factors
            image_info: Optional dict containing image paths, 2D bounding boxes, and depth info
        """
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
            
            # Get original mesh dimensions
            original_dims = mesh.extents
            logger.info(f"Original mesh dimensions: {original_dims}")
            
            # If we have image info, validate scale factors
            if image_info and image_info['bbox_2d'] and image_info['paths']:
                try:
                    # Load first image for reference
                    img = cv2.imread(image_info['paths'][0])
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        bbox = image_info['bbox_2d']
                        
                        # Calculate relative size in image
                        bbox_width = bbox[2] - bbox[0]
                        bbox_height = bbox[3] - bbox[1]
                        relative_width = bbox_width / img_width
                        relative_height = bbox_height / img_height
                        
                        # Use depth information if available
                        if image_info['depth'] is not None:
                            depth = image_info['depth']
                            # Adjust scale factors based on depth
                            depth_scale = 1.0 / depth if depth > 0 else 1.0
                            scale_factors = tuple(s * depth_scale for s in scale_factors)
                            logger.info(f"Adjusted scale factors using depth: {scale_factors}")
                        
                        # Calculate expected dimensions based on image
                        expected_width = relative_width * scale_factors[0]
                        expected_height = relative_height * scale_factors[1]
                        
                        # Compare with mesh dimensions and adjust if needed
                        width_ratio = expected_width / original_dims[0]
                        height_ratio = expected_height / original_dims[1]
                        
                        if abs(1 - width_ratio) > 0.2 or abs(1 - height_ratio) > 0.2:
                            logger.warning(f"Large discrepancy between image and mesh dimensions")
                            logger.warning(f"Image-based dimensions: {expected_width:.3f} x {expected_height:.3f}")
                            logger.warning(f"Mesh dimensions: {original_dims[0]:.3f} x {original_dims[1]:.3f}")
                            
                            # Adjust scale factors to better match image
                            scale_factors = (
                                scale_factors[0] * width_ratio,
                                scale_factors[1] * height_ratio,
                                scale_factors[2]  # Keep Z scale as is
                            )
                            logger.info(f"Adjusted scale factors to match image: {scale_factors}")
                except Exception as e:
                    logger.warning(f"Error during image-based scale validation: {str(e)}")
                    logger.warning("Proceeding with original scale factors")
            
            # Create scaling matrix
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors[0]
            scale_matrix[1, 1] = scale_factors[1]
            scale_matrix[2, 2] = scale_factors[2]
            
            # Apply scaling transformation
            mesh.apply_transform(scale_matrix)
            
            # Export scaled mesh
            try:
                export_options = {}
                if mesh_path.lower().endswith('.obj'):
                    # Ensure proper OBJ export settings
                    export_options = {
                        'vertex_normal': True,
                        'include_texture': True,
                        'include_materials': True,
                        'resolver': trimesh.resolvers.FilePathResolver()
                    }
                
                mesh.export(mesh_path, **export_options)
                logger.info(f"Successfully scaled and saved mesh: {mesh_path}")
                logger.info(f"  Scale factors: x={scale_factors[0]:.3f}, y={scale_factors[1]:.3f}, z={scale_factors[2]:.3f}")
                logger.info(f"  Original dimensions: {original_dims}")
                logger.info(f"  New dimensions: {mesh.extents}")
                logger.info(f"  Vertices: {len(mesh.vertices)}")
                logger.info(f"  Faces: {len(mesh.faces)}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to export scaled mesh {mesh_path}: {str(e)}")
                # Try to restore from backup
                if os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(backup_path, mesh_path)
                return False
                
        except Exception as e:
            logger.error(f"Error scaling mesh {mesh_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _optimize_object_selection(self, hierarchy: Dict[str, Dict[str, Set[str]]]) -> bool:
        """Iteratively optimize object selection by finding better-fitting PartNet objects.
        
        This method:
        1. Starts with root objects (furniture)
        2. For each object, finds all compatible PartNet objects
        3. Evaluates each candidate based on its ability to accommodate children
        4. Updates selections if better matches are found
        
        Returns:
            bool: True if any objects were updated
        """
        logger.info("\n=== Starting Object Selection Optimization ===")
        
        # Track if any changes were made
        changes_made = False
        
        # Process objects in hierarchy order (bottom-up)
        objects_by_level = defaultdict(list)
        for obj_name in hierarchy:
            if obj_name == "ground":
                continue
            level = len(hierarchy[obj_name]["children"])
            objects_by_level[level].append(obj_name)
        
        # Initialize PartNet manager
        dataset_path = Path(project_root) / "src" / "datasets" / "partnet-mobility-v0"
        partnet_manager = PartNetManager(str(dataset_path))
        
        # Process levels from leaves to root
        for level in sorted(objects_by_level.keys()):
            objects = objects_by_level[level]
            logger.info(f"\nProcessing objects at level {level} (children count)")
            
            for obj_name in objects:
                obj_state = self.objects[obj_name]
                base_name = obj_name.split('_')[0]
                
                logger.info(f"\n📦 Analyzing {obj_name}:")
                logger.info(f"  Current dimensions: {[f'{x:.3f}' for x in obj_state.bbox]}")
                
                # Calculate required space for children
                if obj_state.children_on:
                    total_child_area = sum(
                        self.objects[child].bbox[0] * self.objects[child].bbox[1]
                        for child in obj_state.children_on
                    )
                    max_child_height = max(
                        self.objects[child].bbox[2]
                        for child in obj_state.children_on
                    )
                    logger.info(f"  Required child area: {total_child_area:.3f}m²")
                    logger.info(f"  Max child height: {max_child_height:.3f}m")
                else:
                    total_child_area = 0
                    max_child_height = 0
                
                # Get all possible PartNet IDs for this object type
                possible_ids = self.object_id_map.get(base_name, [])
                if not possible_ids:
                    logger.warning(f"  No PartNet IDs found for type: {base_name}")
                    continue
                
                logger.info(f"  Found {len(possible_ids)} potential matches")
                
                # Evaluate each candidate
                best_match = None
                best_score = float('-inf')
                
                for obj_id in possible_ids:
                    # Get dimensions for this candidate
                    dims = self._get_mesh_dimensions_for_id(obj_id)
                    if not dims:
                        continue
                    
                    # Calculate score based on:
                    # 1. Surface area relative to children's needs
                    # 2. Height appropriateness
                    # 3. Overall proportions
                    surface_area = dims[0] * dims[1]
                    area_ratio = surface_area / total_child_area if total_child_area > 0 else 1.0
                    height_ratio = dims[2] / max_child_height if max_child_height > 0 else 1.0
                    
                    # Penalize if too small or too large
                    if area_ratio < 1.0:  # Too small
                        area_score = -10
                    elif area_ratio > 3.0:  # Too large
                        area_score = -5
                    else:
                        area_score = 10 - abs(2 - area_ratio) * 5  # Best score when area_ratio is around 2
                    
                    if height_ratio < 0.1:  # Too short
                        height_score = -10
                    elif height_ratio > 5.0:  # Too tall
                        height_score = -5
                    else:
                        height_score = 10 - abs(1 - height_ratio) * 5  # Best score when height_ratio is around 1
                    
                    # Calculate proportion score (prefer reasonable width/length ratios)
                    aspect_ratio = max(dims[0]/dims[1], dims[1]/dims[0])
                    proportion_score = 10 - min(abs(aspect_ratio - 1) * 2, 10)  # Penalize extreme proportions
                    
                    # Combine scores
                    total_score = area_score + height_score + proportion_score
                    
                    logger.debug(f"  Evaluating ID {obj_id}:")
                    logger.debug(f"    Dimensions: {[f'{x:.3f}' for x in dims]}")
                    logger.debug(f"    Area ratio: {area_ratio:.2f} (score: {area_score:.1f})")
                    logger.debug(f"    Height ratio: {height_ratio:.2f} (score: {height_score:.1f})")
                    logger.debug(f"    Proportion score: {proportion_score:.1f}")
                    logger.debug(f"    Total score: {total_score:.1f}")
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_match = (obj_id, dims)
                
                # Update object if better match found
                if best_match and best_match[1] != obj_state.bbox:
                    old_dims = obj_state.bbox
                    new_dims = best_match[1]
                    obj_state.bbox = new_dims
                    changes_made = True
                    
                    logger.info(f"  ✨ Found better match (ID: {best_match[0]}):")
                    logger.info(f"    Old dimensions: {[f'{x:.3f}' for x in old_dims]}")
                    logger.info(f"    New dimensions: {[f'{x:.3f}' for x in new_dims]}")
                    logger.info(f"    Score: {best_score:.1f}")
                else:
                    logger.info("  ✓ Current selection is optimal")
        
        if changes_made:
            logger.info("\n🔄 Object selections were optimized")
        else:
            logger.info("\n✓ No optimization needed")
        
        return changes_made

    def _get_mesh_dimensions_for_id(self, obj_id: str) -> Optional[Tuple[float, float, float]]:
        """Get mesh dimensions for a specific PartNet ID."""
        try:
            dataset_path = Path(project_root) / "src" / "datasets" / "partnet-mobility-v0" / "dataset"
            obj_dir = dataset_path / obj_id
            
            if not obj_dir.exists():
                return None
            
            # Load and process mesh files
            mesh_files = list((obj_dir / "textured_objs").glob("*.obj"))
            if not mesh_files:
                return None
            
            # Use trimesh for accurate bounds
            import trimesh
            vertices = []
            
            for mesh_file in mesh_files:
                try:
                    mesh = trimesh.load(mesh_file)
                    if isinstance(mesh, trimesh.Trimesh):
                        vertices.extend(mesh.vertices)
                except Exception as e:
                    logger.debug(f"Error loading mesh {mesh_file}: {str(e)}")
                    continue
            
            if not vertices:
                return None
            
            # Calculate bounds
            vertices = np.array(vertices)
            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)
            
            # Calculate dimensions
            dims = maxs - mins
            
            # Add small padding for physics stability
            padding = max(0.01, min(dims) * 0.05)  # 1cm or 5% of smallest dimension
            dims += padding
            
            return tuple(abs(d) for d in dims)
            
        except Exception as e:
            logger.debug(f"Error getting dimensions for ID {obj_id}: {str(e)}")
            return None

def parse_llm_output(output_text: str) -> Tuple[List[str], List[SemanticRelation]]:
    """Parse LLM output into a scene graph structure."""
    try:
        sections = output_text.split("Relationships:")
        objects_section = sections[0].split("Objects:")[1].strip()
        objects = [obj.strip() for obj in objects_section.split("\n") if obj.strip()]
        
        logger.info(f"Parsed {len(objects)} objects from LLM output")
        
        # Create directed graph for relationship analysis
        G = nx.DiGraph()
        
        # Add all objects as nodes
        for obj in objects:
            G.add_node(obj)
        
        # Track different types of relationships
        support_relations = []  # on, above relationships
        spatial_relations = []  # next_to, aligned_with, etc.
        
        # Parse and categorize relationships
        edge_colors = []  # Track edge colors for visualization
        edge_labels = {}  # Track edge labels for visualization
        
        for line in sections[1].strip().split("\n"):
            if not line.strip():
                continue
                
            source, relation_type, target, confidence = [x.strip() for x in line.split("|")]
            confidence = float(confidence)
            
            # Create semantic relation
            relation = SemanticRelation(
                source=source,
                relation=relation_type,
                target=target,
                confidence=confidence
            )
            
            # Add edge to graph with attributes
            G.add_edge(source, target, 
                      relation_type=relation_type,
                      confidence=confidence)
            
            # Track edge visualization properties
            if relation_type in ["on", "above"]:
                support_relations.append(relation)
                edge_colors.append('red')  # Support relationships in red
            else:
                spatial_relations.append(relation)
                edge_colors.append('blue')  # Spatial relationships in blue
            
            edge_labels[(source, target)] = f"{relation_type}\n({confidence:.2f})"
        
        # Analyze and log graph structure
        logger.info("\nAnalyzing scene graph structure:")
        
        # Find root objects (likely furniture/tables)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        logger.info(f"Root objects (no incoming edges): {roots}")
        
        # Find leaf objects (no outgoing edges)
        leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
        logger.info(f"Leaf objects (no outgoing edges): {leaves}")
        
        # Identify support hierarchies
        support_hierarchies = defaultdict(list)
        for rel in support_relations:
            support_hierarchies[rel.target].append(rel.source)
        
        logger.info("\nSupport hierarchies:")
        for parent, children in support_hierarchies.items():
            logger.info(f"  {parent} supports: {children}")
        
        # Find spatial relationships between siblings
        sibling_relations = []
        for rel in spatial_relations:
            # Check if objects share same parent
            source_parents = set(G.predecessors(rel.source))
            target_parents = set(G.predecessors(rel.target))
            if source_parents & target_parents:  # If they share any parents
                sibling_relations.append(rel)
        
        logger.info("\nSpatial relationships between siblings:")
        for rel in sibling_relations:
            logger.info(f"  {rel.source} {rel.relation} {rel.target}")
        
        # Detect any cycles or conflicts
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                logger.warning(f"Detected cycles in relationships: {cycles}")
        except nx.NetworkXNoCycle:
            logger.info("No relationship cycles detected")
        
        # Visualize the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)  # k controls spacing
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.7)
        
        # Draw edges with different colors for different relationship types
        edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, 
                             edgelist=edges,
                             edge_color=edge_colors,
                             arrows=True,
                             arrowsize=20)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos,
                              font_size=10,
                              font_weight='bold')
        
        # Add edge labels
        nx.draw_networkx_edge_labels(G, pos,
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        # Add legend
        plt.plot([], [], 'r-', label='Support Relation')
        plt.plot([], [], 'b-', label='Spatial Relation')
        plt.legend()
        
        plt.title("Scene Graph Structure")
        plt.axis('off')
        
        # Save the graph visualization
        plt.savefig('scene_graph.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("\nGraph visualization saved as 'scene_graph.png'")
        
        # Combine support and spatial relations, prioritizing support relations
        all_relations = support_relations + spatial_relations
        
        return objects, all_relations
        
    except Exception as e:
        logger.error(f"Failed to parse LLM output: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Invalid LLM output format: {str(e)}")