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
import xml.etree.ElementTree as ET
from .scene_visualizer import visualize_scene_plan

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
        self.global_scale = 0.001  # Global scale factor for all objects
        
        # Map semantic names to PartNet IDs
        self.object_id_map = {
            "Table": ["26652", "45262", "45671"],  # Table IDs
            "Monitor": ["4627", "102401", "102694"],  # Monitor IDs
            "Keyboard": ["10238", "10305", "10707"],  # Keyboard IDs
            "Mouse": ["101416", "101425", "101511"]  # Mouse IDs
        }
    
    def precompute_placement(self, relations: List[SpatialRelation],
                            object_states: Optional[Dict[str, ObjectState]] = None) -> Tuple[List[str], Dict[str, ObjectState]]:
        """Precompute optimal object placement order and hierarchy."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Placement Planning")
        logger.info(f"{'='*80}")
        logger.info(f"Input: {len(relations)} relations, {len(object_states) if object_states else 0} objects")
        
        # Initialize return values with defaults
        self.placement_order = []
        if not self.objects and object_states:
            self.objects = object_states
            
        # Generate visualization of initial state
        initial_objects = [
            {
                'name': name,
                'position': [float(p) if isinstance(p, (int, float, str)) else float(p[0]) for p in (state.position if state.position else [0, 0, 0])],
                'dimensions': [float(d) if isinstance(d, (int, float, str)) else float(d[0]) for d in (state.bbox if state.bbox else [0.1, 0.1, 0.1])]
            }
            for name, state in self.objects.items()
        ]
        logger.info("\nPreparing visualization with objects:")
        for obj in initial_objects:
            logger.info(f"  {obj['name']}:")
            logger.info(f"    Position: {obj['position']}")
            logger.info(f"    Dimensions: {obj['dimensions']}")
        visualize_scene_plan(initial_objects, "initial_scene.png")
        logger.info("\nGenerated initial scene visualization: initial_scene.png")
        
        try:
            # First pass: Build object hierarchy
            logger.info("\n=== 1. Building Object Hierarchy ===")
            try:
                hierarchy = self._build_hierarchy(relations)
                self._log_hierarchy_stats(hierarchy)
            except Exception as e:
                logger.error(f"Error building hierarchy: {str(e)}")
                # Create simple hierarchy if building fails
                hierarchy = {"ground": {"children": set(), "parents": set()}}
                for obj_name in self.objects:
                    hierarchy[obj_name] = {"children": set(), "parents": {"ground"}}
                    hierarchy["ground"]["children"].add(obj_name)
            
            # Initialize object states
            logger.info("\n=== 2. Initializing Object States ===")
            try:
                self._initialize_object_states(hierarchy, object_states)
                logger.info(f"Initialized {len(self.objects)} object states")
                for obj_name, obj_state in self.objects.items():
                    logger.info(f"\n📦 {obj_name}:")
                    logger.info(f"   Initial dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
                    if obj_state.position:
                        logger.info(f"   Initial position (x,y,z): {[f'{x:.3f}' for x in obj_state.position]}")
            except Exception as e:
                logger.error(f"Error initializing object states: {str(e)}")
            
            # Generate visualization after initialization
            initialized_objects = [
                {
                    'name': name,
                    'position': state.position if state.position else [0, 0, 0],
                    'dimensions': state.bbox if state.bbox else [0.1, 0.1, 0.1]
                }
                for name, state in self.objects.items()
            ]
            visualize_scene_plan(initialized_objects, "initialized_scene.png")
            logger.info("\nGenerated initialized scene visualization: initialized_scene.png")
            
            # Optimize object selection
            logger.info("\n=== 3. Optimizing Object Selection ===")
            try:
                selection_changes = self._optimize_object_selection(hierarchy)
                if selection_changes:
                    logger.info("Made changes to object selection")
            except Exception as e:
                logger.error(f"Error optimizing selection: {str(e)}")
                selection_changes = False
            
            # Optimize object dimensions
            logger.info("\n=== 4. Optimizing Object Dimensions ===")
            try:
                dimension_changes = self._optimize_object_dimensions(hierarchy)
                if dimension_changes:
                    logger.info("Made changes to object dimensions")
            except Exception as e:
                logger.error(f"Error optimizing dimensions: {str(e)}")
                dimension_changes = False
            
            if selection_changes or dimension_changes:
                logger.info("\n=== 5. Reinitializing Placement Grids ===")
                try:
                    self._reinitialize_placement_grids()
                except Exception as e:
                    logger.error(f"Error reinitializing grids: {str(e)}")
                
                # Generate visualization after optimization
                optimized_objects = [
                    {
                        'name': name,
                        'position': state.position if state.position else [0, 0, 0],
                        'dimensions': state.bbox if state.bbox else [0.1, 0.1, 0.1]
                    }
                    for name, state in self.objects.items()
                ]
                visualize_scene_plan(optimized_objects, "optimized_scene.png")
                logger.info("\nGenerated optimized scene visualization: optimized_scene.png")
            
            # Calculate placement order
            logger.info("\n=== 6. Calculating Placement Order ===")
            try:
                self._calculate_placement_order(hierarchy)
                logger.info("\nFinal placement order:")
                for i, obj_name in enumerate(self.placement_order, 1):
                    obj_state = self.objects[obj_name]
                    logger.info(f"\n{i}. 📦 {obj_name}")
                    logger.info(f"   Dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
                    if obj_state.position:
                        logger.info(f"   Position (x,y,z): {[f'{x:.3f}' for x in obj_state.position]}")
                    logger.info(f"   Parent: {obj_state.parent if obj_state.parent else 'None'}")
                    logger.info(f"   Children: {sorted(obj_state.children_on) if obj_state.children_on else 'None'}")
            except Exception as e:
                logger.error(f"Error calculating placement order: {str(e)}")
                # Use simple order if calculation fails
                self.placement_order = list(self.objects.keys())
            
            # Optimize surface allocation and determine positions
            logger.info("\n=== 7. Optimizing Surface Allocation ===")
            try:
                self._optimize_surface_allocation()
            except Exception as e:
                logger.error(f"Error optimizing surface allocation: {str(e)}")
                # Assign default positions if optimization fails
                self._assign_default_positions()
            
            # Generate final visualization
            final_objects = [
                {
                    'name': name,
                    'position': state.position if state.position else [0, 0, 0],
                    'dimensions': state.bbox if state.bbox else [0.1, 0.1, 0.1]
                }
                for name, state in self.objects.items()
            ]
            visualize_scene_plan(final_objects, "final_scene.png")
            logger.info("\nGenerated final scene visualization: final_scene.png")
            
            # Log final configuration
            logger.info("\n=== Final Object Configurations ===")
            for obj_name, obj_state in self.objects.items():
                logger.info(f"\n📦 {obj_name}:")
                logger.info(f"   Final dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
                if obj_state.position:
                    logger.info(f"   Final position (x,y,z): {[f'{x:.3f}' for x in obj_state.position]}")
                else:
                    logger.warning(f"   ⚠️ Position not assigned")
                if hasattr(obj_state, 'urdf_path'):
                    logger.info(f"   URDF: {obj_state.urdf_path}")
                if obj_state.parent:
                    logger.info(f"   Parent: {obj_state.parent}")
                if obj_state.children_on:
                    logger.info(f"   Children: {sorted(obj_state.children_on)}")
            
            self._log_placement_stats()
            logger.info(f"\n{'='*80}\n")
            
            return self.placement_order, self.objects
            
        except Exception as e:
            logger.error(f"Placement planning encountered an error: {str(e)}")
            logger.error(f"Returning partial results")
            return self.placement_order, self.objects
    
    def _log_hierarchy_stats(self, hierarchy: Dict[str, Dict[str, Set[str]]]):
        """Log detailed statistics about the object hierarchy."""
        stats = self.placement_stats["hierarchy_stats"]
        stats["total_objects"] = len(hierarchy) - 1  # Excluding ground
        stats["root_objects"] = len(hierarchy["ground"]["children"])
        stats["leaf_objects"] = sum(1 for obj in hierarchy if not hierarchy[obj]["children"])
        stats["floating_objects"] = sum(1 for obj in hierarchy if not hierarchy[obj]["parents"] and obj != "ground")
        
        logger.info("\n=== Scene Graph Structure Analysis ===")
        logger.info(f"Total objects: {stats['total_objects']}")
        logger.info(f"Root objects: {stats['root_objects']}")
        logger.info(f"Leaf objects: {stats['leaf_objects']}")
        logger.info(f"Floating objects: {stats['floating_objects']}")
        
        # Log detailed hierarchy
        logger.info("\nHierarchy Tree:")
        def print_hierarchy(node: str, level: int = 0):
            indent = "  " * level
            if node == "ground":
                logger.info(f"{indent}📍 {node}")
            else:
                obj_state = self.objects.get(node)
                dims = [f"{x:.3f}" for x in obj_state.bbox] if obj_state and obj_state.bbox else ["?", "?", "?"]
                pos = [f"{x:.3f}" for x in obj_state.position] if obj_state and obj_state.position else ["?", "?", "?"]
                logger.info(f"{indent}📦 {node}")
                logger.info(f"{indent}   Dimensions (w,l,h): {dims}")
                logger.info(f"{indent}   Position (x,y,z): {pos}")
            
            children = sorted(hierarchy[node]["children"])
            if children:
                logger.info(f"{indent}   Children: {children}")
                for child in children:
                    print_hierarchy(child, level + 1)
        
        # Print from root
        print_hierarchy("ground")
        
        # Log spatial relationships
        logger.info("\nSpatial Relationships:")
        for obj_name, obj_state in self.objects.items():
            if hasattr(obj_state, 'spatial_data') and obj_state.spatial_data:
                relations = obj_state.spatial_data.get('relations', [])
                if relations:
                    logger.info(f"\n📦 {obj_name} relationships:")
                    for rel in relations:
                        logger.info(f"   • {rel.source} {rel.relation_type} {rel.target}")
                        if hasattr(rel, 'distance') and rel.distance is not None:
                            logger.info(f"     Distance: {rel.distance:.3f}m")
                        if hasattr(rel, 'axis') and rel.axis is not None:
                            logger.info(f"     Axis: {rel.axis}")
    
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
        
        # Find furniture pieces (degree analysis and name matching)
        furniture = {node for node in G.nodes() 
                    if self._is_furniture(node) or 
                    (G.in_degree(node) == 0 and G.out_degree(node) > 0)}
        
        # Connect furniture to ground
        for node in furniture:
            if not hierarchy[node]["parents"]:
                hierarchy["ground"]["children"].add(node)
                hierarchy[node]["parents"].add("ground")
        
        # Verify hierarchy is complete
        orphans = [node for node in G.nodes() 
                  if node != "ground" and not hierarchy[node]["parents"]]
        
        if orphans:
            logger.info(f"Found orphaned objects: {orphans}")
            # Try to find best parent based on spatial relationships
            for orphan in orphans:
                best_parent = self._find_best_parent(orphan, G, hierarchy)
                if best_parent:
                    hierarchy[best_parent]["children"].add(orphan)
                    hierarchy[orphan]["parents"].add(best_parent)
        
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
        """Optimize surface allocation for all objects while preserving semantic relationships."""
        logger.info("=== Starting Surface Allocation ===")
        
        # Track placement statistics
        placement_stats = {
            "total_objects": len(self.placement_order),
            "placed_objects": 0,
            "failed_placements": 0,
            "rotated_placements": 0,
            "surface_utilization": {},
            "semantic_relations_preserved": 0,
            "semantic_relations_total": 0
        }
        
        # First, build a semantic relationship map for quick lookup
        semantic_relations = defaultdict(list)
        for obj_name in self.placement_order:
            obj_state = self.objects[obj_name]
            if hasattr(obj_state, 'spatial_data') and obj_state.spatial_data:
                for relation in obj_state.spatial_data.get('relations', []):
                    semantic_relations[obj_name].append(relation)
                    placement_stats["semantic_relations_total"] += 1
        
        # Process objects in hierarchy order
        for obj_name in self.placement_order:
            obj_state = self.objects[obj_name]
            if not obj_state.parent:  # Root object
                # Place at origin
                obj_state.position = (0.0, 0.0, obj_state.bbox[2]/2)
                placement_stats["placed_objects"] += 1
                continue
                
            parent_state = self.objects[obj_state.parent]
            
            # Sort children by size and semantic importance
            children = [(name, self.objects[name]) for name in (obj_state.children_on or [])]
            children.sort(key=lambda x: (
                len(semantic_relations[x[0]]),
                x[1].bbox[0] * x[1].bbox[1]
            ), reverse=True)
            
            # Try to place each child
            successful_placements = []
            failed_placements = []
            
            for child_name, child_state in children:
                # Get semantic relations for this child
                child_relations = semantic_relations[child_name]
                placed_relatives = [rel.target for rel in child_relations 
                                  if rel.target in self.objects and 
                                  self.objects[rel.target].position is not None]
                
                # Find stable position considering semantic relations
                position = self._find_stable_position(
                    parent_state, 
                    child_state,
                    semantic_relations=child_relations,
                    placed_relatives=placed_relatives
                )
                
                if position:
                    # Update object state and occupancy grid
                    child_state.position = position
                    self._update_occupancy_grid(parent_state, child_state)
                    successful_placements.append(child_name)
                    placement_stats["placed_objects"] += 1
                    
                    # Check if semantic relations were preserved
                    for rel in child_relations:
                        if rel.target in self.objects and self.objects[rel.target].position is not None:
                            if self._verify_semantic_relation(rel, child_state, self.objects[rel.target]):
                                placement_stats["semantic_relations_preserved"] += 1
                else:
                    # Try alternative placements while preserving semantic relations
                    success = self._try_alternative_placements(
                        parent_state,
                        child_state,
                        child_relations,
                        placed_relatives
                    )
                    
                    if success:
                        successful_placements.append(child_name)
                        placement_stats["placed_objects"] += 1
                        placement_stats["rotated_placements"] += 1
                    else:
                        failed_placements.append(child_name)
                        placement_stats["failed_placements"] += 1
            
            # Calculate surface utilization
            if children:
                placed_area = sum(
                    self.objects[name].bbox[0] * self.objects[name].bbox[1]
                    for name in successful_placements
                )
                parent_area = parent_state.bbox[0] * parent_state.bbox[1]
                utilization = (placed_area / parent_area) * 100 if parent_area > 0 else 0
                placement_stats["surface_utilization"][obj_name] = utilization
        
        return placement_stats["semantic_relations_preserved"] == placement_stats["semantic_relations_total"]
    
    def _calculate_z_position(self, parent: ObjectState, child: ObjectState, relation_type: str = "on") -> float:
        """Calculate proper z-position for an object based on parent and relationship type."""
        if not parent.position:
            # If parent has no position (e.g., ground), place at object's half-height
            return child.bbox[2]/2
            
        # Base z calculation starts from parent's top surface
        base_z = parent.position[2] + parent.bbox[2]/2
        
        # Add small offset for physics stability
        physics_offset = 0.001  # 1mm
        
        if relation_type == "on":
            # Place directly on top of parent
            return base_z + child.bbox[2]/2 + physics_offset
            
        elif relation_type == "above":
            # Place with some clearance above parent
            clearance = 0.05  # 5cm clearance for "above" relationship
            return base_z + clearance + child.bbox[2]/2 + physics_offset
            
        elif relation_type == "aligned_with":
            # Try to align centers vertically if possible
            return parent.position[2]
            
        # Default to placing on top
        return base_z + child.bbox[2]/2 + physics_offset

    def _find_stable_position(self, parent: ObjectState, child: ObjectState,
                              semantic_relations: List[SemanticRelation] = None,
                              placed_relatives: List[str] = None) -> Optional[Tuple[float, float, float]]:
        """Find a stable position for child object on parent's surface considering semantic relations."""
        try:
            # Determine primary relationship type for z-position calculation
            relation_type = "on"  # default
            if semantic_relations:
                for rel in semantic_relations:
                    if rel.target == parent.name:
                        relation_type = rel.relation_type
                        break
            
            # Calculate proper z-position first
            z_pos = self._calculate_z_position(parent, child, relation_type)
            
            # Validate size compatibility
            if child.bbox[0] > parent.bbox[0] or child.bbox[1] > parent.bbox[1]:
                return None
            
            # Initialize parent's grid if not already done
            if not parent.available_surface or not parent.grid_resolution:
                try:
                    parent.init_grid()
                except Exception as e:
                    logger.error(f"Error initializing grid: {str(e)}")
                    return None
            
            grid_size = len(parent.available_surface)
            child_w = max(1, int(child.bbox[0] / parent.grid_resolution))
            child_l = max(1, int(child.bbox[1] / parent.grid_resolution))
            
            if child_w > grid_size or child_l > grid_size:
                return None
            
            # Define search patterns based on semantic relations
            if semantic_relations and placed_relatives:
                # Calculate preferred position based on semantic relations
                preferred_pos = self._calculate_preferred_position(
                    parent, child, semantic_relations, placed_relatives
                )
                if preferred_pos:
                    grid_x = int((preferred_pos[0] - parent.position[0] + parent.bbox[0]/2) / parent.grid_resolution)
                    grid_y = int((preferred_pos[1] - parent.position[1] + parent.bbox[1]/2) / parent.grid_resolution)
                    
                    # Try positions around preferred position first
                    radius = 1
                    max_radius = max(grid_size - child_w, grid_size - child_l)
                    
                    while radius <= max_radius:
                        # Try positions in expanding square around preferred position
                        for i in range(-radius, radius + 1):
                            for j in range(-radius, radius + 1):
                                gi = grid_x + i
                                gj = grid_y + j
                                
                                if (0 <= gi < grid_size - child_w + 1 and 
                                    0 <= gj < grid_size - child_l + 1):
                                    if all(parent.available_surface[gi+di][gj+dj] 
                                          for di in range(child_w) 
                                          for dj in range(child_l)):
                                        # Convert grid position to world coordinates
                                        world_pos = (
                                            parent.position[0] + (gi * parent.grid_resolution) - (parent.bbox[0] / 2),
                                            parent.position[1] + (gj * parent.grid_resolution) - (parent.bbox[1] / 2),
                                            z_pos
                                        )
                                        
                                        # Verify semantic relations
                                        child.position = world_pos
                                        relations_satisfied = True
                                        
                                        for rel in semantic_relations:
                                            if rel.target in self.objects and self.objects[rel.target].position:
                                                if not self._verify_semantic_relation(rel, child, self.objects[rel.target]):
                                                    relations_satisfied = False
                                                    break
                                        
                                        if relations_satisfied:
                                            return world_pos
                                        
                                        child.position = None
                        radius += 1
            
            # If no semantic relations or couldn't find position satisfying them,
            # fall back to regular grid search
            center_i = (grid_size - child_w) // 2
            center_j = (grid_size - child_l) // 2
            
            # Try center position first
            if all(parent.available_surface[center_i+di][center_j+dj] 
                  for di in range(child_w) 
                  for dj in range(child_l)):
                return (
                    parent.position[0] + (center_i * parent.grid_resolution) - (parent.bbox[0] / 2),
                    parent.position[1] + (center_j * parent.grid_resolution) - (parent.bbox[1] / 2),
                    z_pos
                )
            
            # Spiral out from center
            radius = 1
            while radius <= max(grid_size - child_w, grid_size - child_l):
                # Try positions in a square pattern around center
                for i in range(-radius, radius+1):
                    for j in [-radius, radius]:  # Top and bottom edges
                        gi = center_i + i
                        gj = center_j + j
                        
                        if (0 <= gi < grid_size - child_w + 1 and 
                            0 <= gj < grid_size - child_l + 1):
                            if all(parent.available_surface[gi+di][gj+dj] 
                                  for di in range(child_w) 
                                  for dj in range(child_l)):
                                return (
                                    parent.position[0] + (gi * parent.grid_resolution) - (parent.bbox[0] / 2),
                                    parent.position[1] + (gj * parent.grid_resolution) - (parent.bbox[1] / 2),
                                    z_pos
                                )
                
                # Try left and right edges
                for i in [-radius, radius]:
                    for j in range(-radius+1, radius):
                        gi = center_i + i
                        gj = center_j + j
                        
                        if (0 <= gi < grid_size - child_w + 1 and 
                            0 <= gj < grid_size - child_l + 1):
                            if all(parent.available_surface[gi+di][gj+dj] 
                                  for di in range(child_w) 
                                  for dj in range(child_l)):
                                return (
                                    parent.position[0] + (gi * parent.grid_resolution) - (parent.bbox[0] / 2),
                                    parent.position[1] + (gj * parent.grid_resolution) - (parent.bbox[1] / 2),
                                    z_pos
                                )
                
                radius += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding stable position: {str(e)}")
            return None

    def _calculate_preferred_position(self, parent: ObjectState, child: ObjectState,
                                    semantic_relations: List[SemanticRelation],
                                    placed_relatives: List[str]) -> Optional[Tuple[float, float, float]]:
        """Calculate preferred position based on semantic relations with placed objects."""
        if not placed_relatives:
            return None
            
        # Get positions of all placed relatives
        relative_positions = []
        for rel_name in placed_relatives:
            if rel_name in self.objects and self.objects[rel_name].position:
                relative_positions.append(self.objects[rel_name].position)
        
        if not relative_positions:
            return None
        
        # Calculate average position of relatives
        avg_x = sum(pos[0] for pos in relative_positions) / len(relative_positions)
        avg_y = sum(pos[1] for pos in relative_positions) / len(relative_positions)
        
        # Adjust based on relation types
        x_offset = 0
        y_offset = 0
        
        for rel in semantic_relations:
            if rel.target in self.objects and self.objects[rel.target].position:
                target = self.objects[rel.target]
                
                if rel.relation_type == "next_to":
                    # Place beside the target
                    x_offset = (child.bbox[0] + target.bbox[0]) / 2
                    
                elif rel.relation_type == "in_front_of":
                    # Place in front of target
                    y_offset = (child.bbox[1] + target.bbox[1]) / 2
                    
                elif rel.relation_type == "behind":
                    # Place behind target
                    y_offset = -(child.bbox[1] + target.bbox[1]) / 2
                    
                elif rel.relation_type == "aligned_with":
                    # Try to align with target
                    x_offset = 0
                    y_offset = 0
        
        # Return preferred position
        return (avg_x + x_offset, avg_y + y_offset, parent.position[2] + parent.bbox[2]/2 + child.bbox[2]/2)

    def _verify_semantic_relation(self, relation: SemanticRelation, obj1: ObjectState, obj2: ObjectState) -> bool:
        """Verify if a semantic relation is satisfied by the current object positions."""
        if not obj1.position or not obj2.position:
            return False
            
        # Calculate distances and relative positions
        dx = obj2.position[0] - obj1.position[0]
        dy = obj2.position[1] - obj1.position[1]
        dz = obj2.position[2] - obj1.position[2]
        
        # Distance between centers in XY plane
        distance_xy = np.sqrt(dx*dx + dy*dy)
        
        # Combined dimensions
        combined_width = (obj1.bbox[0] + obj2.bbox[0]) / 2
        combined_length = (obj1.bbox[1] + obj2.bbox[1]) / 2
        combined_height = (obj1.bbox[2] + obj2.bbox[2]) / 2
        
        # Check different relation types with proper z-index consideration
        if relation.relation_type == "on":
            # One object should be directly on top of the other
            return (abs(dx) < combined_width * 0.5 and 
                   abs(dy) < combined_length * 0.5 and 
                   abs(dz - (obj2.bbox[2] + obj1.bbox[2])/2) < 0.01)  # 1cm tolerance
            
        elif relation.relation_type == "above":
            # Object should be above with some clearance
            min_clearance = 0.05  # 5cm minimum clearance
            return (abs(dx) < combined_width * 0.5 and 
                   abs(dy) < combined_length * 0.5 and 
                   dz > min_clearance)
            
        elif relation.relation_type == "next_to":
            # Objects should be at similar height and adjacent
            return (distance_xy < combined_width * 1.5 and 
                   distance_xy > combined_width * 0.5 and
                   abs(dz) < combined_height * 0.3)  # Allow some height difference
            
        elif relation.relation_type == "aligned_with":
            # Objects should be aligned along one axis and at similar heights
            return ((abs(dx) < combined_width * 0.2 or abs(dy) < combined_length * 0.2) and
                   abs(dz) < combined_height * 0.3)
            
        elif relation.relation_type == "in_front_of":
            # One object should be in front and at similar height
            return (abs(dy) > combined_length * 0.5 and 
                   abs(dx) < combined_width * 0.5 and
                   abs(dz) < combined_height * 0.3)
            
        return False

    def _apply_global_scaling(self, urdf_path: str) -> str:
        """Apply global scaling to a URDF file and return path to scaled version.
        
        Args:
            urdf_path: Path to original URDF file
            
        Returns:
            str: Path to scaled URDF file
        """
        try:
            # Convert to Path object for robust path handling
            urdf_path = Path(urdf_path)
            
            # Verify the file exists and is a URDF
            if not urdf_path.exists():
                logger.error(f"URDF file not found: {urdf_path}")
                return str(urdf_path)
                
            if urdf_path.name != "mobility.urdf":
                logger.warning(f"Unexpected URDF filename: {urdf_path.name}, expected 'mobility.urdf'")
            
            # Create scaled URDF path in the same directory
            scaled_path = urdf_path.parent / "mobility_scaled.urdf"
            
            # Check if scaled version already exists
            if scaled_path.exists():
                logger.info(f"Using existing scaled URDF: {scaled_path}")
                return str(scaled_path)
            
            # Parse and modify URDF
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # Track if we made any changes
            changes_made = False
            
            # Update all mesh scales
            for mesh in root.findall(".//mesh"):
                current_scale = mesh.get("scale")
                if current_scale:
                    # Parse existing scale
                    try:
                        sx, sy, sz = map(float, current_scale.split())
                        # Apply additional global scaling
                        new_scale = f"{sx * self.global_scale} {sy * self.global_scale} {sz * self.global_scale}"
                    except ValueError:
                        # If parsing fails, use default global scale
                        new_scale = f"{self.global_scale} {self.global_scale} {self.global_scale}"
                else:
                    # Set new global scale
                    new_scale = f"{self.global_scale} {self.global_scale} {self.global_scale}"
                mesh.set("scale", new_scale)
                changes_made = True
                
                # Also verify mesh file exists
                mesh_file = mesh.get("filename")
                if mesh_file:
                    mesh_path = urdf_path.parent / mesh_file
                    if not mesh_path.exists():
                        logger.warning(f"Mesh file not found: {mesh_path}")
            
            # Update all origins to account for scaling
            for origin in root.findall(".//origin"):
                if "xyz" in origin.attrib:
                    try:
                        x, y, z = map(float, origin.get("xyz").split())
                        new_xyz = f"{x * self.global_scale} {y * self.global_scale} {z * self.global_scale}"
                        origin.set("xyz", new_xyz)
                        changes_made = True
                    except ValueError:
                        logger.warning(f"Failed to parse origin xyz: {origin.get('xyz')}")
            
            if changes_made:
                # Create backup of original file
                backup_path = urdf_path.parent / "mobility.urdf.backup"
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(urdf_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                
                # Save scaled version
                tree.write(scaled_path, xml_declaration=True, encoding='utf-8')
                logger.info(f"Created scaled URDF: {scaled_path}")
                logger.info(f"Applied global scale: {self.global_scale}")
                return str(scaled_path)
            else:
                logger.info(f"No scaling needed for: {urdf_path}")
                return str(urdf_path)
                
        except Exception as e:
            logger.error(f"Error applying global scaling to {urdf_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return str(urdf_path)

    def _initialize_object_states(self, hierarchy: Dict[str, Dict[str, Set[str]]], object_states: Optional[Dict[str, ObjectState]] = None):
        """Initialize object states based on hierarchy and provided states."""
        # Initialize or update object states
        for obj_name in hierarchy:
            if obj_name == "ground":
                continue
                
            # Create or update object state
            if obj_name not in self.objects:
                if object_states and obj_name in object_states:
                    obj_state = object_states[obj_name]
                    
                    # Apply global scaling to URDF if available
                    if hasattr(obj_state, 'urdf_path') and obj_state.urdf_path:
                        obj_state.urdf_path = self._apply_global_scaling(obj_state.urdf_path)
                        
                        # Scale bounding box dimensions
                        if obj_state.bbox:
                            obj_state.bbox = tuple(dim * self.global_scale for dim in obj_state.bbox)
                            logger.info(f"Scaled {obj_name} dimensions to: {[f'{x:.3f}' for x in obj_state.bbox]}")
                    
                    self.objects[obj_name] = obj_state
                else:
                    logger.warning(f"No state information provided for {obj_name}")
                    continue
            
            # Set parent relationship
            parent = next(iter(hierarchy[obj_name]["parents"]), None)
            if parent and parent != "ground":
                self.objects[obj_name].parent = parent
                
            # Set children relationships
            children = hierarchy[obj_name]["children"]
            if children:
                self.objects[obj_name].children_on = children
            
            # Initialize placement grid if needed
            if not self.objects[obj_name].grid_resolution:
                try:
                    self.objects[obj_name].init_grid(resolution=self.default_grid_resolution)
                except Exception as e:
                    logger.error(f"Failed to initialize grid for {obj_name}: {str(e)}")
                
        return self.objects

    def _optimize_object_selection(self, hierarchy: Dict[str, Dict[str, Set[str]]]) -> bool:
        """Optimize object selection based on compatibility and constraints."""
        logger.info("\n=== Object Selection Analysis ===")
        changes_made = False
        
        try:
            for obj_name in hierarchy:
                if obj_name == "ground":
                    continue
                    
                if obj_name not in self.objects:
                    continue
                    
                obj_state = self.objects[obj_name]
                logger.info(f"\n📦 Analyzing {obj_name}:")
                logger.info(f"   Original dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
                if hasattr(obj_state, 'urdf_path'):
                    logger.info(f"   URDF path: {obj_state.urdf_path}")
                
                # Check if object dimensions are compatible with parent
                parent = next(iter(hierarchy[obj_name]["parents"]), None)
                if parent and parent != "ground" and parent in self.objects:
                    parent_state = self.objects[parent]
                    logger.info(f"   Parent: {parent}")
                    logger.info(f"   Parent dimensions (w,l,h): {[f'{x:.3f}' for x in parent_state.bbox]}")
                    
                    # Check if object is too large for parent
                    if (obj_state.bbox[0] > parent_state.bbox[0] or 
                        obj_state.bbox[1] > parent_state.bbox[1]):
                        logger.warning(f"   ⚠️ Object {obj_name} is too large for parent {parent}")
                        logger.warning(f"      Object needs: {obj_state.bbox[0]:.3f}m × {obj_state.bbox[1]:.3f}m")
                        logger.warning(f"      Parent has: {parent_state.bbox[0]:.3f}m × {parent_state.bbox[1]:.3f}m")
                        # Future: Implement object replacement logic here
                        
            return changes_made
            
        except Exception as e:
            logger.error(f"Error in object selection optimization: {str(e)}")
            return False

    def _optimize_object_dimensions(self, hierarchy: Dict[str, Dict[str, Set[str]]]) -> bool:
        """Optimize object dimensions based on scene constraints."""
        logger.info("\n=== Object Dimension Optimization ===")
        changes_made = False
        
        try:
            for obj_name in hierarchy:
                if obj_name == "ground":
                    continue
                    
                if obj_name not in self.objects:
                    continue
                    
                obj_state = self.objects[obj_name]
                logger.info(f"\n📦 Analyzing {obj_name}:")
                logger.info(f"   Original dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
                
                # Check for unrealistic dimensions
                if any(dim <= 0 or dim > 5.0 for dim in obj_state.bbox):  # 5m max dimension
                    logger.warning(f"   ⚠️ Object {obj_name} has unrealistic dimensions")
                    # Scale down if too large
                    max_dim = max(obj_state.bbox)
                    if max_dim > 5.0:
                        scale_factor = 5.0 / max_dim
                        original_dims = obj_state.bbox
                        obj_state.bbox = tuple(dim * scale_factor for dim in obj_state.bbox)
                        changes_made = True
                        logger.info(f"   📏 Scaled down dimensions:")
                        logger.info(f"      From: {[f'{x:.3f}' for x in original_dims]}")
                        logger.info(f"      To:   {[f'{x:.3f}' for x in obj_state.bbox]}")
                        logger.info(f"      Scale factor: {scale_factor:.3f}")
                
                # Check parent-child size relationships
                parent = next(iter(hierarchy[obj_name]["parents"]), None)
                if parent and parent != "ground" and parent in self.objects:
                    parent_state = self.objects[parent]
                    logger.info(f"   Parent: {parent}")
                    logger.info(f"   Parent dimensions (w,l,h): {[f'{x:.3f}' for x in parent_state.bbox]}")
                    
                    # Ensure child isn't larger than parent in any dimension
                    needs_scaling = False
                    scale_factors = []
                    for i, (child_dim, parent_dim) in enumerate(zip(obj_state.bbox, parent_state.bbox)):
                        if child_dim > parent_dim:
                            needs_scaling = True
                            scale_factors.append(parent_dim / child_dim)
                        else:
                            scale_factors.append(1.0)
                    
                    if needs_scaling:
                        # Use minimum scale factor to maintain proportions
                        scale_factor = min(scale_factors)
                        original_dims = obj_state.bbox
                        obj_state.bbox = tuple(dim * scale_factor for dim in obj_state.bbox)
                        changes_made = True
                        logger.info(f"   📏 Scaled to fit parent:")
                        logger.info(f"      From: {[f'{x:.3f}' for x in original_dims]}")
                        logger.info(f"      To:   {[f'{x:.3f}' for x in obj_state.bbox]}")
                        logger.info(f"      Scale factor: {scale_factor:.3f}")
                    else:
                        logger.info("   ✅ Dimensions are compatible with parent")
            
            return changes_made
            
        except Exception as e:
            logger.error(f"Error in dimension optimization: {str(e)}")
            return False

    def _log_placement_stats(self):
        """Log detailed placement statistics."""
        logger.info("\nPlacement Statistics:")
        
        # Object counts
        total_objects = len(self.objects)
        placed_objects = sum(1 for obj in self.objects.values() if obj.position is not None)
        unplaced_objects = total_objects - placed_objects
        
        logger.info(f"Total Objects: {total_objects}")
        logger.info(f"Successfully Placed: {placed_objects}")
        logger.info(f"Failed to Place: {unplaced_objects}")
        
        # Hierarchy statistics
        root_objects = sum(1 for obj in self.objects.values() if not obj.parent)
        leaf_objects = sum(1 for obj in self.objects.values() if not obj.children_on)
        
        logger.info("\nHierarchy Statistics:")
        logger.info(f"Root Objects: {root_objects}")
        logger.info(f"Leaf Objects: {leaf_objects}")
        
        # Surface utilization
        if self.placement_stats["surface_utilization"]:
            logger.info("\nSurface Utilization:")
            for parent, utilization in self.placement_stats["surface_utilization"].items():
                logger.info(f"  {parent}: {utilization:.1f}%")
        
        # Placement failures
        if self.placement_stats["placement_failures"]:
            logger.info("\nPlacement Failures:")
            for obj, reasons in self.placement_stats["placement_failures"].items():
                logger.info(f"  {obj}: {', '.join(reasons)}")
        
        # Stability checks
        stable_count = sum(1 for stable in self.placement_stats["stability_checks"].values() if stable)
        logger.info(f"\nStability Checks Passed: {stable_count}/{total_objects}")
        
        # Collision counts
        if self.placement_stats["collision_counts"]:
            total_collisions = sum(self.placement_stats["collision_counts"].values())
            logger.info(f"Total Collision Resolutions: {total_collisions}")

    def _reinitialize_placement_grids(self):
        """Reinitialize placement grids for all objects."""
        logger.info("Reinitializing placement grids...")
        for obj_name, obj_state in self.objects.items():
            try:
                obj_state.init_grid(resolution=self.default_grid_resolution)
                logger.info(f"Reinitialized grid for {obj_name}")
            except Exception as e:
                logger.error(f"Failed to reinitialize grid for {obj_name}: {str(e)}")

    def _update_occupancy_grid(self, parent: ObjectState, child: ObjectState):
        """Update parent's occupancy grid with child's footprint."""
        if not parent.available_surface or not parent.grid_resolution:
            parent.init_grid(resolution=self.default_grid_resolution)
        
        if not child.position:
            return
            
        # Convert child position to grid coordinates
        grid_x = int((child.position[0] - parent.position[0] + parent.bbox[0]/2) / parent.grid_resolution)
        grid_y = int((child.position[1] - parent.position[1] + parent.bbox[1]/2) / parent.grid_resolution)
        
        # Calculate child footprint in grid cells
        child_w = max(1, int(child.bbox[0] / parent.grid_resolution))
        child_l = max(1, int(child.bbox[1] / parent.grid_resolution))
        
        # Mark grid cells as occupied
        grid_size = len(parent.available_surface)
        for i in range(max(0, grid_x), min(grid_size, grid_x + child_w)):
            for j in range(max(0, grid_y), min(grid_size, grid_y + child_l)):
                parent.available_surface[i][j] = False

    def _assign_default_positions(self):
        """Assign default positions to objects without positions."""
        logger.info("Assigning default positions to unplaced objects...")
        
        # Track used positions to avoid overlap
        used_positions = set()
        grid_size = 1.0  # 1m grid for default placement
        
        for obj_name in self.placement_order:
            obj_state = self.objects[obj_name]
            if obj_state.position is not None:
                used_positions.add((obj_state.position[0], obj_state.position[1]))
                continue
                
            # Start with default position
            x, y = 0.0, 0.0
            z = obj_state.bbox[2]/2  # Place at half height
            
            # If object has a parent, offset from parent's position
            if obj_state.parent and self.objects[obj_state.parent].position:
                parent = self.objects[obj_state.parent]
                x = parent.position[0]
                y = parent.position[1]
                z = parent.position[2] + parent.bbox[2]/2 + obj_state.bbox[2]/2
            
            # Find unused position
            while (x, y) in used_positions:
                x += grid_size
                if x > 5.0:  # Max 5m in x direction
                    x = 0.0
                    y += grid_size
            
            obj_state.position = (x, y, z)
            used_positions.add((x, y))
            logger.info(f"Assigned default position for {obj_name}: {[f'{p:.3f}' for p in obj_state.position]}")

    def _try_alternative_placements(self, parent: ObjectState, child: ObjectState,
                                  semantic_relations: List[SemanticRelation],
                                  placed_relatives: List[str]) -> bool:
        """Try alternative placements for an object when standard placement fails."""
        logger.info(f"Trying alternative placements for {child.name}...")
        
        # Try rotating the object
        original_bbox = child.bbox
        rotated_bbox = (original_bbox[1], original_bbox[0], original_bbox[2])  # Swap width and length
        child.bbox = rotated_bbox
        
        # Try placement with rotated dimensions
        position = self._find_stable_position(
            parent, child,
            semantic_relations=semantic_relations,
            placed_relatives=placed_relatives
        )
        
        if position:
            child.position = position
            self._update_occupancy_grid(parent, child)
            logger.info(f"Found stable position after rotation: {[f'{x:.3f}' for x in position]}")
            return True
            
        # Restore original dimensions if rotation didn't help
        child.bbox = original_bbox
        
        # Try different z-levels if applicable
        if semantic_relations:
            for rel in semantic_relations:
                if rel.relation_type == "above" and rel.target in self.objects:
                    target = self.objects[rel.target]
                    if target.position:
                        # Try placing at different heights above target
                        clearances = [0.05, 0.1, 0.2]  # Try 5cm, 10cm, 20cm clearances
                        for clearance in clearances:
                            z = target.position[2] + target.bbox[2]/2 + clearance + child.bbox[2]/2
                            test_position = (target.position[0], target.position[1], z)
                            
                            # Verify position doesn't collide with other objects
                            if not self._check_collision(child, test_position):
                                child.position = test_position
                                self._update_occupancy_grid(parent, child)
                                logger.info(f"Found stable position at height {clearance}m: {[f'{x:.3f}' for x in test_position]}")
                                return True
        
        return False

    def _check_collision(self, obj: ObjectState, position: Tuple[float, float, float]) -> bool:
        """Check if an object at the given position would collide with any other objects."""
        for other_name, other_obj in self.objects.items():
            if other_name == obj.name or not other_obj.position:
                continue
                
            # Simple box collision check
            if (abs(position[0] - other_obj.position[0]) < (obj.bbox[0] + other_obj.bbox[0])/2 and
                abs(position[1] - other_obj.position[1]) < (obj.bbox[1] + other_obj.bbox[1])/2 and
                abs(position[2] - other_obj.position[2]) < (obj.bbox[2] + other_obj.bbox[2])/2):
                return True
        
        return False

    def plan_object_placement(self, objects, relations):
        """Plan object placement with enhanced logging and visualization."""
        # Log initial state
        self.logger.info("=== Initial Object State Analysis ===")
        for obj in objects:
            self.logger.info(f"\nAnalyzing {obj.name}:")
            self.logger.info(f"  Initial position: {obj.position if obj.position else 'Not set'}")
            self.logger.info(f"  Initial dimensions: {obj.dimensions}")
            self.logger.info(f"  Parent: {obj.parent.name if obj.parent else 'None'}")
            self.logger.info(f"  Children: {[child.name for child in obj.children]}")
            
            # Check for potential issues
            if obj.parent:
                parent_area = float(obj.parent.dimensions[0]) * float(obj.parent.dimensions[1])
                obj_area = float(obj.dimensions[0]) * float(obj.dimensions[1])
                if obj_area > parent_area:
                    self.logger.warning(f"  ⚠️ Object area ({obj_area:.2f}) exceeds parent area ({parent_area:.2f})")
                
                # Check center of mass
                if obj.position:
                    parent_bounds = {
                        'x': [-float(obj.parent.dimensions[0])/2, float(obj.parent.dimensions[0])/2],
                        'y': [-float(obj.parent.dimensions[1])/2, float(obj.parent.dimensions[1])/2],
                        'z': [0, float(obj.parent.dimensions[2])]
                    }
                    
                    pos = [float(p) for p in obj.position]
                    if (pos[0] < parent_bounds['x'][0] or pos[0] > parent_bounds['x'][1] or
                        pos[1] < parent_bounds['y'][0] or pos[1] > parent_bounds['y'][1] or
                        pos[2] < parent_bounds['z'][0] or pos[2] > parent_bounds['z'][1]):
                        self.logger.warning(f"  ⚠️ Object position {pos} outside parent bounds {parent_bounds}")
        
        # Generate visualization before placement
        pre_placement_objects = [
            {
                'name': obj.name,
                'position': obj.position if obj.position else ['0', '0', '0'],
                'dimensions': obj.dimensions
            }
            for obj in objects
        ]
        visualize_scene_plan(pre_placement_objects, "pre_placement.png")
        
        # ... rest of placement logic ...
        
        # Log placement attempts
        self.logger.info("\n=== Placement Attempts ===")
        for obj in objects:
            if not obj.position:
                self.logger.warning(f"\nFailed to place {obj.name}:")
                if obj.parent:
                    self.logger.warning(f"  Parent surface area: {float(obj.parent.dimensions[0])*float(obj.parent.dimensions[1]):.2f}")
                    self.logger.warning(f"  Required area: {float(obj.dimensions[0])*float(obj.dimensions[1]):.2f}")
                    self.logger.warning(f"  Available positions tried: {self.get_available_positions(obj)}")
                else:
                    self.logger.warning("  No parent object")
            else:
                self.logger.info(f"\nSuccessfully placed {obj.name}:")
                self.logger.info(f"  Final position: {obj.position}")
                self.logger.info(f"  Stability check: {self.check_stability(obj)}")
        
        # Generate visualization after placement
        post_placement_objects = [
            {
                'name': obj.name,
                'position': obj.position if obj.position else ['0', '0', '0'],
                'dimensions': obj.dimensions
            }
            for obj in objects
        ]
        visualize_scene_plan(post_placement_objects, "post_placement.png")
        
        return objects
        
    def get_available_positions(self, obj):
        """Get list of available positions that were considered."""
        # This is a placeholder - implement actual position tracking
        return ["Implement position tracking"]
        
    def check_stability(self, obj):
        """Check if object placement is stable."""
        if not obj.position or not obj.parent:
            return False
            
        # Convert positions and dimensions to float
        pos = [float(p) for p in obj.position]
        dims = [float(d) for d in obj.dimensions]
        parent_dims = [float(d) for d in obj.parent.dimensions]
        
        # Check if object's base is fully supported
        obj_min_x = pos[0] - dims[0]/2
        obj_max_x = pos[0] + dims[0]/2
        obj_min_y = pos[1] - dims[1]/2
        obj_max_y = pos[1] + dims[1]/2
        
        parent_min_x = -parent_dims[0]/2
        parent_max_x = parent_dims[0]/2
        parent_min_y = -parent_dims[1]/2
        parent_max_y = parent_dims[1]/2
        
        return (obj_min_x >= parent_min_x and obj_max_x <= parent_max_x and
                obj_min_y >= parent_min_y and obj_max_y <= parent_max_y)

    def quick_place_objects(self, relations: List[SpatialRelation], object_states: Dict[str, ObjectState]) -> Tuple[List[str], Dict[str, ObjectState]]:
        """Quick object placement based purely on spatial relations with error handling."""
        logger.info(f"\n{'='*80}")
        logger.info("Starting Quick Object Placement")
        logger.info(f"{'='*80}")
        logger.info(f"Input: {len(relations)} relations, {len(object_states)} objects")
        
        try:
            self.objects = object_states.copy()  # Make a copy to avoid modifying original
        except Exception as e:
            logger.error(f"Error copying object states: {e}")
            self.objects = object_states
        
        # 1. Process and log initial object states
        logger.info("\n=== 1. Initial Object States ===")
        for name, state in self.objects.items():
            try:
                logger.info(f"\n📦 {name}:")
                logger.info(f"   Original dimensions: {state.bbox if state.bbox else 'Not set'}")
                logger.info(f"   URDF: {state.urdf_path if hasattr(state, 'urdf_path') else 'Not set'}")
            except Exception as e:
                logger.error(f"Error logging object {name}: {e}")
        
        # 2. Update object dimensions using LLM with fallback
        logger.info("\n=== 2. Updating Object Dimensions ===")
        try:
            self._update_object_dimensions_llm()
        except Exception as e:
            logger.error(f"Error in LLM dimension update: {e}")
            self._apply_default_dimensions()
        
        # Validate dimensions after update
        self._validate_object_dimensions()
        
        # 3. Process spatial relations to determine positions
        logger.info("\n=== 3. Processing Spatial Relations ===")
        relation_map = defaultdict(list)
        try:
            for rel in relations:
                relation_map[rel.source].append(rel)
                logger.info(f"Relation: {rel.source} {rel.relation_type} {rel.target}")
        except Exception as e:
            logger.error(f"Error processing relations: {e}")
        
        # 4. Place objects based on relations
        logger.info("\n=== 4. Placing Objects ===")
        placed_objects = set()
        placement_attempts = 0
        max_attempts = len(self.objects) * 2  # Prevent infinite loops
        
        try:
            # First place root objects with spacing
            root_objects = self._find_root_objects(relations)
            logger.info(f"\nRoot objects: {root_objects}")
            
            spacing = 1.0  # 1 meter spacing between root objects
            current_x = 0.0
            
            for obj_name in root_objects:
                if obj_name in self.objects:
                    try:
                        obj = self.objects[obj_name]
                        # Place root objects along X axis with spacing
                        obj.position = (current_x, 0.0, obj.bbox[2]/2)
                        current_x += obj.bbox[0] + spacing
                        placed_objects.add(obj_name)
                        logger.info(f"Placed root object {obj_name} at {obj.position}")
                    except Exception as e:
                        logger.error(f"Error placing root object {obj_name}: {e}")
                        # Use default position
                        self.objects[obj_name].position = (current_x, 0.0, 0.5)
                        current_x += spacing
                        placed_objects.add(obj_name)
            
            # Then place remaining objects based on relations
            while len(placed_objects) < len(self.objects) and placement_attempts < max_attempts:
                placement_attempts += 1
                initial_placed_count = len(placed_objects)
                
                for obj_name, obj_state in self.objects.items():
                    if obj_name in placed_objects:
                        continue
                    
                    try:
                        # Check if we can place this object
                        obj_relations = relation_map[obj_name]
                        can_place = all(rel.target in placed_objects for rel in obj_relations)
                        
                        if can_place:
                            position = self._calculate_position_from_relations(obj_name, obj_state, obj_relations)
                            if position:
                                obj_state.position = position
                                placed_objects.add(obj_name)
                                logger.info(f"Placed object {obj_name} at {position}")
                            else:
                                logger.warning(f"Could not calculate position for {obj_name}")
                    except Exception as e:
                        logger.error(f"Error placing object {obj_name}: {e}")
                        # Use default position if calculation fails
                        obj_state.position = (len(placed_objects) * spacing, 0.0, 0.5)
                        placed_objects.add(obj_name)
                
                # Check if we made progress
                if len(placed_objects) == initial_placed_count:
                    logger.warning("No progress made in placement iteration")
                    # Place remaining objects with default positions
                    self._place_remaining_objects(placed_objects, spacing)
                    break
        
        except Exception as e:
            logger.error(f"Error in main placement loop: {e}")
            # Place any remaining objects
            self._place_remaining_objects(placed_objects, spacing=1.0)
        
        # 5. Generate visualization
        logger.info("\n=== 5. Generating Visualization ===")
        try:
            final_objects = [
                {
                    'name': name,
                    'position': [float(p) for p in state.position] if state.position else [0, 0, 0],
                    'dimensions': [float(d) for d in state.bbox] if state.bbox else [0.1, 0.1, 0.1]
                }
                for name, state in self.objects.items()
            ]
            visualize_scene_plan(final_objects, "quick_placement_scene.png")
            logger.info("\nGenerated scene visualization: quick_placement_scene.png")
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
        
        # 6. Log final configuration
        logger.info("\n=== Final Object Configurations ===")
        for obj_name, obj_state in self.objects.items():
            try:
                logger.info(f"\n📦 {obj_name}:")
                logger.info(f"   Final dimensions (w,l,h): {[f'{x:.3f}' for x in obj_state.bbox]}")
                logger.info(f"   Final position (x,y,z): {[f'{x:.3f}' for x in obj_state.position]}")
            except Exception as e:
                logger.error(f"Error logging final state for {obj_name}: {e}")
        
        logger.info(f"\n{'='*80}\n")
        return list(placed_objects), self.objects

    def _apply_default_dimensions(self):
        """Apply default dimensions if LLM update fails."""
        default_dims = {
            'Table': (1.2, 0.8, 0.75),    # Standard desk size
            'Monitor': (0.6, 0.1, 0.4),   # Standard monitor
            'Keyboard': (0.45, 0.15, 0.03), # Standard keyboard
            'Mouse': (0.12, 0.07, 0.04),   # Standard mouse
            'Chair': (0.6, 0.6, 1.0),      # Standard chair
            'default': (0.5, 0.5, 0.5)     # Default for unknown objects
        }
        
        for name, state in self.objects.items():
            try:
                obj_type = name.split('_')[0]
                state.bbox = default_dims.get(obj_type, default_dims['default'])
                logger.info(f"Set {name} dimensions to: {state.bbox}")
            except Exception as e:
                logger.error(f"Error setting dimensions for {name}: {e}")
                state.bbox = default_dims['default']

    def _validate_object_dimensions(self):
        """Validate and fix object dimensions if necessary."""
        min_size = 0.01  # 1cm minimum
        max_size = 5.0   # 5m maximum
        
        for name, state in self.objects.items():
            try:
                if not hasattr(state, 'bbox') or not state.bbox:
                    logger.warning(f"Missing dimensions for {name}, applying defaults")
                    state.bbox = (0.5, 0.5, 0.5)
                    continue
                
                # Ensure dimensions are within reasonable bounds
                new_bbox = []
                for dim in state.bbox:
                    try:
                        dim_float = float(dim)
                        if dim_float < min_size:
                            dim_float = min_size
                        elif dim_float > max_size:
                            dim_float = max_size
                        new_bbox.append(dim_float)
                    except (ValueError, TypeError):
                        new_bbox.append(0.5)  # Default if conversion fails
                
                state.bbox = tuple(new_bbox)
                logger.info(f"Validated dimensions for {name}: {state.bbox}")
            except Exception as e:
                logger.error(f"Error validating dimensions for {name}: {e}")
                state.bbox = (0.5, 0.5, 0.5)

    def _place_remaining_objects(self, placed_objects: Set[str], spacing: float = 1.0):
        """Place any remaining objects that couldn't be placed through relations."""
        current_x = len(placed_objects) * spacing
        
        for obj_name, obj_state in self.objects.items():
            if obj_name not in placed_objects:
                try:
                    obj_state.position = (current_x, 0.0, obj_state.bbox[2]/2)
                    current_x += obj_state.bbox[0] + spacing
                    placed_objects.add(obj_name)
                    logger.info(f"Placed remaining object {obj_name} at {obj_state.position}")
                except Exception as e:
                    logger.error(f"Error placing remaining object {obj_name}: {e}")
                    obj_state.position = (current_x, 0.0, 0.5)
                    current_x += spacing
                    placed_objects.add(obj_name)

    def _update_object_dimensions_llm(self):
        """Update object dimensions using LLM suggestions."""
        try:
            # Prepare object descriptions for LLM
            object_descriptions = []
            for name, state in self.objects.items():
                obj_type = name.split('_')[0]
                object_descriptions.append(f"{name} (type: {obj_type})")
            
            # Create LLM prompt
            prompt = f"""Given these objects: {', '.join(object_descriptions)}
            Suggest realistic dimensions (width, length, height in meters) for each object.
            Consider typical furniture and computer peripheral sizes.
            Format response as JSON with object names as keys and [width, length, height] as values."""
            
            # Get LLM response
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse dimensions from response
            try:
                dimensions = json.loads(response.choices[0].message.content)
                for obj_name, dims in dimensions.items():
                    if obj_name in self.objects:
                        self.objects[obj_name].bbox = tuple(dims)
                        logger.info(f"Updated {obj_name} dimensions to: {dims}")
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                
        except Exception as e:
            logger.error(f"Error updating dimensions with LLM: {str(e)}")

    def _find_root_objects(self, relations: List[SpatialRelation]) -> Set[str]:
        """Find objects that don't have any 'above' or 'on' relations to other objects."""
        dependent_objects = set()
        for rel in relations:
            if rel.relation_type in ['on', 'above']:
                dependent_objects.add(rel.source)
        
        return set(self.objects.keys()) - dependent_objects

    def _calculate_position_from_relations(self, obj_name: str, obj_state: ObjectState, 
                                        relations: List[SpatialRelation]) -> Optional[Tuple[float, float, float]]:
        """Calculate object position based on its spatial relations."""
        if not relations:
            return None
            
        # Get reference object (prefer 'on' relations)
        ref_relation = next((rel for rel in relations if rel.relation_type == 'on'), relations[0])
        ref_object = self.objects[ref_relation.target]
        
        # Base position calculations on relation type
        if ref_relation.relation_type == 'on':
            # Place directly on top of reference object
            x = ref_object.position[0]
            y = ref_object.position[1]
            z = ref_object.position[2] + ref_object.bbox[2]/2 + obj_state.bbox[2]/2
            
        elif ref_relation.relation_type == 'next_to':
            # Place beside reference object
            x = ref_object.position[0] + ref_object.bbox[0]/2 + obj_state.bbox[0]/2
            y = ref_object.position[1]
            z = ref_object.position[2]
            
        elif ref_relation.relation_type == 'in_front_of':
            # Place in front of reference object
            x = ref_object.position[0]
            y = ref_object.position[1] + ref_object.bbox[1]/2 + obj_state.bbox[1]/2
            z = ref_object.position[2]
            
        else:  # Default to placing near the reference object
            x = ref_object.position[0] + ref_object.bbox[0]
            y = ref_object.position[1]
            z = ref_object.position[2]
        
        return (x, y, z)

    def simple_scale_and_place(self, object_states: Dict[str, ObjectState]) -> Dict[str, ObjectState]:
        """Simply scale objects and place them at random positions."""
        logger.info(f"\n{'='*80}")
        logger.info("Starting Simple Scale and Place")
        logger.info(f"{'='*80}")
        logger.info(f"Processing {len(object_states)} objects")
        
        self.objects = object_states.copy()
        
        # 1. Apply default dimensions to all objects
        logger.info("\n=== 1. Setting Object Dimensions ===")
        default_dims = {
            'Table': (1.2, 0.8, 0.75),    # Standard desk size
            'Monitor': (0.6, 0.1, 0.4),   # Standard monitor
            'Keyboard': (0.45, 0.15, 0.03), # Standard keyboard
            'Mouse': (0.12, 0.07, 0.04),   # Standard mouse
            'Chair': (0.6, 0.6, 1.0),      # Standard chair
            'default': (0.5, 0.5, 0.5)     # Default for unknown objects
        }
        
        for name, state in self.objects.items():
            try:
                # Get object type from name
                obj_type = name.split('_')[0]
                # Apply dimensions
                state.bbox = default_dims.get(obj_type, default_dims['default'])
                logger.info(f"Set {name} dimensions to: {state.bbox}")
            except Exception as e:
                logger.error(f"Error setting dimensions for {name}: {e}")
                state.bbox = default_dims['default']
        
        # 2. Place objects in a random grid
        logger.info("\n=== 2. Placing Objects ===")
        grid_size = 3.0  # 3x3 meter area
        grid_cells = 3   # 3x3 grid
        cell_size = grid_size / grid_cells
        
        import random
        random.seed(42)  # For reproducibility
        
        used_positions = set()
        for name, state in self.objects.items():
            try:
                # Try to find an unused grid cell
                attempts = 0
                while attempts < 10:  # Maximum 10 attempts per object
                    # Generate random grid position
                    grid_x = random.randint(0, grid_cells-1)
                    grid_y = random.randint(0, grid_cells-1)
                    pos = (grid_x, grid_y)
                    
                    if pos not in used_positions:
                        used_positions.add(pos)
                        # Convert grid position to world coordinates
                        x = (grid_x - grid_cells/2) * cell_size
                        y = (grid_y - grid_cells/2) * cell_size
                        z = state.bbox[2]/2  # Place at half height
                        
                        state.position = (x, y, z)
                        logger.info(f"Placed {name} at grid {pos}, world position: {state.position}")
                        break
                    
                    attempts += 1
                
                if attempts == 10:
                    # If no free position found, place at a slightly offset position
                    x = random.uniform(-grid_size/2, grid_size/2)
                    y = random.uniform(-grid_size/2, grid_size/2)
                    z = state.bbox[2]/2
                    state.position = (x, y, z)
                    logger.info(f"Placed {name} at fallback position: {state.position}")
                
            except Exception as e:
                logger.error(f"Error placing {name}: {e}")
                # Use fallback position
                state.position = (0.0, 0.0, 0.5)
        
        # 3. Generate visualization
        logger.info("\n=== 3. Generating Visualization ===")
        try:
            final_objects = [
                {
                    'name': name,
                    'position': [float(p) for p in state.position] if state.position else [0, 0, 0],
                    'dimensions': [float(d) for d in state.bbox] if state.bbox else [0.1, 0.1, 0.1]
                }
                for name, state in self.objects.items()
            ]
            visualize_scene_plan(final_objects, "simple_placement_scene.png")
            logger.info("\nGenerated scene visualization: simple_placement_scene.png")
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
        
        logger.info(f"\n{'='*80}\n")
        return self.objects