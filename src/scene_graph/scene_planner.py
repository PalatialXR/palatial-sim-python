from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
from enum import Enum
import pybullet as p
from models.scene_graph import SpatialRelation, SemanticRelation, ObjectState
from models.scene_graph import ObjectNode, SubLinkage, ObjectState

class ScenePlanner:
    def __init__(self):
        self.objects: Dict[str, ObjectState] = {}
        self.placement_order: List[str] = []
        self.physics_client = None
    
    def init_physics(self, gui=True):
        """Initialize physics simulation with ground plane."""
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        self.ground_id = p.loadURDF("plane.urdf")
        p.setRealTimeSimulation(0)
        
    def precompute_placement(self, relations: List[SemanticRelation]):
        """Precompute optimal object placement considering surface area and stability."""
        # First pass: Build object hierarchy and load bounding boxes
        hierarchy = self._build_hierarchy(relations)
        
        # Second pass: Calculate placement order and initial positions
        self._calculate_placement_order(hierarchy)
        
        # Third pass: Optimize surface area usage
        self._optimize_surface_allocation()
        
        return self.placement_order, self.objects
    
    def _build_hierarchy(self, relations: List[SemanticRelation]) -> Dict:
        """Build object hierarchy starting from ground plane."""
        hierarchy = {"ground": {"children": set(), "parents": set()}}
        
        # First add all objects to ensure complete graph
        for rel in relations:
            if rel.source not in hierarchy:
                hierarchy[rel.source] = {"children": set(), "parents": set()}
            if rel.target not in hierarchy:
                hierarchy[rel.target] = {"children": set(), "parents": set()}
        
        # Then establish relationships
        for rel in relations:
            if rel.relation == SpatialRelation.ON:
                hierarchy[rel.target]["children"].add(rel.source)
                hierarchy[rel.source]["parents"].add(rel.target)
            elif rel.relation == SpatialRelation.BELOW:
                hierarchy[rel.source]["children"].add(rel.target)
                hierarchy[rel.target]["parents"].add(rel.source)
        
        # Connect floating objects to ground
        for obj in hierarchy:
            if obj != "ground" and not hierarchy[obj]["parents"]:
                hierarchy["ground"]["children"].add(obj)
                hierarchy[obj]["parents"].add("ground")
        
        return hierarchy
    
    def _calculate_placement_order(self, hierarchy: Dict):
        """Calculate optimal placement order ensuring stability."""
        visited = set()
        self.placement_order = []
        
        def visit(node: str):
            if node in visited:
                return
            visited.add(node)
            
            # First place parent objects
            for parent in hierarchy[node]["parents"]:
                if parent != "ground":
                    visit(parent)
                    
            self.placement_order.append(node)
            
            # Then place children objects
            for child in sorted(hierarchy[node]["children"]):
                visit(child)
        
        visit("ground")
        self.placement_order.remove("ground")  # Remove ground plane as it's already loaded
    
    def _optimize_surface_allocation(self):
        """Optimize placement of objects on surfaces using grid-based approach."""
        for obj_name in self.placement_order:
            obj_state = self.objects[obj_name]
            if not obj_state.children_on:
                continue
                
            # Sort children by size for better packing
            children = sorted(obj_state.children_on, 
                           key=lambda x: np.prod(self.objects[x].bbox),
                           reverse=True)
            
            # Find valid positions for each child using grid-based placement
            for child_name in children:
                child = self.objects[child_name]
                position = self._find_stable_position(obj_state, child)
                if position is not None:
                    child.position = position
                    self._update_occupancy_grid(obj_state, child)
    
    def _find_stable_position(self, parent: ObjectState, child: ObjectState) -> Optional[Tuple[float, float, float]]:
        """Find a stable position for child object on parent's surface."""
        child_w = int(child.bbox[0] / parent.grid_resolution)
        child_l = int(child.bbox[1] / parent.grid_resolution)
        
        # Find valid position in grid
        valid_pos = None
        for i in range(parent.available_surface.shape[0] - child_w):
            for j in range(parent.available_surface.shape[1] - child_l):
                if np.all(parent.available_surface[i:i+child_w, j:j+child_l]):
                    valid_pos = (i, j)
                    break
            if valid_pos is not None:
                break
                
        if valid_pos is None:
            return None
            
        # Convert grid position to world coordinates
        grid_x, grid_y = valid_pos
        world_x = parent.position[0] + (grid_x * parent.grid_resolution) - (parent.bbox[0] / 2)
        world_y = parent.position[1] + (grid_y * parent.grid_resolution) - (parent.bbox[1] / 2)
        world_z = parent.position[2] + parent.bbox[2]/2 + child.bbox[2]/2
        
        return (world_x, world_y, world_z)
    
    def _update_occupancy_grid(self, parent: ObjectState, child: ObjectState):
        """Update parent's occupancy grid after placing child object."""
        if child.position is None:
            return
            
        # Convert world position to grid coordinates
        grid_x = int((child.position[0] - parent.position[0] + parent.bbox[0]/2) / parent.grid_resolution)
        grid_y = int((child.position[1] - parent.position[1] + parent.bbox[1]/2) / parent.grid_resolution)
        
        # Mark occupied cells
        child_w = int(child.bbox[0] / parent.grid_resolution)
        child_l = int(child.bbox[1] / parent.grid_resolution)
        parent.available_surface[grid_x:grid_x+child_w, grid_y:grid_y+child_l] = False

def parse_llm_output(output_text: str) -> Tuple[List[str], List[SemanticRelation]]:
    """Parse LLM output into objects and relations.
    
    Args:
        output_text: String containing objects and relationships in format:
                    Objects:
                    [list of objects]
                    Relationships:
                    [source] | [relation] | [target] | [confidence]
                    
    Returns:
        Tuple of (list of object names, list of semantic relations)
        These form the nodes and edges of the scene graph, which will be used
        with URDF files to construct the full scene.
    """
    sections = output_text.split("Relationships:")
    objects_section = sections[0].split("Objects:")[1].strip()
    objects = [obj.strip() for obj in objects_section.split("\n") if obj.strip()]
    
    relations = []
    for line in sections[1].strip().split("\n"):
        if not line.strip():
            continue
        source, relation, target, confidence = [x.strip() for x in line.split("|")]
        relations.append(SemanticRelation(
            source=source,
            relation=SpatialRelation(relation),
            target=target,
            confidence=float(confidence)
        ))
    
    return objects, relations