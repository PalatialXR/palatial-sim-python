from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import pybullet as p
import numpy as np
import xml.etree.ElementTree as ET
import os

class SpatialRelation(Enum):
    ON = "on"
    NEXT_TO = "next_to"
    ABOVE = "above"
    BELOW = "below"

@dataclass
class SemanticRelation:
    source: str
    relation: SpatialRelation
    target: str
    confidence: float

class Scene3DGenerator:
    def __init__(self):
        self.physics_client = None
        self.object_bboxes = {}
        self.object_offsets = {}
        self.placed_objects = {}
        self.urdf_dir = os.path.join(os.path.dirname(__file__))
        
    def init_physics(self, gui=True):
        """Initialize physics simulation with real-time disabled for stable placement."""
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        # Disable physics during placement
        p.setRealTimeSimulation(0)
    
    def get_bbox_from_urdf(self, urdf_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get bounding box dimensions and origin offset from URDF file.
        
        Args:
            urdf_path: Path to the URDF file relative to urdf_dir
            
        Returns:
            Tuple of:
                - (width, length, height) in meters
                - (x_offset, y_offset, z_offset) for collision origin
            
        Raises:
            FileNotFoundError: If URDF file doesn't exist
            ValueError: If collision geometry is missing or invalid
        """
        if urdf_path in self.object_bboxes:
            print(f"Using cached bbox for {urdf_path}")
            return self.object_bboxes[urdf_path], self.object_offsets[urdf_path]
        
        print(f"\nLoading URDF: {urdf_path}")
        full_path = os.path.join(self.urdf_dir, urdf_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"URDF file not found: {full_path}")
            
        try:
            tree = ET.parse(full_path)
            collision = tree.find(".//collision")
            
            if collision is None:
                raise ValueError(f"No collision element found in {urdf_path}")
            
            box = collision.find("geometry/box")
            if box is None:
                raise ValueError(f"No box geometry found in {urdf_path}")
                
            size_str = box.get("size")
            if not size_str:
                raise ValueError(f"Box size not specified in {urdf_path}")
            
            # Get collision origin offset
            origin = collision.find("origin")
            offset = [0.0, 0.0, 0.0]  # Default if no origin specified
            if origin is not None:
                xyz = origin.get("xyz")
                if xyz:
                    offset = [float(x) for x in xyz.split()]
            
            print(f"Found collision box size: {size_str}")
            print(f"Found collision origin offset: {offset}")
            
            try:
                bbox = tuple(float(x) for x in size_str.split())
                offset = tuple(offset)
                
                if len(bbox) != 3:
                    raise ValueError(f"Expected 3 dimensions for box, got {len(bbox)}")
                if any(x <= 0 for x in bbox):
                    raise ValueError(f"All dimensions must be positive, got {bbox}")
                    
                print(f"Parsed bbox dimensions (w,l,h): {bbox}")
                print(f"Parsed origin offset (x,y,z): {offset}")
                
                self.object_bboxes[urdf_path] = bbox
                self.object_offsets[urdf_path] = offset
                return bbox, offset
                
            except ValueError as e:
                raise ValueError(f"Invalid dimensions in {urdf_path}: {str(e)}")
                
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse URDF file {urdf_path}: {str(e)}")
    
    def place_objects(self, relations: List[SemanticRelation]) -> Dict[str, Tuple[float, float, float]]:
        """Place objects in the scene based on their semantic relations."""
        scene_graph = self._build_graph(relations)
        root = self._find_base_object(scene_graph)
        object_positions = {}
        
        # First calculate all positions
        def calculate_positions(obj_name: str, parent_pos=None):
            print(f"\nCalculating position for: {obj_name}")
            urdf_path = f"{obj_name}.urdf"
            bbox, offset = self.get_bbox_from_urdf(urdf_path)
            
            if parent_pos is None:
                pos = self._sample_ground_position(bbox, offset)
                print(f"Root object position: {[f'{x:.3f}' for x in pos]}")
            else:
                parent_urdf = f"{scene_graph[obj_name]['parent']}.urdf"
                parent_bbox, parent_offset = self.object_bboxes[parent_urdf], self.object_offsets[parent_urdf]
                relation = scene_graph[obj_name]['relation']
                pos = self._sample_position_by_relation(relation, bbox, offset, parent_bbox, parent_offset, parent_pos)
            
            object_positions[obj_name] = pos
            
            for child in scene_graph[obj_name].get('children', []):
                calculate_positions(child, pos)
        
        # Calculate positions for all objects
        print("\nPhase 1: Calculating object positions")
        calculate_positions(root)
        
        # Sort objects by height level to ensure proper placement order
        sorted_objects = sorted(
            object_positions.keys(),
            key=lambda obj: scene_graph[obj]['height_level']
        )
        
        # Place objects in order from lowest to highest
        print("\nPhase 2: Placing objects in order")
        object_ids = {}
        for obj_name in sorted_objects:
            pos = object_positions[obj_name]
            full_path = os.path.join(self.urdf_dir, f"{obj_name}.urdf")
            print(f"Placing {obj_name} at position {[f'{x:.3f}' for x in pos]}")
            print(f"Height level: {scene_graph[obj_name]['height_level']}")
            
            # Set upright orientation (no rotation)
            orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            # Load URDF with position and orientation
            obj_id = p.loadURDF(full_path, pos, orientation, useFixedBase=obj_name=="table")
            object_ids[obj_name] = obj_id
            
            # Set additional physics properties to keep objects stable
            p.changeDynamics(
                obj_id, 
                -1,  # -1 means base link
                linearDamping=0.9,  # Add damping to reduce bouncing
                angularDamping=0.9,  # Add rotational damping
                jointDamping=0.9,  # Damping for any joints
                restitution=0.1,  # Low restitution to reduce bouncing
                lateralFriction=1.0,  # High friction to prevent sliding
                spinningFriction=0.1,  # Add spinning friction
                rollingFriction=0.1,  # Add rolling friction
            )
            
            # Let each object settle briefly
            for _ in range(10):
                p.stepSimulation()
        
        # Let all objects settle before enabling real-time
        print("\nLetting objects settle...")
        for _ in range(100):
            p.stepSimulation()
            
        # Check final positions and orientations
        print("\nFinal object states:")
        for obj_name, obj_id in object_ids.items():
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            print(f"{obj_name}:")
            print(f"  Position: {[f'{x:.3f}' for x in pos]}")
            print(f"  Orientation: {[f'{x:.3f}' for x in orn]}")
        
        # Enable physics after all objects are stable
        print("\nEnabling physics simulation")
        p.setRealTimeSimulation(1)
        
        return object_positions
        
    def _sample_ground_position(self, bbox: Tuple[float, float, float], offset: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Sample a position on the ground, accounting for collision origin offset.
        Places objects very close to the ground to minimize initial drop."""
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        # Place base objects just slightly above ground to minimize drop
        z = bbox[2]/2 + offset[2] + 0.001  # Add 1mm clearance
        return (x, y, z)
    
    def _sample_position_by_relation(
        self, 
        relation: SpatialRelation,
        bbox: Tuple[float, float, float],
        offset: Tuple[float, float, float],
        parent_bbox: Tuple[float, float, float],
        parent_offset: Tuple[float, float, float],
        parent_pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Calculate position based on spatial relation to parent object, accounting for collision origin offsets."""
        print(f"\nCalculating position for relation: {relation.value}")
        print(f"Object bbox (w,l,h): {[f'{x:.3f}' for x in bbox]}")
        print(f"Object offset (x,y,z): {[f'{x:.3f}' for x in offset]}")
        print(f"Parent bbox (w,l,h): {[f'{x:.3f}' for x in parent_bbox]}")
        print(f"Parent offset (x,y,z): {[f'{x:.3f}' for x in parent_offset]}")
        print(f"Parent position (x,y,z): {[f'{x:.3f}' for x in parent_pos]}")
        
        px, py, pz = parent_pos
        result_pos = None
        
        if relation == SpatialRelation.ON:
            parent_top = pz + parent_offset[2] + parent_bbox[2]/2
            z = parent_top + bbox[2]/2
            result_pos = (px + offset[0], py + offset[1], z)
            print(f"Placing ON: parent_top = {parent_top:.3f}, final_z = {z:.3f}")
            
        elif relation == SpatialRelation.BELOW:
            parent_bottom = pz + parent_offset[2] - parent_bbox[2]/2
            z = parent_bottom - bbox[2]/2
            result_pos = (px + offset[0], py + offset[1], z)
            print(f"Placing BELOW: parent_bottom = {parent_bottom:.3f}, final_z = {z:.3f}")
            
        elif relation == SpatialRelation.NEXT_TO:
            z = bbox[2]/2 + offset[2]  # Place on ground with offset
            if np.random.choice([True, False]):
                offset_dist = np.random.uniform(0.3, 0.5)
                x = px + offset_dist * np.sign(np.random.uniform(-1, 1)) + offset[0]
                result_pos = (x, py + offset[1], z)
                print(f"Placing NEXT_TO: X-axis offset {offset_dist:.3f}")
            else:
                offset_dist = np.random.uniform(0.3, 0.5)
                y = py + offset_dist * np.sign(np.random.uniform(-1, 1)) + offset[1]
                result_pos = (px + offset[0], y, z)
                print(f"Placing NEXT_TO: Y-axis offset {offset_dist:.3f}")
                
        elif relation == SpatialRelation.ABOVE:
            clearance = 0.1
            parent_top = pz + parent_offset[2] + parent_bbox[2]/2
            z = parent_top + clearance + bbox[2]/2
            result_pos = (px + offset[0], py + offset[1], z)
            print(f"Placing ABOVE: parent_top = {parent_top:.3f}, clearance = {clearance:.3f}, final_z = {z:.3f}")
        
        else:
            result_pos = (px + offset[0], py + offset[1], pz + offset[2])
            print("Using default position (same as parent) with offset")
            
        print(f"Final position (x,y,z): {[f'{x:.3f}' for x in result_pos]}\n")
        return result_pos

    def _build_graph(self, relations: List[SemanticRelation]) -> Dict:
        """Build a directed graph of object relationships ensuring proper parent-child ordering.
        
        Args:
            relations: List of semantic relations between objects
            
        Returns:
            Dictionary representing the scene graph with proper parent-child relationships
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        # Initialize graph
        graph = {}
        
        # First pass: Create nodes and track all relations
        for rel in relations:
            if rel.source not in graph:
                graph[rel.source] = {'children': [], 'relation': None, 'height_level': 0}
            if rel.target not in graph:
                graph[rel.target] = {'children': [], 'relation': None, 'height_level': 0}
        
        # Second pass: Establish parent-child relationships based on spatial relations
        for rel in relations:
            source_node = graph[rel.source]
            target_node = graph[rel.target]
            
            # Determine parent-child relationship based on spatial relation
            if rel.relation == SpatialRelation.ON:
                # Object ON another is a child of the base object
                source_node['parent'] = rel.target
                source_node['relation'] = rel.relation
                target_node['children'].append(rel.source)
                source_node['height_level'] = target_node['height_level'] + 1
                
            elif rel.relation == SpatialRelation.BELOW:
                # Object BELOW another is a parent of that object
                target_node['parent'] = rel.source
                target_node['relation'] = SpatialRelation.ON
                source_node['children'].append(rel.target)
                target_node['height_level'] = source_node['height_level'] + 1
                
            elif rel.relation == SpatialRelation.ABOVE:
                # Similar to ON but with clearance
                source_node['parent'] = rel.target
                source_node['relation'] = rel.relation
                target_node['children'].append(rel.source)
                source_node['height_level'] = target_node['height_level'] + 1
                
            elif rel.relation == SpatialRelation.NEXT_TO:
                # For NEXT_TO, make the target the parent to maintain hierarchy
                source_node['parent'] = rel.target
                source_node['relation'] = rel.relation
                target_node['children'].append(rel.source)
                source_node['height_level'] = target_node['height_level']
        
        # Validate no circular dependencies
        visited = set()
        def check_cycles(node_name: str, path: set):
            if node_name in path:
                cycle = ' -> '.join(list(path) + [node_name])
                raise ValueError(f"Circular dependency detected: {cycle}")
            
            if node_name in visited:
                return
                
            visited.add(node_name)
            path.add(node_name)
            
            for child in graph[node_name]['children']:
                check_cycles(child, path.copy())
        
        # Check for cycles starting from each root node
        roots = [obj for obj, data in graph.items() if 'parent' not in data]
        if not roots:
            raise ValueError("No root object found in the scene graph")
            
        for root in roots:
            check_cycles(root, set())
            
        print("\nScene graph structure:")
        for obj, data in graph.items():
            parent = data.get('parent', 'None')
            relation = data.get('relation', 'None')
            children = data['children']
            level = data['height_level']
            print(f"{obj}: parent={parent}, relation={relation}, children={children}, height_level={level}")
            
        return graph
    
    def _find_base_object(self, graph: Dict) -> str:
        return next(obj for obj, data in graph.items() if 'parent' not in data)