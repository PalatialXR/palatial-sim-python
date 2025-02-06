from enum import Enum
from typing import List, Optional, Set, Tuple, Dict
from pydantic import BaseModel, Field
import numpy as np

# Update enums to be simple strings for compatibility
SPATIAL_RELATIONS = [
    "on", "next_to", "above", "below", "in_front_of",
    "behind", "left_of", "right_of", "inside", "between", "aligned_with"
]

class ObjectState(BaseModel):
    """Physical state of an object during scene planning"""
    name: str
    bbox: Tuple[float, float, float]
    position: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float]] = None
    children_on: Set[str] = set()
    parent: Optional[str] = None
    grid_resolution: Optional[float] = None
    available_surface: Optional[List[List[bool]]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def init_grid(self, resolution: float = 0.1):
        """Initialize placement grid for surface.
        
        Args:
            resolution: Grid cell size in meters (default: 0.1m or 10cm)
        """
        self.grid_resolution = resolution
        
        # Calculate grid size based on the larger dimension with some padding
        padding_factor = 1.2  # Add 20% padding for better placement options
        max_dim = max(self.bbox[0], self.bbox[1]) * padding_factor
        grid_size = max(
            int(max_dim / resolution),
            10  # Minimum grid size of 10x10
        )
        
        # Create grid with all cells initially available
        self.available_surface = [[True] * grid_size for _ in range(grid_size)]

class SpatialProperties(BaseModel):
    """Spatial properties of an object in the scene"""
    position_x: Optional[float] = Field(None, description="X coordinate in world space")
    position_y: Optional[float] = Field(None, description="Y coordinate in world space")
    position_z: Optional[float] = Field(None, description="Z coordinate in world space")
    orientation_roll: Optional[float] = Field(None, description="Roll angle in degrees")
    orientation_pitch: Optional[float] = Field(None, description="Pitch angle in degrees")
    orientation_yaw: Optional[float] = Field(None, description="Yaw angle in degrees")
    width: Optional[float] = Field(None, description="Width in meters")
    length: Optional[float] = Field(None, description="Length in meters")
    height: Optional[float] = Field(None, description="Height in meters")

class HierarchicalProperties(BaseModel):
    """Hierarchical relationships of an object in the scene"""
    level: int = Field(..., description="Hierarchy level (0 for ground objects)")
    is_anchor: bool = Field(..., description="Whether object contacts ground/base")
    parent: Optional[str] = Field(None, description="Name of supporting object")
    children: List[str] = Field(default_factory=list, description="Supported object names")

class SpatialRelation(BaseModel):
    """Spatial relationship between objects"""
    source: str = Field(..., description="Source object name")
    target: str = Field(..., description="Target object name")
    relation_type: str = Field(..., description="Type of spatial relationship")
    distance: Optional[float] = Field(None, description="Distance in meters")
    contact_area: Optional[float] = Field(None, description="Contact area in m²")

class SemanticRelation(BaseModel):
    """Semantic relationship between objects with confidence score"""
    source: str = Field(..., description="Source object name")
    relation: str = Field(..., description="Type of spatial relationship")
    target: str = Field(..., description="Target object name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the relationship")

class SceneObject(BaseModel):
    """Object in the scene"""
    name: str = Field(..., description="Unique object identifier")
    category: str = Field(..., description="Object Sapien Partnet Mobility Dataset category")
    description: str = Field(..., description="Detailed Object description")
    spatial: Optional[SpatialProperties] = Field(None, description="Spatial properties")
    hierarchy: HierarchicalProperties = Field(..., description="Hierarchical relationships")

class SceneGraph(BaseModel):
    """Complete scene graph"""
    objects: List[SceneObject] = Field(..., description="Scene objects")
    relationships: List[SpatialRelation] = Field(..., description="Spatial relationships")