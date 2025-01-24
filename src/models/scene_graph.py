from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from enum import Enum

class SpatialRelationType(str, Enum):
    ON = "on"
    NEXT_TO = "next_to"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    INSIDE = "inside"
    ABOVE = "above"
    BELOW = "below"
    SUPPORTED_BY = "supported_by"

class SpatialOffset(BaseModel):
    x: float = Field(0.0, description="X offset in meters")
    y: float = Field(0.0, description="Y offset in meters")
    z: float = Field(0.0, description="Z offset in meters")

class SpatialRotation(BaseModel):
    roll: float = Field(0.0, description="Roll angle in radians")
    pitch: float = Field(0.0, description="Pitch angle in radians")
    yaw: float = Field(0.0, description="Yaw angle in radians")

class AlignmentType(str, Enum):
    CENTER = "center"
    EDGE = "edge"
    CORNER = "corner"
    CUSTOM = "custom"

class SpatialConstraint(BaseModel):
    relation_type: SpatialRelationType
    target_object: str = Field(..., description="Name of the target object")
    offset: SpatialOffset = Field(default_factory=SpatialOffset)
    rotation: SpatialRotation = Field(default_factory=SpatialRotation)
    alignment: AlignmentType = Field(AlignmentType.CENTER)
    margin: float = Field(0.1, description="Minimum distance between objects in meters")
    max_distance: Optional[float] = Field(None, description="Maximum allowed distance in meters")

class SceneObject(BaseModel):
    name: str
    category: str
    urdf_path: str = Field(..., description="Path to the URDF file for this object")
    constraints: List[SpatialConstraint] = Field(default_factory=list)
    position: Optional[Tuple[float, float, float]] = Field(None, description="Current position (x,y,z) in meters")
    rotation: Optional[Tuple[float, float, float]] = Field(None, description="Current rotation (roll,pitch,yaw) in radians")

class SceneGraph(BaseModel):
    root_object: str = Field(..., description="Name of the base/root object")
    objects: List[SceneObject] = Field(..., description="List of all objects in the scene")
    
    class Config:
        schema_extra = {
            "example": {
                "root_object": "table",
                "objects": [
                    {
                        "name": "table",
                        "category": "furniture",
                        "urdf_path": "assets/urdf/table.urdf",
                        "constraints": [],
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0]
                    },
                    {
                        "name": "monitor",
                        "category": "electronics",
                        "urdf_path": "assets/urdf/monitor.urdf",
                        "constraints": [
                            {
                                "relation_type": "on",
                                "target_object": "table",
                                "offset": {"x": 0, "y": 0, "z": 0},
                                "rotation": {"roll": 0, "pitch": 0, "yaw": 0},
                                "alignment": "center",
                                "margin": 0.05
                            }
                        ],
                        "position": None,
                        "rotation": None
                    }
                ]
            }
        } 