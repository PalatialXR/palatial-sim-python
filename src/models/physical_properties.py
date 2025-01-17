from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class MaterialType(str, Enum):
    RIGID = "rigid"
    SOFT = "soft"
    DEFORMABLE = "deformable"
    LIQUID = "liquid"
    ARTICULATED = "articulated"

class MaterialProperties(BaseModel):
    material_type: MaterialType
    density: float = Field(..., description="Density in kg/m³")
    friction: float = Field(..., description="Friction coefficient between 0 and 1")
    roughness: float = Field(..., description="Surface roughness between 0 and 1")
    elasticity: float = Field(..., description="Elasticity coefficient between 0 and 1")

class GeometricProperties(BaseModel):
    width: float = Field(..., description="Width in meters")
    height: float = Field(..., description="Height in meters")
    depth: float = Field(..., description="Depth in meters")
    volume: float = Field(..., description="Volume in m³")
    symmetrical: bool = Field(..., description="Is object symmetrical")

class DynamicProperties(BaseModel):
    mass: float = Field(..., description="Mass in kg")
    can_roll: bool = Field(..., description="Can the object roll")
    can_slide: bool = Field(..., description="Can the object slide")
    stable_base: bool = Field(..., description="Has stable base")

class InteractionProperties(BaseModel):
    graspable: bool = Field(..., description="Can be grasped")
    stackable: bool = Field(..., description="Can be stacked")
    max_load: Optional[float] = Field(None, description="Maximum supportable weight in N")
    containment_volume: Optional[float] = Field(None, description="Containable volume in m³")

class PhysicalProperties(BaseModel):
    material: MaterialProperties
    geometry: GeometricProperties
    dynamics: DynamicProperties
    interaction: InteractionProperties

class ObjectDescription(BaseModel):
    category: str
    subcategory: Optional[str] = None
    properties: PhysicalProperties
    dynamic_properties: DynamicProperties
    interaction_properties: InteractionProperties
    geometric_properties: GeometricProperties
    material_properties: MaterialProperties 