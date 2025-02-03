"""
This module contains centralized prompts used throughout the scene understanding
and generation system. Keeping prompts in a central location makes it easier to
maintain consistency and make updates across the application.
"""

# Scene Understanding Prompts
SCENE_ANALYSIS_PROMPT = """
Analyze the given scene and identify:
1. All objects present in the scene
2. Their spatial relationships (e.g., 'on top of', 'next to', 'in front of')
3. Key attributes of objects (size, color, orientation)
4. The overall scene layout and purpose

Provide the analysis in a structured format.
"""

OBJECT_RELATIONSHIP_PROMPT = """
Given two objects in the scene, describe their spatial relationship in detail:
- Relative position (above, below, beside, etc.)
- Approximate distance
- Any interactions or functional relationships
- Orientation relative to each other
"""

# Scene Generation Prompts
SCENE_GENERATION_PROMPT = """
Generate a detailed scene description based on the following elements:
1. Room type and purpose
2. Required furniture and objects
3. Desired layout and arrangement
4. Specific constraints or requirements

Include spatial relationships and positioning details.
"""

OBJECT_PLACEMENT_PROMPT = """
For the given object, specify:
1. Optimal placement location
2. Orientation and rotation
3. Relationship to nearby objects
4. Any specific constraints or requirements
5. Reasoning for the placement decisions
"""

# 3D Model Generation Prompts
MODEL_REQUIREMENTS_PROMPT = """
Specify the requirements for a 3D model:
1. Object type and category
2. Key dimensions and proportions
3. Important features and details
4. Material properties
5. Any specific constraints or limitations
"""

# Scene Validation Prompts
SCENE_VALIDATION_PROMPT = """
Validate the generated scene by checking:
1. Object placement feasibility
2. Spatial relationship consistency
3. Adherence to physical constraints
4. Fulfillment of functional requirements
5. Overall scene coherence
"""

# Customization Functions
def get_scene_analysis_prompt(scene_type: str = None) -> str:
    """
    Get a customized scene analysis prompt based on scene type.
    
    Args:
        scene_type: Type of scene (e.g., 'office', 'kitchen', 'living_room')
        
    Returns:
        Customized prompt string
    """
    if not scene_type:
        return SCENE_ANALYSIS_PROMPT
        
    return f"{SCENE_ANALYSIS_PROMPT}\nSpecific focus areas for {scene_type} scenes:"

def get_object_placement_prompt(object_type: str) -> str:
    """
    Get a customized object placement prompt for specific object types.
    
    Args:
        object_type: Type of object to place
        
    Returns:
        Customized prompt string
    """
    return f"{OBJECT_PLACEMENT_PROMPT}\nSpecific considerations for {object_type}:" 