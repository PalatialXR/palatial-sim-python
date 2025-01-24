# Segment Annotate

A tool for semantic segmentation, object description, and 3D model conversion. This tool combines SAM2 for segmentation, Gemini for spatial understanding, GPT-4   for object description, and Meshy API for 3D model conversion.

## Project Structure

```
segment_annotate/
├── src/                        # Source code
│   ├── segmentation/          # Segmentation module
│   │   ├── segmenter.py      # Core segmentation
│   │   ├── spatial_analyzer.py # Gemini integration
│   │   ├── interactive_segmenter.py # SAM2 interface
│   │   └── description_generator.py  # GPT-4 Vision
│   ├── depth/                # Depth estimation module
│   │   ├── depth_estimator.py # Depth Anything wrapper
│   │   └── visualization.py  # Depth visualization tools
│   ├── models/               # Data models
│   │   └── physical_properties.py  # Property schemas
│   ├── conversion/           # 3D conversion
│   │   └── meshy_converter.py  # Meshy API integration
│   └── main.py              # Main entry point
├── results/                  # Generated outputs
│   ├── segmentation/        # Segmentation results
│   ├── depth/              # Depth estimation results
│   └── models/             # 3D model outputs
├── debug/                   # Debug visualizations
├── requirements.txt         # Dependencies
└── .env                    # API keys
```

## Features

- **Segmentation Pipelines**
  - **Automatic Segmentation**
    - Uses SAM2 (Segment Anything Model 2)
    - Grid-based point selection
    - High-confidence mask filtering
    - Post-processing for better object boundaries
  
  - **Gemini-guided Segmentation**
    - Gemini 2.0 Flash for spatial understanding
    - Multiple detection modes:
      - **2D Box Detection**: Precise rectangular boundaries
      - **3D Box Detection**: Depth-aware object boundaries
      - **Direct Point Detection**: Single-point object identification
    - Intelligent point selection:
      - Box modes: 5 points per object (corners + center)
      - Point mode: Direct object center points
    - Improved accuracy for complex objects

  - **Depth Estimation**
    - Uses Depth Anything model
    - Optional prompt depth from LiDAR/sensors
    - High-resolution depth maps
    - Depth visualization tools
    - Integration with 3D model generation

- **Physical Object Description**
  - GPT-4 Vision analysis
  - Material properties
  - Geometric measurements
  - Dynamic characteristics
  - Interaction properties

- **3D Model Conversion**
  - Meshy API integration
  - Textured 3D models
  - Multiple topology options
  - Symmetry detection
  - Automatic UV mapping

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/segment_annotate.git
cd segment_annotate
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
MESHY_API_KEY=your_meshy_api_key
GOOGLE_API_KEY=your_google_api_key
```

5. Download the SAM2 checkpoint:
```bash
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Usage

### Automatic Segmentation

Process an image using the automatic grid-based segmentation:
```bash
python src/main.py --image path/to/image.jpg --pipeline auto --max-objects 20
```

### Gemini-guided Segmentation

Process an image using Gemini's spatial understanding with different modes:

1. 2D Box Detection (Default):
```bash
python src/main.py --image path/to/image.jpg --pipeline gemini --mode 2d --max-objects 10 --point-offset 0.2 --debug
```

2. 3D Box Detection:
```bash
python src/main.py --image path/to/image.jpg --pipeline gemini --mode 3d --max-objects 10 --point-offset 0.2 --debug
```

3. Direct Point Detection:
```bash
python src/main.py --image path/to/image.jpg --pipeline gemini --mode points --max-objects 10 --debug
```

### Depth Estimation

Estimate depth from an image:
```bash
python src/main.py --image path/to/image.jpg --pipeline depth --output-dir results/depth
```

With LiDAR prompt:
```bash
python src/main.py --image path/to/image.jpg --pipeline depth --prompt-depth path/to/lidar.png --output-dir results/depth
```

### Physics Simulation and Scene Reconstruction

The tool now supports physics-based scene reconstruction with video recording:

```bash
python test_graph.py --image path/to/image.jpg --depth path/to/depth.png
```

Features:
- PyBullet-based physics simulation
- Stable object placement and stacking
- Automatic relationship detection (on, next_to, etc.)
- High-quality video recording (1920x1080)
- TinyRenderer for headless operation
- Configurable physics parameters

Physics simulation parameters can be configured through `SceneConfig`:
```python
@dataclass
class SceneConfig:
    gravity: Tuple[float, float, float] = (0, 0, -9.81)
    timestep: float = 1/240.0
    stability_threshold: float = 0.02
    settlement_steps: int = 200
```

The simulation:
1. Loads RGB image and depth map
2. Creates physics objects with proper mass and friction
3. Places objects according to spatial relationships
4. Ensures stability through iterative settling
5. Records a 5-second video at 30 FPS
6. Saves the result as 'simulation.mp4'

### 3D Model Conversion

By default, the tool only performs segmentation. To also generate 3D models:
```bash
python src/main.py --image path/to/image.jpg --pipeline auto --convert-3d
```

### Advanced Options

- `--output-dir`: Specify custom output directory
- `--checkpoint`: Path to SAM2 checkpoint
- `--model-cfg`: Path to SAM2 model config
- `--point-offset`: Offset ratio for point selection (Box modes only)
- `--temperature`: Temperature for Gemini generation
- `--debug`: Enable debug visualizations
- `--convert-3d`: Enable 3D model conversion (requires Meshy API key)

## Output Structure

Each run creates timestamped directories:

### Segmentation Output (`results/segmentation/output_TIMESTAMP/`)
- `segments/`: Individual segment images (PNG with transparency)
- `masks/`: Binary mask files
- `segment_descriptions.json`: GPT-4 Vision descriptions
- `segments_metadata.json`: Segment metadata
- `masks_metadata.json`: Mask metadata
- `segmentation_visualization.png`: Visual overlay of segments

### Depth Output (`results/depth/output_TIMESTAMP/`)
- `depth_map.png`: Estimated depth map
- `depth_visualization.png`: Colored depth visualization
- `depth_metadata.json`: Depth estimation metadata
- `comparison.png`: Side-by-side with prompt depth (if provided)

### Debug Output (`debug/gemini_debug_TIMESTAMP/`)
- `gemini_analysis.png`: Visualization of detected boxes/points
- `gemini_boxes.json`: Raw detection data from Gemini
- `gemini_points.json`: Point detection data (points mode only)

### 3D Models Output (`results/models/output_TIMESTAMP/`)
- `3d_models/`: Generated OBJ files
- `3d_conversion_results.json`: Conversion metadata and results

## Physical Properties

The tool analyzes several physical properties for each segment:

- **Material Properties**
  - Material type (rigid, soft, deformable, etc.)
  - Density (kg/m³)
  - Friction coefficient
  - Surface roughness
  - Elasticity

- **Geometric Properties**
  - Dimensions (width, height, depth in meters)
  - Volume (m³)
  - Symmetry analysis

- **Dynamic Properties**
  - Mass estimation
  - Movement capabilities (rolling, sliding)
  - Stability analysis

- **Interaction Properties**
  - Graspability
  - Stackability
  - Load-bearing capacity
  - Containment volume

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key (for GPT-4o-mini)
- Meshy API key (for 3D conversion)
- Google API key (for Gemini)

## To-Do

### High Priority
1. Fix PromptDA for depth estimation
2. Debug scaling issues in Gemini-guided segmentation
3. Fix spatial placement algorithm
4. Create URDF files for better simulation
5. Create new simulation environment with PyBullet to import URDF files and spatial information

### Future Enhancements
- Improve video recording quality
- Add support for articulated objects
- Implement real-time visualization
- Add multi-camera views
- Support dynamic scenes

## License

MIT License - see LICENSE file for details 

## Testing Scene Generation

### Prerequisites
1. Make sure you have the required URDF files in the correct locations:
   ```
   assets/urdf/furniture/table.urdf
   assets/urdf/electronics/monitor.urdf
   ```

2. Install dependencies:
   ```bash
   pip install numpy opencv-python pybullet
   ```

### Running the Test
1. Run the test script:
   ```bash
   python test_scene.py
   ```

The script will:
1. Create a scene with a table and monitor
2. Place the objects using physics simulation
3. Record a 5-second video of the stable scene
4. Save the result as `output/simulation.mp4`

### Expected Behavior
- The table will be placed first at a random position near the ground
- The monitor will be placed on top of the table
- Physics simulation will run until objects are stable
- A video will be recorded showing the final scene

### Troubleshooting
- If objects are unstable, try adjusting the physics parameters in `SceneConfig`
- If objects are too close/far, adjust the workspace bounds in `SceneGenerator`
- Check URDF files if objects appear with incorrect dimensions
- Make sure all URDF files are in their correct category folders under `assets/urdf/` 

# Scene Graph Generator

A Python-based 3D scene generation system that creates physically accurate scenes from semantic spatial relationships using PyBullet physics engine.

## Overview

This system allows you to:
- Generate 3D scenes from semantic spatial relationships (e.g., "monitor ON table")
- Maintain physical accuracy using PyBullet physics simulation
- Handle complex object relationships and hierarchies
- Ensure stable object placement and physics interactions

## Features

### Spatial Relations
The system supports four types of spatial relationships:
- `ON`: Places an object directly on top of another (e.g., monitor on table)
- `NEXT_TO`: Places an object beside another with appropriate spacing
- `ABOVE`: Places an object above another with clearance
- `BELOW`: Places an object below another (inverse of ABOVE)

### Physics Simulation
- Real-time physics using PyBullet
- Stable object placement with collision detection
- Proper handling of object masses and inertia
- Friction and damping for realistic behavior
- Automatic settling of objects before enabling real-time physics

### Object Handling
- URDF-based object definitions
- Automatic bounding box calculation
- Proper handling of collision origins and offsets
- Support for complex object hierarchies
- Prevention of circular dependencies

## Usage

### Basic Example
```python
from scene_graph.scene_generator import Scene3DGenerator, SemanticRelation, SpatialRelation

# Initialize generator
generator = Scene3DGenerator()
generator.init_physics(gui=True)

# Define spatial relations
relations = [
    SemanticRelation(
        source="monitor",
        relation=SpatialRelation.ON,
        target="table",
        confidence=1.0
    )
]

# Generate scene
positions = generator.place_objects(relations)
```

### Object Requirements
Objects must be defined in URDF format with:
- Proper collision geometries
- Inertial properties
- Contact parameters
- Visual meshes (optional)

Example URDF structure:
```xml
<robot name="object">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <collision>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <contact>
        <lateral_friction value="1.0"/>
      </contact>
    </collision>
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Technical Details

### Scene Graph Building
1. Creates a directed graph of object relationships
2. Validates and prevents circular dependencies
3. Assigns height levels for proper placement order
4. Handles parent-child relationships based on spatial relations

### Physics Properties
- Objects are placed with proper orientations
- Base objects (like tables) can be fixed in place
- Physics properties for stability:
  - Linear and angular damping
  - Friction coefficients
  - Restitution values
  - Joint properties

### Position Calculation
- Accounts for object dimensions and offsets
- Handles collision origin offsets
- Maintains proper spacing between objects
- Ensures stable stacking and placement

## Dependencies
- PyBullet
- NumPy
- Python 3.7+

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install pybullet numpy
```

## Contributing
When adding new objects:
1. Create URDF files with proper physics properties
2. Test stability with existing objects
3. Ensure proper collision geometries
4. Add appropriate inertial properties

## License
MIT License 