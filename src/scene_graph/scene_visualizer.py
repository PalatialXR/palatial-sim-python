import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class SceneVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def draw_cuboid(self, position, dimensions, label="", color='b', alpha=0.3):
        """Draw a cuboid at given position with given dimensions."""
        try:
            # Convert position and dimensions to float values
            x, y, z = [float(p) if isinstance(p, (int, float, str)) else float(p[0]) for p in position]
            w, l, h = [float(d) if isinstance(d, (int, float, str)) else float(d[0]) for d in dimensions]
            
            # Define vertices
            vertices = np.array([
                [x-w/2, y-l/2, z],
                [x+w/2, y-l/2, z],
                [x+w/2, y+l/2, z],
                [x-w/2, y+l/2, z],
                [x-w/2, y-l/2, z+h],
                [x+w/2, y-l/2, z+h],
                [x+w/2, y+l/2, z+h],
                [x-w/2, y+l/2, z+h]
            ])
            
            # Define faces using vertex indices
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            
            # Plot faces
            for face in faces:
                face_x = [v[0] for v in face]
                face_y = [v[1] for v in face]
                face_z = [v[2] for v in face]
                self.ax.plot_surface(
                    np.array([face_x]).T,
                    np.array([face_y]).T,
                    np.array([face_z]).T,
                    color=color,
                    alpha=alpha
                )
            
            # Add label at center top
            if label:
                # Calculate text position at center top of cuboid
                text_x = float(x)  # Center x
                text_y = float(y)  # Center y
                text_z = float(z + h)  # Top of cuboid
                self.ax.text(text_x, text_y, text_z, label, fontsize=8)
                
        except Exception as e:
            print(f"Error drawing cuboid for {label}: {str(e)}")
            print(f"Position: {position}")
            print(f"Dimensions: {dimensions}")

    def visualize_scene(self, objects):
        """Visualize all objects in the scene."""
        self.ax.clear()
        
        # Draw ground plane
        ground_size = 3
        xx, yy = np.meshgrid([-ground_size, ground_size], [-ground_size, ground_size])
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.2)
        
        # Color map for different object types
        colors = {
            'Table': 'brown',
            'Monitor': 'blue',
            'Keyboard': 'green',
            'Mouse': 'red'
        }
        
        # Draw each object
        for obj in objects:
            obj_type = obj['name'].split('_')[0]
            color = colors.get(obj_type, 'purple')
            
            if 'position' in obj and 'dimensions' in obj:
                self.draw_cuboid(
                    obj['position'],
                    obj['dimensions'],
                    obj['name'],
                    color=color
                )
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Scene Layout Visualization')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
        # Add grid
        self.ax.grid(True)
        
        return self.fig

    def save_visualization(self, filepath):
        """Save the visualization to a file."""
        self.fig.savefig(filepath)
        plt.close(self.fig)

def visualize_scene_plan(objects, output_path):
    """Visualize a scene plan and save it to a file."""
    visualizer = SceneVisualizer()
    visualizer.visualize_scene(objects)
    visualizer.save_visualization(output_path) 