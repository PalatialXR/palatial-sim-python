import trimesh
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
from dataclasses import dataclass
import torch

@dataclass
class SegmentationParams:
    """Parameters for point cloud segmentation and object extraction"""
    # Downsampling
    voxel_size: float = 0.02
    
    # Plane segmentation
    distance_threshold: float = 0.02
    ransac_n: int = 3
    num_iterations: int = 1000
    
    # Clustering
    cluster_tolerance: float = 0.05
    min_points: int = 100
    max_points: int = 25000

def fill_mesh_holes(mesh: o3d.geometry.TriangleMesh, hole_size: float = 0.1) -> o3d.geometry.TriangleMesh:
    """
    Fill holes in a mesh using a combination of methods
    
    Args:
        mesh: Input Open3D mesh
        hole_size: Maximum hole size to fill (relative to mesh size)
        
    Returns:
        Processed mesh with holes filled
    """
    # Get mesh dimensions
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
    max_hole_size = float(np.max(bbox_size)) * hole_size
    
    # Clean the mesh first
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Fill holes using Poisson surface reconstruction
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max_hole_size/10,
            max_nn=30
        )
    )
    
    # Perform Poisson reconstruction with parameters tuned for hole filling
    mesh_filled, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=8,
        width=0,
        scale=1.1,
        linear_fit=True
    )
    
    # Remove low density vertices (these are typically in areas of poor reconstruction)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh_filled.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean up the result
    mesh_filled.remove_degenerate_triangles()
    mesh_filled.remove_duplicated_triangles()
    mesh_filled.remove_duplicated_vertices()
    mesh_filled.remove_non_manifold_edges()
    
    return mesh_filled

def generate_mesh_lods(
    mesh: o3d.geometry.TriangleMesh,
    lod_faces: List[int],
    output_dir: Optional[str] = None,
    extension: str = ".ply"
) -> Dict[int, o3d.geometry.TriangleMesh]:
    """
    Generate multiple Level of Detail (LOD) versions of a mesh
    
    Args:
        mesh: Input mesh to generate LODs from
        lod_faces: List of target face counts for each LOD
        output_dir: Directory to save LOD meshes (optional)
        extension: File extension for saved meshes
        
    Returns:
        Dictionary mapping face counts to LOD meshes
    """
    mesh_lods = {}
    for face_count in lod_faces:
        print(f"Generating LOD with {face_count} faces...")
        mesh_lod = mesh.simplify_quadric_decimation(face_count)
        
        # Clean the LOD mesh
        mesh_lod.remove_degenerate_triangles()
        mesh_lod.remove_duplicated_triangles()
        mesh_lod.remove_duplicated_vertices()
        mesh_lod.remove_non_manifold_edges()
        
        if output_dir:
            output_path = Path(output_dir) / f"lod_{face_count}{extension}"
            o3d.io.write_triangle_mesh(str(output_path), mesh_lod)
        
        mesh_lods[face_count] = mesh_lod
    
    print(f"Successfully generated {len(lod_faces)} LOD meshes")
    return mesh_lods

def process_dust3r_output(
    glb_path: str,
    output_dir: Optional[str] = None,
    simplify: bool = False,
    target_faces: int = 100000,
    reconstruction_method: str = 'poisson',
    generate_lods: bool = False,
    lod_faces: Optional[List[int]] = None
) -> Tuple[o3d.geometry.TriangleMesh, trimesh.Trimesh]:
    """
    Process a GLB file from DUST3R into workable formats. Handles both mesh and point cloud inputs.
    
    Args:
        glb_path: Path to the DUST3R output GLB file
        output_dir: Directory to save processed files (optional)
        simplify: Whether to simplify the mesh
        target_faces: Target number of faces if simplifying
        reconstruction_method: Method to use for reconstruction ('poisson' or 'ball_pivot')
        generate_lods: Whether to generate LOD meshes
        lod_faces: List of target face counts for LOD generation
        
    Returns:
        Tuple of (Open3D mesh, Trimesh mesh)
    """
    # Load the GLB file using trimesh
    print(f"Loading GLB file: {glb_path}")
    scene = trimesh.load(glb_path)
    
    # Handle different input types
    if isinstance(scene, trimesh.Scene):
        # Get the geometry from the scene
        geometry = list(scene.geometry.values())
        if len(geometry) == 0:
            raise ValueError("No geometry found in GLB file")
        input_geom = geometry[0]
    else:
        input_geom = scene
    
    # Check if input is a point cloud
    is_point_cloud = isinstance(input_geom, trimesh.PointCloud) or not hasattr(input_geom, 'faces')
    
    if is_point_cloud:
        print("Input is a point cloud, converting to mesh...")
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        
        # Convert vertices to numpy array
        if hasattr(input_geom, 'vertices'):
            vertices = np.asarray(input_geom.vertices, dtype=np.float64)
        else:
            vertices = np.asarray(input_geom.points, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        # Transfer colors if they exist
        if hasattr(input_geom, 'colors'):
            colors = np.asarray(input_geom.colors, dtype=np.float64)
            if colors.shape[1] == 4:  # RGBA
                colors = colors[:, :3]  # Convert to RGB
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif hasattr(input_geom.visual, 'vertex_colors'):
            colors = np.asarray(input_geom.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals if they don't exist
        print("Estimating point cloud normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Compute adaptive radius for ball pivoting
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        # Reconstruct mesh from point cloud
        if reconstruction_method == 'poisson':
            print("Performing Poisson surface reconstruction...")
            o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=9,
                width=0,
                scale=1.1,
                linear_fit=False
            )
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            o3d_mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Crop to original bounding box
            bbox = pcd.get_axis_aligned_bounding_box()
            o3d_mesh = o3d_mesh.crop(bbox)
            
        else:  # ball_pivot
            print("Performing Ball Pivot surface reconstruction...")
            radii = [radius, radius * 2]  # Use adaptive radius
            o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
        
        print(f"Reconstructed mesh has {len(o3d_mesh.vertices)} vertices and {len(o3d_mesh.triangles)} triangles")
        
        # Clean up the mesh
        print("Cleaning up the mesh...")
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_non_manifold_edges()
        
    else:
        print(f"Input is a mesh with {len(input_geom.vertices)} vertices and {len(input_geom.faces)} faces")
        # Convert to Open3D format
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(input_geom.vertices, dtype=np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(input_geom.faces, dtype=np.int32))
        
        # Transfer properties if they exist
        if hasattr(input_geom.visual, 'uv') and input_geom.visual.uv is not None:
            o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(input_geom.visual.uv, dtype=np.float64))
        
        if hasattr(input_geom.visual, 'vertex_colors') and input_geom.visual.vertex_colors is not None:
            colors = np.asarray(input_geom.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Fill holes
    print("Filling holes in mesh...")
    o3d_mesh = fill_mesh_holes(o3d_mesh)
    
    # Optional mesh simplification
    if simplify and len(o3d_mesh.triangles) > target_faces:
        print(f"Simplifying mesh to {target_faces} faces...")
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
    
    # Ensure normals are computed
    o3d_mesh.compute_vertex_normals()
    
    # Convert back to trimesh for analysis
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Save processed files if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in different formats
        ply_path = output_dir / "mesh.ply"
        obj_path = output_dir / "mesh.obj"
        
        print(f"Saving processed files to {output_dir}")
        o3d.io.write_triangle_mesh(str(ply_path), o3d_mesh)
        trimesh_mesh.export(str(obj_path))
    
    # Generate LODs if requested
    if generate_lods:
        if lod_faces is None:
            # Default LOD face counts
            lod_faces = [100000, 50000, 10000, 1000]
        print("\nGenerating LOD meshes...")
        mesh_lods = generate_mesh_lods(
            o3d_mesh,
            lod_faces,
            output_dir=output_dir and Path(output_dir) / "lods"
        )
    
    return o3d_mesh, trimesh_mesh

def analyze_dust3r_mesh(mesh: trimesh.Trimesh) -> dict:
    """
    Analyze the DUST3R mesh properties
    
    Args:
        mesh: Trimesh object to analyze
        
    Returns:
        Dictionary containing mesh analysis results
    """
    # Basic mesh properties
    analysis = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "volume": mesh.volume,
        "is_watertight": mesh.is_watertight,
        "is_oriented": mesh.is_winding_consistent,
        "bounds": mesh.bounds.tolist()
    }
    
    # Visual properties
    if hasattr(mesh, 'visual'):
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            analysis.update({
                "has_texture": mesh.visual.uv is not None,
                "has_material": mesh.visual.material is not None,
                "texture_mode": "texture"
            })
        elif isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
            analysis.update({
                "has_vertex_colors": mesh.visual.vertex_colors is not None,
                "has_face_colors": mesh.visual.face_colors is not None,
                "texture_mode": "color"
            })
        else:
            analysis.update({
                "has_texture": False,
                "has_vertex_colors": False,
                "has_face_colors": False,
                "texture_mode": "none"
            })
    
    return analysis

def convert_to_pointcloud(
    input_path: Union[str, trimesh.Trimesh, trimesh.Scene, o3d.geometry.PointCloud]
) -> o3d.geometry.PointCloud:
    """
    Convert various input formats to Open3D point cloud
    
    Args:
        input_path: Input file path or geometry object
        
    Returns:
        Open3D point cloud
    """
    if isinstance(input_path, o3d.geometry.PointCloud):
        return input_path
        
    if isinstance(input_path, str):
        # Handle file inputs
        if input_path.endswith('.pcd'):
            return o3d.io.read_point_cloud(input_path)
        elif input_path.endswith('.ply'):
            return o3d.io.read_point_cloud(input_path)
        elif input_path.endswith('.glb') or input_path.endswith('.obj'):
            # Load with trimesh
            scene = trimesh.load(input_path)
            if isinstance(scene, trimesh.Scene):
                meshes = list(scene.geometry.values())
                if not meshes:
                    raise ValueError(f"No geometry found in {input_path}")
                # Combine all meshes
                vertices = []
                colors = []
                for mesh in meshes:
                    vertices.append(np.asarray(mesh.vertices))
                    if hasattr(mesh.visual, 'vertex_colors'):
                        colors.append(np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0)
                vertices = np.vstack(vertices)
                if colors:
                    colors = np.vstack(colors)
            else:
                vertices = np.asarray(scene.vertices)
                colors = (np.asarray(scene.visual.vertex_colors)[:, :3] / 255.0 
                         if hasattr(scene.visual, 'vertex_colors') else None)
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd
        else:
            raise ValueError("Unsupported file format. Use .pcd, .ply, .glb, or .obj")
            
    elif isinstance(input_path, (trimesh.Trimesh, trimesh.Scene)):
        # Handle trimesh objects
        if isinstance(input_path, trimesh.Scene):
            meshes = list(input_path.geometry.values())
            if not meshes:
                raise ValueError("Empty scene")
            # Combine all meshes
            vertices = []
            colors = []
            for mesh in meshes:
                vertices.append(np.asarray(mesh.vertices))
                if hasattr(mesh.visual, 'vertex_colors'):
                    colors.append(np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0)
            vertices = np.vstack(vertices)
            if colors:
                colors = np.vstack(colors)
        else:
            vertices = np.asarray(input_path.vertices)
            colors = (np.asarray(input_path.visual.vertex_colors)[:, :3] / 255.0 
                     if hasattr(input_path.visual, 'vertex_colors') else None)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    
    else:
        raise ValueError("Unsupported input type")

def extract_objects_from_pointcloud(
    input_cloud: Union[o3d.geometry.PointCloud, str, trimesh.Trimesh, trimesh.Scene],
    params: Optional[SegmentationParams] = None,
    output_dir: Optional[str] = None,
    visualize: bool = False
) -> List[o3d.geometry.PointCloud]:
    """
    Extract individual objects from a point cloud using segmentation and clustering.
    
    Args:
        input_cloud: Input point cloud, file path, or 3D model
        params: Segmentation parameters
        output_dir: Directory to save extracted objects
        visualize: Whether to visualize the results
        
    Returns:
        List of extracted object point clouds
    """
    # Set default parameters if none provided
    if params is None:
        params = SegmentationParams()
    
    # Convert input to point cloud
    try:
        pcd = convert_to_pointcloud(input_cloud)
    except Exception as e:
        raise ValueError(f"Failed to convert input to point cloud: {e}")
    
    # Check if point cloud is empty
    if len(pcd.points) == 0:
        raise ValueError("Input point cloud is empty")
    
    print(f"Processing point cloud with {len(pcd.points)} points")
    
    # Calculate appropriate voxel size based on point cloud density if not specified
    if params.voxel_size <= 0:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
        diagonal = np.linalg.norm(bbox_size)
        params.voxel_size = diagonal / 100.0  # Aim for roughly 100 voxels across longest dimension
    
    # 1. Preprocess the point cloud
    print("Preprocessing point cloud...")
    # Estimate normals if they don't exist
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.voxel_size * 2,
            max_nn=30
        )
    )
    
    # Downsample using voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size=params.voxel_size)
    if pcd_down is None or len(pcd_down.points) == 0:
        print("Warning: Downsampling resulted in empty point cloud. Adjusting voxel size...")
        params.voxel_size /= 2.0
        pcd_down = pcd.voxel_down_sample(voxel_size=params.voxel_size)
        if pcd_down is None or len(pcd_down.points) == 0:
            raise ValueError(f"Downsampling failed. Original cloud has {len(pcd.points)} points. Try adjusting voxel_size manually.")
    
    print(f"Downsampled to {len(pcd_down.points)} points")
    
    # Ensure we have enough points for RANSAC
    if len(pcd_down.points) < params.ransac_n:
        print("Warning: Too few points after downsampling. Adjusting parameters...")
        params.ransac_n = min(3, len(pcd_down.points))
    
    # 2. Remove major planes (e.g., floor, walls, tables)
    print("Removing planar surfaces...")
    remaining_cloud = pcd_down
    planes = []
    for _ in range(3):  # Try to find up to 3 major planes
        if len(remaining_cloud.points) < params.ransac_n:
            break
            
        try:
            plane_model, inliers = remaining_cloud.segment_plane(
                distance_threshold=params.distance_threshold,
                ransac_n=params.ransac_n,
                num_iterations=params.num_iterations
            )
            
            if len(inliers) < params.min_points:
                break
                
            # Extract plane
            plane_cloud = remaining_cloud.select_by_index(inliers)
            planes.append(plane_cloud)
            
            # Remove plane points
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
            print(f"Removed plane with {len(inliers)} points")
            
        except RuntimeError as e:
            print(f"Warning: Plane segmentation failed: {e}")
            break
    
    # If no points remain after plane removal, return empty list
    if len(remaining_cloud.points) == 0:
        print("No points remain after plane removal")
        return []
    
    # 3. Cluster remaining points into objects
    print("Clustering remaining points into objects...")
    try:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            labels = np.array(remaining_cloud.cluster_dbscan(
                eps=params.cluster_tolerance,
                min_points=params.min_points,
                print_progress=True
            ))
    except RuntimeError as e:
        print(f"Warning: Clustering failed: {e}")
        return []
    
    if len(labels) == 0 or labels.max() < 0:
        print("No clusters found")
        return []
    
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")
    
    # Extract each cluster as an object
    objects = []
    for i in range(max_label + 1):
        cluster_points = remaining_cloud.select_by_index(np.where(labels == i)[0])
        if len(cluster_points.points) > params.min_points and len(cluster_points.points) < params.max_points:
            objects.append(cluster_points)
            print(f"Object {i}: {len(cluster_points.points)} points")
    
    if len(objects) == 0:
        print("No valid objects found after filtering")
        return []
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save planes
        for i, plane in enumerate(planes):
            o3d.io.write_point_cloud(
                str(output_dir / f"plane_{i}.ply"),
                plane
            )
        
        # Save objects
        for i, obj in enumerate(objects):
            o3d.io.write_point_cloud(
                str(output_dir / f"object_{i}.ply"),
                obj
            )
    
    # Visualize if requested
    if visualize and len(objects) > 0:
        # Assign different colors to each object
        colors = np.random.rand(len(objects), 3)
        vis_objects = []
        
        # Color the planes in gray
        for plane in planes:
            plane_colored = o3d.geometry.PointCloud(plane)
            plane_colored.paint_uniform_color([0.7, 0.7, 0.7])
            vis_objects.append(plane_colored)
        
        # Color each object uniquely
        for i, obj in enumerate(objects):
            obj_colored = o3d.geometry.PointCloud(obj)
            obj_colored.paint_uniform_color(colors[i])
            vis_objects.append(obj_colored)
        
        # Visualize
        o3d.visualization.draw_geometries(vis_objects)
    
    return objects

def process_and_reconstruct_objects(
    input_cloud: Union[o3d.geometry.PointCloud, str],
    output_dir: Optional[str] = None,
    segmentation_params: Optional[SegmentationParams] = None,
    reconstruction_params: Optional[dict] = None,
    visualize: bool = False
) -> List[Tuple[o3d.geometry.TriangleMesh, trimesh.Trimesh]]:
    """
    Extract objects from a point cloud and reconstruct each as a mesh
    
    Args:
        input_cloud: Input point cloud or path
        output_dir: Directory to save results
        segmentation_params: Parameters for object segmentation
        reconstruction_params: Parameters for mesh reconstruction
        visualize: Whether to visualize results
        
    Returns:
        List of (Open3D mesh, Trimesh mesh) tuples for each reconstructed object
    """
    # Extract objects
    objects = extract_objects_from_pointcloud(
        input_cloud,
        params=segmentation_params,
        output_dir=output_dir and Path(output_dir) / "objects",
        visualize=visualize
    )
    
    # Set default reconstruction parameters
    if reconstruction_params is None:
        reconstruction_params = {
            'simplify': True,
            'target_faces': 10000,
            'reconstruction_method': 'poisson'
        }
    
    # Reconstruct each object
    reconstructed = []
    for i, obj_cloud in enumerate(objects):
        print(f"\nReconstructing object {i}...")
        
        # Convert to GLB format (in memory)
        temp_cloud = trimesh.PointCloud(
            vertices=np.asarray(obj_cloud.points),
            colors=np.asarray(obj_cloud.colors) if obj_cloud.has_colors() else None
        )
        
        # Reconstruct
        o3d_mesh, trimesh_mesh = process_dust3r_output(
            temp_cloud,
            output_dir=output_dir and Path(output_dir) / f"reconstructed/object_{i}",
            **reconstruction_params
        )
        
        reconstructed.append((o3d_mesh, trimesh_mesh))
    
    return reconstructed

# Example usage
if __name__ == "__main__":
    # Process the DUST3R output
    glb_path = "src/utils/assets/dust3r.glb"
    o3d_mesh, trimesh_mesh = process_dust3r_output(
        glb_path,
        output_dir="processed_output",
        simplify=True,
        target_faces=100000
    )
    
    # Analyze the mesh
    analysis = analyze_dust3r_mesh(trimesh_mesh)
    print("\nMesh Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")

    # Example of object extraction from GLB
    try:
        params = SegmentationParams(
            voxel_size=0.0,  # Auto-calculate based on point cloud size
            distance_threshold=0.02,
            cluster_tolerance=0.05,
            min_points=100,
            max_points=25000
        )

        # Extract and reconstruct objects directly from GLB
        reconstructed_objects = process_and_reconstruct_objects(
            glb_path,  # Use the GLB file directly
            output_dir="scene_objects",
            segmentation_params=params,
            visualize=True
        )
        
        print(f"\nSuccessfully reconstructed {len(reconstructed_objects)} objects")
        
    except Exception as e:
        print(f"Error processing model: {e}")

# PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")

# CUDA version
print(f"CUDA Version: {torch.version.cuda}")

# Current device information (if CUDA is available)
if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.get_device_name()}")
    print(f"Device Count: {torch.cuda.device_count()}")