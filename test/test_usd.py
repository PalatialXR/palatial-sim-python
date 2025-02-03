from pxr import Usd, UsdGeom, Sdf

# Create a new stage in memory
stage = Usd.Stage.CreateInMemory()

# Define a new Xform prim at the root of the stage
table_prim = stage.DefinePrim('/Table', 'Xform')

# Define a new Mesh prim as a child of the Xform
mesh_prim = UsdGeom.Mesh.Define(stage, '/Table/Mesh')

# Define the points of the table
points = [
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
]

# Define the face vertex counts of the table
face_vertex_counts = [4, 4, 4, 4, 4, 4]

# Define the face vertex indices of the table
face_vertex_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 5, 1, 1, 5, 6, 2,
    2, 6, 7, 3, 3, 7, 4, 0
]

# Create the points attribute on the mesh
points_attr = mesh_prim.CreatePointsAttr(points)

# Create the face vertex counts attribute on the mesh
face_vertex_counts_attr = mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)

# Create the face vertex indices attribute on the mesh
face_vertex_indices_attr = mesh_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)

# Print the resulting USD stage
print(stage.GetRootLayer().ExportToString())
