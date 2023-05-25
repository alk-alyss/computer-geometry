import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import cm
from scipy.sparse import csr_matrix, lil_matrix, diags, eye
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig

def create_test_mesh():

    theta = np.linspace(0, 2*np.pi, 7)[:-1]
    x, y, z = np.cos(theta), np.zeros(6), np.sin(theta)

    vertices = np.vstack((x,y,z)).T
    vertices = np.vstack((np.array([0,1,0]), vertices))

    triangles = np.array([
        [0,1,2], [0,2,3], [0,3,4], [0,4,5], [0,5,6], [0,6,1],
        [0,2,1], [0,3,2], [0,4,3], [0,5,4], [0,6,5], [0,1,6]
    ])

    delta = delta_coordinates_single(0, vertices, triangles)

    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles)
        ),

        o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.array([vertices[0,:], vertices[0,:]-delta])),
            o3d.utility.Vector2iVector(np.array([[0,1]]))
        ).paint_uniform_color(np.array([0,1,0]))
    ])

#TASK-3
def adjacency_matrix_dense(triangles, num_vertices=None):

    #A = ....

    return A

#TASK-3 (Lab)
def adjacency_matrix_sparse(triangles, num_vertices = None):

    # A = ...

    return A

#TASK-3 (Lab) 
def degree_matrix(adj, exponent=1):

    num_vertices = adj.shape[0]
    diagonals = np.zeros(num_vertices)

    if exponent==1:
        for i in range(num_vertices):
            diagonals[i] = adj[i,:].toarray().sum()
        return diags(diagonals, format="csr", dtype=np.int32)
    else:
        for i in range(num_vertices):
            diagonals[i] = adj[i,:].toarray().sum().astype(np.float32)**exponent
        return diags(diagonals, format="csr", dtype=np.float32)

#TASK-2
def delta_coordinates_single(idx, vertices, triangles, k=1):

    pass

#TASK-4
def delta_coordinates(vertices, triangles, use_laplacian=True):

    pass

#TASK-1
def k_ring_recursive(idx, triangles, k = 1):

    pass
    #...

#TASK-3
def k_ring_adjacency(idx, triangles, k=1, num_vertices=None):

    pass
    #...

def sample_colormap(scalars, name="inferno"):

    avail_maps = ["inferno", "magma", "viridis", "cividis"]

    if name not in avail_maps:
        warnings.warn(f"Only {avail_maps} colormaps are supported. Using inferno.")
        name = "inferno"

    colormap = cm.get_cmap(name, 12)
    colors = colormap(scalars)

    return colors[:,:-1]

#TASK-5
def graph_laplacian(triangles):

    pass
    #...

#TASK-4
def random_walk_laplacian(triangles, subtract=True):

    pass
    #...

if __name__ == "__main__":

    create_test_mesh()