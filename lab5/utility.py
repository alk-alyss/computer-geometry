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

    if num_vertices is None:
        num_vertices = triangles.max()+1

    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=np.uint16)

    for tri in triangles:
        v1, v2, v3 = tri
        adj_matrix[v1, v2] = 1
        adj_matrix[v2, v1] = 1
        adj_matrix[v2, v3] = 1
        adj_matrix[v3, v2] = 1
        adj_matrix[v3, v1] = 1
        adj_matrix[v1, v3] = 1

    return adj_matrix

#TASK-3 (Lab)
def adjacency_matrix_sparse(triangles, num_vertices = None):

    if num_vertices is None:
        num_vertices = triangles.max()+1

    adj_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.uint16)

    for tri in triangles:
        v1, v2, v3 = tri
        adj_matrix[v1, v2] = 1
        adj_matrix[v2, v1] = 1
        adj_matrix[v2, v3] = 1
        adj_matrix[v3, v2] = 1
        adj_matrix[v3, v1] = 1
        adj_matrix[v1, v3] = 1

    return adj_matrix.tocsr()

def adjacency_matrix(triangles, num_vertices=None):
    return adjacency_matrix_sparse(triangles, num_vertices).toarray()
    # return adjacency_matrix_dense(triangles, num_vertices)

#TASK-3 (Lab)
def degree_matrix(adj, exponent=1):

    num_vertices = adj.shape[0]
    diagonals = np.zeros(num_vertices)

    if exponent==1:
        diagonals = adj.sum(axis=0)
        return diags(diagonals, format="csr", dtype=np.int32)
    else:
        diagonals = np.float_power(adj.sum(axis=0), exponent)
        return diags(diagonals, format="csr", dtype=np.float32)

#TASK-2
def delta_coordinates_single(idx, vertices, triangles, k=1):

    vi = vertices[idx]

    neighbors = k_ring_adjacency(idx, triangles, k)
    delta = vi - vertices[neighbors,:].mean(0)

    return delta

#TASK-4
def delta_coordinates(vertices, triangles, laplacian=None, use_laplacian=True):

    if laplacian is not None:
        return laplacian @ vertices

    if triangles is not None:
        if use_laplacian:
            L = random_walk_laplacian(triangles)
            delta = L @ vertices
        else:
            delta = np.zeros_like(vertices)
            for i, vi in enumerate(vertices):
                neighbors = k_ring_adjacency(i, triangles, 1)
                delta[i] = vi - vertices[neighbors, :].mean(0)

        return delta

#TASK-1
def k_ring(idx, adj_list, k = 1):
    open_list = []
    depths = []
    closed_list = []

    open_list.append(idx)
    depths.append(0)

    while len(open_list) > 0:
        current_vertex = open_list.pop(0)
        depth = depths.pop(0)

        closed_list.append(current_vertex)

        if depth + 1 > k:
            continue

        for vertex in adj_list[current_vertex]:
            if vertex not in open_list and vertex not in closed_list:
                open_list.append(vertex)
                depths.append(depth+1)

    closed_list.pop(0)
    return closed_list

def k_ring_recursive(idx, triangles, k=1):

    if not k:
        return np.array([])

    if isinstance(idx, int):
        idx = np.array([idx], dtype=np.uint16)

    new_ids = np.array([], dtype=np.uint16)

    for id in idx:
        for t in triangles:
            if id in t:
                new_ids = np.hstack((new_ids, t[t-id != 0]))

    new_ids = np.unique(new_ids)

    return np.unique(np.hstack((new_ids, k_ring_recursive(new_ids, triangles, k-1)))).astype(np.uint32)

#TASK-3
def k_ring_adjacency(idx, triangles, k=1, num_vertices=None):

    adj_matrix = adjacency_matrix(triangles, num_vertices)

    adj_matrix = adj_matrix ** k

    neighbors = adj_matrix[idx, :]

    return neighbors.nonzero()[1]

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

    num_vertices = triangles.max()+1

    A = adjacency_matrix(triangles, num_vertices=num_vertices)
    D = degree_matrix(A, exponent=1)

    L = D - A

    return L

#TASK-4
def random_walk_laplacian(triangles, subtract=True):

    num_vertices = triangles.max()+1

    A = adjacency_matrix(triangles, num_vertices=num_vertices)
    Dinv = degree_matrix(A, exponent=-1)

    if subtract:
        L = eye(num_vertices, num_vertices, 0) - Dinv @ A
    else:
        L = Dinv @ A

    return L

if __name__ == "__main__":

    create_test_mesh()