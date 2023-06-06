import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import cm
from scipy.sparse import csr_matrix, lil_matrix, diags, eye
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig
from typing import List

def adjacency_matrix_dense(triangles:np.ndarray, num_vertices:int=None) -> np.ndarray:

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

def adjacency_matrix_sparse(triangles:np.ndarray, num_vertices = None) -> csr_matrix:

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

def adjacency_matrix(triangles:np.ndarray, num_vertices:int=None) -> np.ndarray | csr_matrix:
    # return adjacency_matrix_sparse(triangles, num_vertices)
    return adjacency_matrix_dense(triangles, num_vertices)

def degree_matrix(adj:np.ndarray, exponent:int=1) -> csr_matrix:

    num_vertices = adj.shape[0]
    diagonals = np.zeros(num_vertices)

    if exponent==1:
        diagonals = adj.sum(axis=0)
        return diags(diagonals, format="csr", dtype=np.int32)
    else:
        diagonals = np.float_power(adj.sum(axis=0), exponent)
        return diags(diagonals, format="csr", dtype=np.float32)

def delta_coordinates(vertices:np.ndarray, laplacian:np.ndarray) -> np.ndarray:

    return laplacian @ vertices

def k_ring(idx:int, adj_list:np.ndarray, k:int = 1) -> List[int]:
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

def k_ring_recursive(idx:int | List[int], triangles:np.ndarray, k:int=1) -> np.ndarray:

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

def k_ring_adjacency(idx:np.ndarray, triangles:np.ndarray, k=1, num_vertices:int=None) -> np.ndarray:

    adj_matrix = adjacency_matrix(triangles, num_vertices)

    adj_matrix = adj_matrix ** k

    neighbors = adj_matrix[idx, :]

    return neighbors.nonzero()[1]

def sample_colormap(scalars:np.ndarray, name:str="inferno") -> np.ndarray:

    avail_maps = ["inferno", "magma", "viridis", "cividis"]

    if name not in avail_maps:
        warnings.warn(f"Only {avail_maps} colormaps are supported. Using inferno.")
        name = "inferno"

    colormap = cm.get_cmap(name, 12)
    colors = colormap(scalars)

    return colors[:,:-1]

def graph_laplacian(triangles:np.ndarray) -> np.ndarray:

    num_vertices = triangles.max()+1

    A = adjacency_matrix(triangles, num_vertices=num_vertices)
    D = degree_matrix(A, exponent=1)

    L = D - A

    return L

def random_walk_laplacian(triangles:np.ndarray, subtract:bool=True) -> np.ndarray:

    num_vertices = triangles.max()+1

    A = adjacency_matrix(triangles, num_vertices=num_vertices)
    Dinv = degree_matrix(A, exponent=-1)

    if subtract:
        L = eye(num_vertices, num_vertices, 0) - Dinv @ A
    else:
        L = Dinv @ A

    return L

def generate_noise(num_of_vertices:int, seed:int=42) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)

    delta = rng.random((num_of_vertices, 3))
    delta -= 0.5

    return delta

if __name__ == "__main__":
    print(generate_noise(40))