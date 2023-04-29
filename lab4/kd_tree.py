from cmath import inf
import numpy as np
import open3d as o3d
import heapq as heapq

class kd_node:

    def __init__(self, index, left_child, right_child):

        self.index = index
        
        self.left_child = left_child
        self.right_child = right_child
    
    def depth_first_search(self):

        indices = depth_first_search(self, indices = [])
        return indices

class kd_tree:

    def __init__(self, points:np.array):

        self.root = self.build_kd_tree(points) 

    def build_kd_tree(self, points:np.array): 

        root = build_kd_tree(points, dim=points.shape[1], indices=np.arange(len(points)), level=0)
        return root

    def get_nodes_of_level(self, level):

        nodes = get_nodes_of_level(self.root, level, nodes = [])
        return nodes

    def find_nearest_neighbor(self, points, id):

       _, istar = find_nearest_neighbor(points, id, self.root, points.shape[1])
       return istar

    def find_points_in_radius(self, points, id, radius):

        indices = find_points_in_radius(points, id, radius, self.root, points.shape[1])
        return indices

    def find_k_nearest_neighbors(self, points, id, K):

        heap, _ = find_k_nearest_neighbors(points, id, K)

        indices = []
        while heap:
            _, index = heapq.heappop(heap)
            indices.append(index)

        return indices

def build_kd_tree(points:np.array, dim, indices, level):

    if len(indices) == 0:
        return
    
    axis = level % dim

    order = np.argsort(points[indices, axis])
    sorted_indices = indices[order]

    median_idx = (len(indices)-1) // 2
    index = sorted_indices[median_idx]

    indices_left = sorted_indices[:median_idx]
    indices_right = sorted_indices[median_idx+1:] 

    left_child = build_kd_tree(points, dim, indices=indices_left, level=level+1)
    right_child = build_kd_tree(points, dim, indices=indices_right, level=level+1)

    return kd_node(index = index, left_child =  left_child, right_child = right_child)

def get_nodes_of_level(root:kd_node, level, nodes):

    if root == None:
        return nodes

    if level == 0:
        nodes.append(root)

    else:
        nodes = get_nodes_of_level(root.left_child, level-1, nodes)
        nodes = get_nodes_of_level(root.right_child, level-1, nodes)

    return nodes

def depth_first_search(root:kd_node, indices):

    if root == None:
        return indices
    
    indices = depth_first_search(root.left_child, indices)
    indices = depth_first_search(root.right_child, indices)

    indices.append(root.index)

    return indices

def find_nearest_neighbor(points:np.array, id, root:kd_node, dim, level=0, dstar=inf, istar=-1):

    if root == None:
        return dstar, istar

    axis = level % dim
    d_ = points[id, axis] - points[root.index, axis]

    is_on_left = d_ < 0
    
    if is_on_left:
        dstar, istar = find_nearest_neighbor(points, id, root.left_child, dim, level+1, dstar, istar)

        if d_**2 < dstar:
            dstar, istar = find_nearest_neighbor(points, id, root.right_child, dim, level+1, dstar, istar)

    else:
        dstar, istar = find_nearest_neighbor(points, id, root.right_child, dim, level+1, dstar, istar)

        if d_**2 < dstar:
            dstar, istar = find_nearest_neighbor(points, id, root.left_child, dim, level+1, dstar, istar)

    if root.index != id:
        d = np.linalg.norm(points[root.index, :] - points[id, :])

        if d < dstar:
            dstar = d
            istar = root.index

        
    return dstar, istar
    
def find_points_in_radius(points:np.array, id, radius, root:kd_node, dim, level=0, indices=[]):
    
    if root == None:
        return indices

    axis = level % dim
    d_ = points[id, axis] - points[root.index, axis]

    is_on_left = d_ < 0
    
    if is_on_left:
        indices = find_points_in_radius(points, id, radius, root.left_child, dim, level+1)

        if d_**2 < radius:
            indices = find_points_in_radius(points, id, radius, root.right_child, dim, level+1)

    else:
        indices = find_points_in_radius(points, id, radius, root.right_child, dim, level+1)

        if d_**2 < radius:
            indices = find_points_in_radius(points, id, radius, root.left_child, dim, level+1)

    if root.index != id:
        d = np.linalg.norm(points[root.index, :] - points[id, :])

        if d < radius:
            indices.append(root.index)

        
    return indices

def find_k_nearest_neighbors(points:np.array, id, root:kd_node, dim, level=0, heap=[], dstar=inf):
        
    if root == None:
        return heap, dstar

    axis = level % dim
    d_ = points[id, axis] - points[root.index, axis]

    is_on_left = d_ < 0
    
    if is_on_left:
        dstar, istar = find_nearest_neighbor(points, id, root.left_child, dim, level+1, dstar, istar)

        if d_**2 < dstar:
            dstar, istar = find_nearest_neighbor(points, id, root.right_child, dim, level+1, dstar, istar)

    else:
        dstar, istar = find_nearest_neighbor(points, id, root.right_child, dim, level+1, dstar, istar)

        if d_**2 < dstar:
            dstar, istar = find_nearest_neighbor(points, id, root.left_child, dim, level+1, dstar, istar)

    if root.index != id:
        d = np.linalg.norm(points[root.index, :] - points[id, :])

        if d < dstar:
            dstar = d
            istar = root.index

        
    return heap, dstar