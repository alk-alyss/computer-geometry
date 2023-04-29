import numpy as np
import open3d as o3d

red = np.array([1,0,0])
green = np.array([0,1,0])
blue = np.array([0,0,1])
yellow = np.array([1,1,0])
magenta = np.array([1,0,1])
cyan = np.array([0,1,1])
black = np.array([0,0,0])
white = np.array([1,1,1])

def downsample(point_cloud, a):

    points = np.asarray(point_cloud.points)
    N = np.shape(points)[0]

    indices = np.arange(N)
    M = N // a
    indices = np.random.choice(indices, M, replace = False)

    points = points[indices,:]

    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud, M

def get_center(point_cloud):

    points = np.asarray(point_cloud.points)
    
    center = np.sum(points,axis=0) / points.shape[0]

    return center

def unit_sphere_normalization(point_cloud):

    points = np.asarray(point_cloud.points)

    distances = np.sum(np.square(points),axis=1)

    points = points / np.sqrt(np.max(distances))

    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def translate(point_cloud, translation_vec):

    points = np.asarray(point_cloud.points)

    points += translation_vec

    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

def find_AABB(point_cloud):
    
    points = np.asarray(point_cloud.points)

    minxyz = np.min(points,axis=0)
    maxxyz = np.max(points,axis=0)

    aabb = o3d.geometry.AxisAlignedBoundingBox(minxyz,maxxyz)
    aabb.color = black
    return aabb

def find_nearest_neighbor_exchaustive(points, id):

    indices = np.arange(np.shape(points)[0])
    indices = np.delete(indices, id)

    distances = np.sum(np.square(points[indices,:] - points[id,:]), axis=1)

    return indices[np.argmin(distances)]

def find_k_nearest_neighbors_exchaustive(points, id, K):

    indices = np.arange(np.shape(points)[0])
    indices = np.delete(indices, id)

    distances = np.sum(np.square(points[indices,:] - points[id,:]), axis=1)
    sorted = np.argsort(distances)

    return indices[sorted[:K]]

def find_points_in_radius_exchaustive(points:np.array, id, radius):

    indices = np.arange(np.shape(points)[0])
    indices = np.delete(indices, id)

    distances = np.sum(np.square(points[indices,:] - points[id,:]), axis=1)
    
    return indices[distances < radius]