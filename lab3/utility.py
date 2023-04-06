import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

red = np.array([1,0,0])
green = np.array([0,1,0])
blue = np.array([0,0,1])
yellow = np.array([1,1,0])
magenta = np.array([1,0,1])
cyan = np.array([0,1,1])
black = np.array([0,0,0])
white = np.array([1,1,1])


# Task 1.1: Find center mass
############################################
def get_center(mesh):
    vertices = np.asarray(mesh.vertices)
    
    center = vertices.mean(0)

    return center


# Task 1.2: Normalize a mesh to fit inside the unit sphere 
############################################
def unit_sphere_normalization(mesh):
    vertices = np.asarray(mesh.vertices)

    vertices /= np.max(np.linalg.norm(vertices, axis=1))

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


# Task 1.3: Translate a mesh 
############################################
def translate(mesh, translation_vec):
    vertices = np.asarray(mesh.vertices)

    vertices += translation_vec

    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    return mesh



def deg2rad(theta):
    return theta * np.pi / 180


# Task 1.4: Rotate an set of vertices around the z-axis
############################################
def rotate_around_z_np(vertices, theta):

    theta  = deg2rad(theta)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    r = Rotation.from_euler('z', theta, degrees=True)
    rotation_matrix = r.as_matrix
    
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices    


def rotate_around_z(mesh, theta):
    vertices = np.asarray(mesh.vertices)

    rotated_vertices = rotate_around_z_np(vertices, theta)

    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)

    return mesh

# Task 2: Find the axis aligned bounding box
############################################
def find_AABB(mesh):

    vertices = np.asarray(mesh.vertices)

    minxyz = np.min(vertices, axis=0)
    maxxyz = np.max(vertices, axis=0)

    aabb = o3d.geometry.AxisAlignedBoundingBox(minxyz, maxxyz)
    aabb.color = black
    return aabb
    
# Task 3: Calculate the principal component of the mesh
############################################
def find_principal_component(mesh):
    vertices = np.asarray(mesh.vertices)
    # TODO
    # Calculate the covariance matrix
    covariance_matrix = np.cov(vertices, rowvar=False)
    # Note: Setting rowvar=False because each column represents a 
    #       variable, and each row a different observation (point) 

    # compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # principal component is the eigenvector corresponding to the largest eigenvalue
    principal_component = sorted_eigenvectors[0]

    # creating an LineSet with Open3D to visualize the axis
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector([np.array([0.0, 0.0, 0.0]), principal_component])
    axis.lines = o3d.utility.Vector2iVector([[0, 1]])
    axis.colors = o3d.utility.Vector3dVector([red])

    return axis

    




# Task 4: Mesh Plane intersection
############################################
def fing_mesh_plane_intersection_triangles(vertices, triangles, plane_vec):

    # get the number of triangles in the mesh
    num_triangles = triangles.shape[0]

    # unroll the vertices, so that continous triad of vertices 
    # will belong in the same triangle
    triangles = triangles.reshape(-1)
    unrolled_vertices = vertices[triangles]

    # adding an extra dim (homogenous)
    unrolled_vertices = np.concatenate(
        [
            unrolled_vertices,
            np.ones((unrolled_vertices.shape[0], 1))
        ],
        axis=1
    )
    print(unrolled_vertices.shape)

    # calculating the distance of the points to the plane
    unrolled_distances = unrolled_vertices.dot(plane_vec)
    # see if the distance is positive or negative
    unrolled_distances = unrolled_distances > 0


    # reshaping the triangles back to original shape
    triangles = triangles.reshape((-1, 3))
    unrolled_distances = unrolled_distances.reshape((-1, 3))
    print(unrolled_distances.shape)

    # create a boolean array to store the indexes of the intersection 
    # initially all points are intersection candidates
    intersect_idxs = np.ones(num_triangles, dtype=bool)

    # triangles that all vertices are above the plane
    intersect_idxs[unrolled_distances.all(axis=-1)] = 0

    # triangles that all vertices are below the plane
    unrolled_distances = np.logical_not(unrolled_distances)
    intersect_idxs[unrolled_distances.all(axis=-1)] = 0
    
    return intersect_idxs




def get_xz_plane():
    # Create a plane model
    plane = o3d.geometry.TriangleMesh.create_box(width=1.0, height= 0.0001, depth=1.0)
    plane.compute_vertex_normals()
    plane_center = get_center(plane)
    plane = translate(plane, -plane_center)
    plane.paint_uniform_color(red)

    return plane