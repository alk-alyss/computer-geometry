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
def find_principal_components(mesh):
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
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # creating an LineSet with Open3D to visualize the axis
    axies = []
    colors = [red, green, blue]
    for i, principal_component in enumerate(eigenvectors):
        cylinder_radius = 0.5
        cone_radius = cylinder_radius*1.6
        scaling_factor = eigenvalues[i]
        print(scaling_factor)
        cylinder_height = 200 * scaling_factor

        axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=cylinder_height)
        axis.paint_uniform_color(colors[i])
        axis.compute_triangle_normals()

        axis.scale(0.05, np.array([0, 0, 0]))

        vec1 = np.array([0, 0, cylinder_height])
        vec2 = principal_component

        normal = np.cross(vec1, vec2)
        normal = normal / np.linalg.norm(normal)

        angle = np.arccos(vec1.dot(vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2))

        rotationVector = normal * angle

        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotationVector)
        axis.rotate(rotation, center=(0,0,0))

        axies.append(axis)

    return axies

    




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


def fing_mesh_plane_intersection_triangles_efficient(vertices, triangles, plane_vec):

    # adding an extra dim (homogenous)
    vertices = np.concatenate(
        [
            vertices,
            np.ones((vertices.shape[0], 1))
        ],
        axis=1
    )

    # calculating the distance of the points to the plane
    distances = vertices.dot(plane_vec)
    # see if the distance is positive or negative
    distances = distances > 0

    # create a boolean array to store the indexes of the intersection 
    # initially all points are intersection candidates
    intersect_idxs = np.ones(triangles.shape[0], dtype=bool)
    above_idxs = np.zeros(triangles.shape[0], dtype=bool)
    below_idxs = np.zeros(triangles.shape[0], dtype=bool)

    triangles = triangles.reshape(-1)

    distances = distances[triangles]
    distances = distances.reshape((-1, 3))

    # triangles that all vertices are above the plane
    intersect_idxs[distances.all(axis=-1)] = 0
    above_idxs[distances.all(axis=-1)] = 1

    # triangles that all vertices are below the plane
    distances = np.logical_not(distances)
    intersect_idxs[distances.all(axis=-1)] = 0
    below_idxs[distances.all(axis=-1)] = 1
    
    return intersect_idxs, above_idxs, below_idxs


def fing_mesh_plane_intersection_triangles_loop(vertices, triangles, plane_vec):

    intersect_idxs = np.ones(triangles.shape[0], dtype=bool)

    for i, triangle in enumerate(triangles):
        triangle_vertices = vertices[triangle]

        triangle_vertices = np.concatenate(
            [
                triangle_vertices,
                np.ones((triangle_vertices.shape[0], 1))
            ],
            axis = 1
        )

        distances = triangle_vertices.dot(plane_vec)
        distances = distances > 0

        if distances.all():
            intersect_idxs[i] = 0

        distances = np.logical_not(distances)

        if distances.all():
            intersect_idxs[i] = 0

    return intersect_idxs


def fing_mesh_plane_intersection_triangles_loop_efficient(vertices, triangles, plane_vec):

    vertices = np.concatenate(
        [
            vertices,
            np.ones((vertices.shape[0], 1))
        ],
        axis=1
    )

    # calculating the distance of the points to the plane
    distances = vertices.dot(plane_vec)
    # see if the distance is positive or negative
    distances = distances > 0

    intersect_idxs = np.ones(triangles.shape[0], dtype=bool)

    for i, triangle in enumerate(triangles):
        triangle_distances = distances[triangle]

        if triangle_distances.all():
            intersect_idxs[i] = 0

        triangle_distances = np.logical_not(triangle_distances)

        if triangle_distances.all():
            intersect_idxs[i] = 0

    return intersect_idxs


def get_xz_plane():
    # Create a plane model
    plane = o3d.geometry.TriangleMesh.create_box(width=1.0, height= 0.0001, depth=1.0)
    plane.compute_triangle_normals()
    plane_center = get_center(plane)
    plane = translate(plane, -plane_center)
    plane.paint_uniform_color(red)

    return plane