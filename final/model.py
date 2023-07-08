import open3d as o3d
import numpy as np
from scipy.sparse.linalg import eigsh
from copy import deepcopy

import utility as U

class Model:
    def __init__(self, path):
        #geometry container for future reference
        self.filename:str = path
        self.eigenvalues:np.ndarray = None
        self.eigenvectors:np.ndarray = None
        self.current_eigenvector = 0

        #reading geometry type
        geometry_type = o3d.io.read_file_geometry_type(path)

        #checking the type of geometry
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            self.geometry = o3d.io.read_triangle_model(path).meshes[0].mesh

        if self.geometry is None:
            print("[Info]", path, "appears to not be a triangle mesh")
            return

        #preprocessing and setting geometry
        self.geometry = self._preprocess(self.geometry)

        #setting vertex and triangle data for easy access
        self.vertices = deepcopy(np.asarray(self.geometry.vertices))
        self.triangles = deepcopy(np.asarray(self.geometry.triangles))

        #initializing kd-tree for quick searches
        self.tree = o3d.geometry.KDTreeFlann(self.geometry)

        self.calculate_normals()


    def _preprocess(self, m):
        print(len(m.triangles))

        m = m.simplify_quadric_decimation(target_number_of_triangles=20000)

        vertices, triangles = np.asarray(m.vertices), np.asarray(m.triangles)

        #centering
        vertices = vertices - vertices.mean(0)

        #unit_sphere_normalization
        norm = np.max((vertices * vertices).sum(-1))
        vertices = vertices / np.sqrt(norm)

        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

    def _calc_eigenvectors(self, count=100):
        '''
            Calculate count amount of low valued eigenvectors,
            count amount of high valued eigenvectors
            and store them in ascending order
        '''

        if self.eigenvectors is not None:
            return

        print("Calculating eigenvectors...")

        L = U.laplacian(self.triangles, type="graph")

        #performing eigendecomposition
        vals, vecs = eigsh(L, k=count*2, which="BE")

        #sorting according to eigenvalue
        self.eigenvalues = np.argsort(vals)
        self.eigenvectors = vecs[:, self.eigenvalues]

        print("done")

    def calculate_normals(self):
        self.geometry.compute_triangle_normals()
        self.geometry.compute_vertex_normals()

    def reset_geometry(self):
        self.geometry = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.triangles)
        )

        self.calculate_normals()

    def get_eigenvalues(self, high=False) -> np.ndarray:
        '''
            Return one end of the eigenvalue range.
            By default return the lowest eigenvalues.
            If high==True return the highest eigenvalues.
        '''

        self._calc_eigenvectors()

        start = 0
        end = self.eigenvalues.shape[0]

        eigs_count = end//2

        if high:
            start = eigs_count
        else:
            end = eigs_count

        return self.eigenvalues[start:end]
    
    def get_eigenvectors(self, high=False) -> np.ndarray:
        '''
            Return one end of the eigenvectors range.
            By default return the lowest eigenvectors.
            If high==True return the highest eigenvectors.
        '''

        self._calc_eigenvectors()

        start = 0
        end = self.eigenvectors.shape[1]

        eigs_count = end//2

        if high:
            start = eigs_count
        else:
            end = eigs_count

        return self.eigenvectors[:, start:end]

    def inc_eigenvector(self, simplify=False):
        self.current_eigenvector += 1

        max_eig = self._max_eig(simplify)
        if self.current_eigenvector > max_eig:
            self.current_eigenvector = max_eig

    def dec_eigenvector(self, simplify=False):
        self.current_eigenvector -= 1

        min_eig = self._min_eig(simplify)
        if self.current_eigenvector < min_eig:
            self.current_eigenvector = min_eig

    def min_eigenvector(self, simplify=False):
        self.current_eigenvector = self._min_eig(simplify)

    def max_eigenvector(self, simplify=False):
        self.current_eigenvector = self._max_eig(simplify)

    def _min_eig(self, simplify=False):
        min_eig = 0
        if simplify:
            min_eig = 3

        return min_eig

    def _max_eig(self, simplify=False):
        max_eig = len(self.eigenvalues)
        if simplify:
            max_eig //= 2

        return max_eig-1

    def show_eigenvector(self):
        self._calc_eigenvectors()

        print(f"Current eigenvector: {self.eigenvalues[self.current_eigenvector]}")

        scalars = self.eigenvectors[:,self.current_eigenvector]
        scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

        colors = U.sample_colormap(scalars)

        self.geometry.vertex_colors = o3d.utility.Vector3dVector(colors)

    def apply_noise(self, noise_factor=3, perlin=False):

        new_vecs = np.asarray(self.geometry.vertices)

        # Generate noise
        if perlin:
            noise = U.generate_perlin_noise(self.vertices)
        else:
            noise = U.generate_gaussian_noise(self.vertices.shape[0])

        # Calculate delta vectors
        delta = (noise*new_vecs.T).T

        # Calculate new vectors from original and delta vectors
        new_vecs = new_vecs + (noise_factor/100)*delta

        # Display new vectors
        self.geometry.vertices = o3d.utility.Vector3dVector(new_vecs)

    def simplify_mesh(self, keep_count=10):

        low_eigenvectors = self.get_eigenvectors()

        #forming the eigenvector matrix with only the significant components
        print(f"Keep {keep_count} eigenvectors")

        transformation_matrix = low_eigenvectors[:, :keep_count]
        new_vecs = transformation_matrix @ (transformation_matrix.T @ self.vertices)

        #setting the vertices to be the filtered ones
        self.geometry.vertices = o3d.utility.Vector3dVector(new_vecs)

        self.current_eigenvector = keep_count