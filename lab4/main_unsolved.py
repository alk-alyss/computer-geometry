import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent, KeyEvent
from open3d.visualization.rendering import Camera
import numpy as np
import utility as util
import kd_tree

task = 2

# shaders

defaultUnlit = rendering.MaterialRecord()
defaultUnlit.shader = "defaultUnlit"
defaultUnlit.point_size = 3

unlitLine = rendering.MaterialRecord()
unlitLine.shader = "unlitLine"
unlitLine.line_width = 3

class AppWindow:

    def __init__(self, width, height, window_name="Lab"):

        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window(window_name, width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([1,1,1,1])

        # basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        # self._scene.set_on_mouse(self._on_mouse_pressed)

        #set up camera
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.look_at(center, center - [0, 0, 12], [0, 1, 0])
        
        self.geometries = {}
        self.wireframe_on = False
        self.aabb_on = False
        self.pr_comp_on = False

    def _on_layout(self, layout_context):
        
        r = self.window.content_rect
        self._scene.frame = r

    def add_geometry(self, geometry, name, shader):

        self._scene.scene.add_geometry(name, geometry, shader)
        self.geometries[name] = geometry

    def remove_geometry(self, name):

        self._scene.scene.remove_geometry(name)

    def _on_key_pressed(self, event):

        pass

if __name__=="__main__":

    gui.Application.instance.initialize()

    # initialize GUI
    app = AppWindow(1280, 720)

    # load a mesh 
    mesh = o3d.io.read_triangle_mesh("lab4/Armadillo.ply")
    point_cloud = o3d.geometry.PointCloud(mesh.vertices)
    
    # point cloud preprocessing 

    a = 1
    point_cloud, N = util.downsample(point_cloud, a)
    print(N)

    point_cloud = util.unit_sphere_normalization(util.translate(point_cloud,-util.get_center(point_cloud)))

    points = np.asarray(point_cloud.points)
    C = np.ones((N,1)) * [0,0,0]

    # build kd tree

    kdtree = kd_tree.kd_tree(points)
    
    # tasks

    if task == 0:

        level = 6
        nodes = kdtree.get_nodes_of_level(level)
        cols = np.random.random((len(nodes),3))
        
        for (node,col) in zip(nodes,cols):

            indices = node.depth_first_search()
            C[indices,:] = col
            
    if task == 1:
        
        # task 1 : find nearest neighbor
        
        id = np.random.randint(N)

        istar = kdtree.find_nearest_neighbor(points,id)
        # istar_ = util.find_nearest_neighbor_exchaustive(points, id)

        C[[id]] = [1,0,0]
        C[[istar]] = [0,1,0]

    elif task == 2:

        # task 2: find points in radius 

        id = np.random.randint(N)

        radius = 0.2**2
        indices = kdtree.find_points_in_radius(points, id, radius)
        # indices_ = util.find_points_in_radius_exchaustive(points, id, radius)

        C[indices+[id],:] = [0,1,0]

        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(np.sqrt(radius)))
        sphere.translate(points[id,:])
        app.add_geometry(sphere.paint_uniform_color([0,0,0]), "sphere", defaultUnlit)

    elif task == 3:

        # task 3: find k nearest neighbors 

        id = np.random.randint(N)

        K = 20
        indices = kdtree.find_k_nearest_neighbors(points, id, K)
        # indices_ = util.find_k_nearest_neighbors_exchaustive(points, id, K)

        C[indices+[id],:] = [0,1,0]

    point_cloud.colors = o3d.utility.Vector3dVector(C)
    app.add_geometry(point_cloud, "point_cloud", defaultUnlit)

    gui.Application.instance.run()


