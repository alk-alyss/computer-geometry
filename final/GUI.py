import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent, KeyEvent
from open3d.visualization.rendering import Camera
import sys
import os
import numpy as np
import utility as U
import platform
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eigh

isMacOS = (platform.system() == "Darwin")

class AppWindow:

    MENU_OPEN = 1
    MENU_QUIT = 2

    def __init__(self, width, height):

        resource_path = gui.Application.instance.resource_path
        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window("Test", width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)

        #initializing menubar
        self._init_menubar()

        #basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)

        #geometry container for future reference
        self.geometry = None
        self.vertices = None
        self.triangles = None
        self.tree = None

        self.laplacian = None

        self.eigenvalues = None
        self.eigenvectors = None
        self.current_eigenvector = 0

        #materials
        self.matlit = rendering.MaterialRecord()
        self.matlit.shader = "defaultLit"
        self.matlit.point_size = 6

    def _init_menubar(self):

        if gui.Application.instance.menubar is None:

            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)

            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)

            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)

            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _preprocess(self, m):

        vertices, triangles = np.asarray(m.vertices), np.asarray(m.triangles)

        #centering
        vertices = vertices - vertices.mean(0)

        #unit_sphere_normalization
        norm = np.max((vertices * vertices).sum(-1))
        vertices = vertices / np.sqrt(norm)

        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

    def _find_match(self, query):

        if self.geometry is not None:

            ind = 0
            if self.tree is not None:
                _, ind, _ = self.tree.search_knn_vector_3d(query, 1)
                ind = int(np.asarray(ind)[0])
                self.selected_vertex = ind
                return self.vertices[ind]

            else:
                d = self.vertices - query
                d = np.argmin((d * d).sum(-1))
                self.selected_vertex = d
                return self.vertices[ind]

    def load(self, path):

        #clearing scene
        self._scene.scene.clear_geometry()

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
        self.wire = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
        self.pc = o3d.geometry.PointCloud(self.geometry.vertices)

        #setting vertex and triangle data for easy access
        self.vertices = np.asarray(self.geometry.vertices)
        self.triangles = np.asarray(self.geometry.triangles)
        self.laplacian = U.random_walk_laplacian(self.triangles)

        #initializing kd-tree for quick searches
        self.tree = o3d.geometry.KDTreeFlann(self.geometry)

        #adding mesh to the scene and reconfiguring camera
        self._scene.scene.add_geometry("__model__", self.geometry, self.matlit)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_layout(self, layout_context):

        r = self.window.content_rect
        self._scene.frame = r

    def _on_key_pressed(self, event):

        if event.type == KeyEvent.Type.UP:
            return gui.Widget.EventCallbackResult.HANDLED

        # print("key pressed: ", event.key)

        #R key - reset geometry and redraw scene
        if event.key == 114:
            self._reset_geometry()
            self._redraw_scene()
            return gui.Widget.EventCallbackResult.HANDLED

        #N key - apply noise to the mesh
        elif event.key == 110:
            self._apply_noise()
            return gui.Widget.EventCallbackResult.HANDLED

        #S key - simplify mesh
        elif event.key == 115:
            self._simplify_mesh()
            return gui.Widget.EventCallbackResult.HANDLED

        #C key - find similar coatings
        elif event.key == 99:
            self._similar_coatings()
            return gui.Widget.EventCallbackResult.HANDLED

        #O key - find similar objects
        elif event.key == 111:
            self._similar_objects()
            return gui.Widget.EventCallbackResult.HANDLED

        #V key - eigenvector visualization mode
        elif event.key == 118:
            self._calc_eigenvectors()
            print("eigenvectors calculated.")
            self._show_eigenvector()
            return gui.Widget.EventCallbackResult.HANDLED

        #left or bottom arrow keys - decrease eigenvector counter
        elif event.key == 263 or event.key == 266:
            self.current_eigenvector = self.current_eigenvector -1 if self.current_eigenvector > 0 else 0
            print("current eigenvector: ", self.current_eigenvector)
            self._show_eigenvector()
            return gui.Widget.EventCallbackResult.HANDLED

        #right or up arrow keys - increase eigenvector counter
        elif event.key == 264 or event.key == 265:
            self.current_eigenvector = self.current_eigenvector +1 if self.current_eigenvector < self.eigenvectors.shape[0]-1 else self.eigenvectors.shape[0]-1
            print("current eigenvector: ", self.current_eigenvector)
            self._show_eigenvector()
            return gui.Widget.EventCallbackResult.HANDLED

        #enter key - placeholder
        elif event.key == 10:
            return gui.Widget.EventCallbackResult.HANDLED

        else:
            return gui.Widget.EventCallbackResult.IGNORED

    def _set_projection(self):
        self._scene.scene.camera.set_projection(
            Camera.Projection(1), -2.0, 2.0, -2.0, 2.0, 0.1, 100.0
        )

    def _reset_geometry(self):

        self.geometry = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.triangles)
        )

    def _redraw_scene(self):

        #clearing scene
        self._scene.scene.clear_geometry()

        self._scene.scene.add_geometry("__model__", self.geometry, self.matlit)

    def _calc_eigenvectors(self):

        if self.laplacian is not None:

            L = self.laplacian

            #performing eigendecomposition
            vals, vecs = eigh(L)

            #sorting according to eigenvalue
            self.eigenvalues = np.argsort(vals)
            self.eigenvectors = vecs[:, self.eigenvalues]

    def _show_eigenvector(self):

        if self.eigenvectors is not None:

            # colors = np.zeros_like(self.vertices)

            scalars = self.eigenvectors[:,self.current_eigenvector]
            scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

            # colors[:,0] = scalars
            colors = U.sample_colormap(scalars)

            self.geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
            self._redraw_scene()

    def _apply_noise(self):
        pass

    def _simplify_mesh(self):
        pass

    def _similar_coatings(self):
        pass

    def _similar_objects(self):
        pass

def main():

    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()

if __name__ == "__main__":

    main()