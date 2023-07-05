import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import KeyEvent
from open3d.visualization.rendering import Camera
import sys
import os
import numpy as np
import utility as U
import platform
from scipy.sparse import csr_matrix
from model import Model
from typing import List

isMacOS = (platform.system() == "Darwin")

class AppWindow:

    MENU_OPEN = 1
    MENU_QUIT = 2

    def __init__(self, width, height):

        resource_path = gui.Application.instance.resource_path
        self.w_width = width
        self.w_height = height

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

        self.model:Model = None

        self.event_type = None

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
        self.filename = filename
        self.load(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def load(self, path):

        self.model = Model(path)

        self._redraw_scene()

        #reconfiguring camera
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_layout(self, layout_context):

        self._scene.frame = self.window.content_rect

    def _on_key_pressed(self, event):

        if event.type == KeyEvent.Type.UP:
            return gui.Widget.EventCallbackResult.HANDLED

        match event.key:
            case 114: #R - reset geometry
                self._reset_geometry()
            case 103: #G - apply gaussian noise to the mesh
                self._apply_noise()
            case 112: #N - apply perlin noise to the mesh
                self._apply_noise(perlin=True)
            case 115: #S - simplify mesh
                self._simplify_mesh()
                self.event_type = "simplify"
            case 99: #C - find similar coatings
                self._similar_coatings()
            case 111: #O - find similar objects
                self._similar_objects()
            case 118: #V - eigenvector visualization mode
                self._show_eigenvector()
                self.event_type = "visualize"

        arrowKeyPressed = True

        simplify = self.event_type == "simplify"

        match event.key:
            case 265: #up arrow - next eigenvector
                self.model.inc_eigenvector(simplify)
            case 266: #down arrow - previous eigenvector
                self.model.dec_eigenvector(simplify)
            case 263: #left arrow - go to lowest eigenvector
                self.model.min_eigenvector(simplify)
            case 264: #right arrow - go to highest eigenvector
                self.model.max_eigenvector(simplify)
            case _: # default
                arrowKeyPressed = False

        if arrowKeyPressed:
            match self.event_type:
                case "visualize":
                    self._show_eigenvector()
                case "simplify":
                    self._simplify_mesh(self.model.current_eigenvector)

        return gui.Widget.EventCallbackResult.HANDLED

    def _set_projection(self):
        self._scene.scene.camera.set_projection(
            Camera.Projection(1), -2.0, 2.0, -2.0, 2.0, 0.1, 100.0
        )

    def _reset_geometry(self):

        self.model.reset_geometry()

        self._redraw_scene()

    def _redraw_scene(self):

        #clearing scene
        self._scene.scene.clear_geometry()

        self._scene.scene.add_geometry(f"__model__", self.model.geometry, self.matlit)

    def _no_model(self):
        print("There is no mesh in the scene")

    def _show_eigenvector(self):

        if self.model is None:
            self._no_model()
            return

        if self.event_type == "simplify":
            self.model.min_eigenvector()

        self.model.show_eigenvector()

        self._redraw_scene()

    def _apply_noise(self, perlin=False):
        if self.model is None:
            self._no_model
            return

        print(f"Applying {'perlin' if perlin else 'gaussian'} noise to mesh...")

        self.model.apply_noise(perlin=perlin)

        self._redraw_scene()

        print("done")

    def _simplify_mesh(self, keep_count=10):
        if self.model is None:
            self._no_model
            return

        print("Simplifying mesh...")

        self.model.simplify_mesh(keep_count)

        self._redraw_scene()
        
        print("done")

    def _similar_coatings(self):

        if self.model is None:
            self._no_model()
            return

        print("Finding similar coatings...")

        high_eigenvectors = self.model.get_eigenvectors(high=True)

        print("done")

    def _similar_objects(self):

        if self.model is None:
            self._no_model()
            return

        print("Finding similar objects...")

        low_eigenvectors = self.model.get_eigenvectors()

        print("done")

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
