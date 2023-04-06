import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent, KeyEvent
from open3d.visualization.rendering import Camera
import sys
import os
import numpy as np
from time import sleep
import utility as U


def create_sphere(x, y, radius):

    m = o3d.geometry.TriangleMesh.create_sphere(radius = radius, resolution = 30).translate(np.array([x,y,0.5]))
    
    return m

def window_to_scene_coords(x, y, scene):

    # x = x - scene.frame.x
    # y = y - scene.frame.y

    world = scene.scene.camera.unproject(
        x, y, 0, scene.frame.width, scene.frame.height
    )    

    return world

defaultUnlit = rendering.MaterialRecord()
defaultUnlit.shader = "defaultUnlit"

class AppWindow:

    def __init__(self, width, height):

        resource_path = gui.Application.instance.resource_path
        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window("Test", width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)

        #basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        self._scene.set_on_mouse(self._on_mouse_pressed)

        #set up camera
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.look_at(center, center - [0, 0, 12], [0, 1, 0])

        self.triangles = []
        self.points = []

    def _on_layout(self, layout_context):
        
        r = self.window.content_rect
        self._scene.frame = r

    def _which_triangle(self, point):

        for i, tri in enumerate(self.geometries):
            if tri.contains(point):
                return tri
        
    def _find_triangle_by_id(self, id):

        for i, tri in enumerate(self.geometries):
            if tri.id == id:
                return tri

    def _erase_points(self):

        self._scene.scene.remove_geometry("v1")
        self._scene.scene.remove_geometry("v2")
        self._scene.scene.remove_geometry("v3")

    def _on_key_pressed(self, event):

        print(event.key)

        #C key
        if event.key == 99:

            if self.selected_geometry is None:
                return gui.Widget.EventCallbackResult.IGNORED
            
            #unlit material
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            #creating circumcircle
            circumcircle = self.selected_geometry.circumcircle

            #adding it to the scene
            self._scene.scene.add_geometry("circumcircle", circumcircle.as_o3d_lineset.paint_uniform_color(np.array([1,0,0])), mat)

            return gui.Widget.EventCallbackResult.HANDLED

        
        else:
            return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_pressed(self, event):

        if event.type == MouseEvent.Type.BUTTON_DOWN:

            #if this is the first click
            if self.first_click:

                #set projection to match a 2d context
                self._scene.scene.camera.set_projection(
                    Camera.Projection(1), -1, 1, -1, 1, 0.1, 100
                )

                #falsify click flag
                self.first_click = False

            xy = window_to_scene_coords(event.x, event.y, self._scene)
            sph = create_sphere(xy[0], xy[1], radius = 0.01)

            #unlit for looking like a point
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            self.points.append(xy)
            self._scene.scene.add_geometry(f"v{len(self.points)}", sph, mat)

            if len(self.points) == 3:
                self._erase_points()

                triangle = U.Triangle2D(self.points[0][:2], self.points[1][:2], self.points[2][:2])

                self._scene.scene.add_geometry(f"t{len(self.triangles)}", triangle.as_o3d_lineset, mat)
                self.triangles.append(triangle)

                self.points.clear()
            

            return gui.Widget.EventCallbackResult.CONSUMED
        
        elif event.type == MouseEvent.Type.BUTTON_UP:
            print("mouse button lifted") 
            return gui.Widget.EventCallbackResult.CONSUMED
        
        else:
            return gui.Widget.EventCallbackResult.IGNORED

    def _set_projection(self):
        self._scene.scene.camera.set_projection(
            Camera.Projection(1), -2.0, 2.0, -2.0, 2.0, 0.1, 100.0
        )

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