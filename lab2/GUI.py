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

        #adding simple geometry
        self.geometries = []
        self.geometries_with_background = []
        #selected geometry
        self.selected_geometry = None
        #query point
        self.query_point = None
        #triangles that contain violations
        self.violations = []
        #outside triangles
        self.out_counter = 1
        #dual graph
        self.dual = []

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

    def _erase_temporary_geometries(self):

        #query point
        self._scene.scene.remove_geometry("query")
        #selected triangle
        self._scene.scene.remove_geometry("selected")
        #circumcircle
        self._scene.scene.remove_geometry("circumcircle")
        #delauney violation
        self._scene.scene.remove_geometry("violation")
        #selected triangle vertices
        self._scene.scene.remove_geometry("v1")
        self._scene.scene.remove_geometry("v2")
        self._scene.scene.remove_geometry("v3")

        #emptying buffers
        self.violations = []

    def _redraw_scene(self):

        base_tri = U.Triangle2D(np.array([
            [-0.5, -0.5],
            [ 0.5, -0.5],
            [ 0,    0.5]
        ]))

        self._scene.scene.add_geometry("tri", base_tri.as_o3d_mesh_fb, defaultUnlit)

        for tri in self.geometries:
            self._scene.scene.add_geometry(tri.id, tri.as_o3d_lineset, defaultUnlit)

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

        #S key
        elif event.key == 115:

            if self.selected_geometry is None or self.query_point is None:
                return gui.Widget.EventCallbackResult.IGNORED

            #unlit material
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            #splitting into three triangles
            self.geometries = self.geometries + self.selected_geometry.split(self.query_point)
            self._scene.scene.add_geometry(self.geometries[-1].id, self.geometries[-1].as_o3d_lineset, mat)
            self._scene.scene.add_geometry(self.geometries[-2].id, self.geometries[-2].as_o3d_lineset, mat)
            self._scene.scene.add_geometry(self.geometries[-3].id, self.geometries[-3].as_o3d_lineset, mat)

            for i, tri in enumerate(self.geometries):
                if tri.id == self.selected_geometry.id:
                    self.geometries.pop(i)
            self._scene.scene.remove_geometry(self.selected_geometry.id)

            #resetting query point and selected geometry
            self.selected_geometry = None
            self.query_point = None

            return gui.Widget.EventCallbackResult.HANDLED
        
        #V key
        elif event.key == 118:

            if self.selected_geometry is None:
                return gui.Widget.EventCallbackResult.IGNORED

            #if violations already exist
            if self.violations:
                return gui.Widget.EventCallbackResult.IGNORED

            #unlit material
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            #searching list for neighboring triangles
            neighbors = []
            for tri in self.geometries:
                if tri.has_common_edge(self.selected_geometry):
                    neighbors.append(tri)

            #grab circumcircle
            circumcircle = self.selected_geometry.circumcircle

            #iterate neighbors
            for n in neighbors:

                #find non-common vertex
                ncv = n.non_common_vertex(self.selected_geometry)

                #if the circumcircle of the current triangle contains it then there is a violation
                if circumcircle.contains(ncv):
                    self._scene.scene.add_geometry("violation",
                        create_sphere(ncv[0], ncv[1], radius = 0.01).paint_uniform_color(U.red),
                        mat
                    )

                    self.violations.append(self.selected_geometry)
                    self.violations.append(self._find_triangle_by_id(n.id))

            return gui.Widget.EventCallbackResult.HANDLED
        
        #F key
        elif event.key == 102:
            
            if self.selected_geometry is None:
                return gui.Widget.EventCallbackResult.IGNORED
            
            if not self.violations:
                return gui.Widget.EventCallbackResult.IGNORED

            assert len(self.violations) == 2

            #removing violating triangles from the scene 
            self._scene.scene.remove_geometry(self.violations[0].id)
            self._scene.scene.remove_geometry(self.violations[1].id)

            #flipping the edge, thereby creating 2 new triangles
            t1, t2 = U.flip_edge(self.violations[0], self.violations[1])

            #adding the new triangles to the list
            self.geometries.append(t1)
            self.geometries.append(t2)

            #unlit material
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            #adding the new triangles
            self._scene.scene.add_geometry(t1.id, t1.as_o3d_lineset, mat)
            self._scene.scene.add_geometry(t2.id, t2.as_o3d_lineset, mat)

            self.selected_geometry = None
            self.violations = []
            self.query_point = None

            #cleaning and redrawing the scene to fix visual bug
            self._scene.scene.clear_geometry()
            self._redraw_scene()

            return gui.Widget.EventCallbackResult.HANDLED

        #T key
        elif event.key == 116 and event.type == KeyEvent.Type.UP:

            #unlit material
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            for i, tri in enumerate(self.geometries):
                barycenter = create_sphere(tri.barycenter[0], tri.barycenter[1], 0.01).paint_uniform_color(U.blue)
                self._scene.scene.add_geometry(f"barycenter{i}", barycenter, mat)

                for j, other_tri in enumerate(self.geometries):
                    if j <= i: continue

                    if tri.has_common_edge(other_tri):
                        self.dual.append(U.Line2D(tri.barycenter, other_tri.barycenter))
                        self._scene.scene.add_geometry(f"line{i}{j}", self.dual[-1]._o3d, mat)

            return gui.Widget.EventCallbackResult.HANDLED

        #P key
        elif event.key == 112:

            if self.selected_geometry is None:
                return gui.Widget.EventCallbackResult.IGNORED

            #unlit material
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"

            v1 = create_sphere(self.selected_geometry.v1[0], self.selected_geometry.v1[1], 0.01).paint_uniform_color(U.blue)
            v2 = create_sphere(self.selected_geometry.v2[0], self.selected_geometry.v2[1], 0.01).paint_uniform_color(U.blue)
            v3 = create_sphere(self.selected_geometry.v3[0], self.selected_geometry.v3[1], 0.01).paint_uniform_color(U.blue)

            self._scene.scene.add_geometry("v1", v1, mat)
            self._scene.scene.add_geometry("v2", v2, mat)
            self._scene.scene.add_geometry("v3", v3, mat)

            return gui.Widget.EventCallbackResult.HANDLED

        #R key
        elif event.key == 114:

            
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

                #creating initial triangle
                tri = U.Triangle2D(np.array([
                    [-0.5, -0.5],
                    [ 0.5, -0.5],
                    [ 0,    0.5]
                ]))

                #adding it to the list of geometries
                self.geometries.append(tri)
                self.geometries_with_background.append(tri)

                #add the root of the tree to the scene
                self._scene.scene.add_geometry("tri",
                                               tri.as_o3d_mesh_fb,
                                               defaultUnlit)
                
                tri.set_color(np.array([1, 1, 1]))
                #add a second copy for visualziation purposes
                self._scene.scene.add_geometry("tril",
                                               tri.as_o3d_lineset,
                                               defaultUnlit)
            
            else:

                #re-adding previous selection to the list, since it was unused
                # if self.selected_geometry is not None:
                #     self.geometries.append(self.selected_geometry)

                #initializing sphere
                xy = window_to_scene_coords(event.x, event.y, self._scene)
                sph = create_sphere(xy[0], xy[1], radius = 0.005)

                #unlit for looking like a point
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"

                #removing previous sphere and adding current sphere
                self._erase_temporary_geometries()
                self._scene.scene.add_geometry("query", sph, mat)

                #Finding which triangle the point lies inside
                query_point = np.array(xy[:2])
                self.query_point = query_point
                target_tri = self._which_triangle(query_point)  

                #if no triangle was hit, then the clicked location is outside of the main triangle
                if target_tri is None:
                    return gui.Widget.EventCallbackResult.CONSUMED  
                
                print("Target triangle: ", target_tri.id)

                #highlighting the selected triangle
                self.selected_geometry = target_tri
                self._scene.scene.add_geometry("selected", 
                                               self.selected_geometry.as_o3d_lineset.paint_uniform_color(np.array([1,1,0])), 
                                               mat)

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