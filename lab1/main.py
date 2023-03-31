import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent
from open3d.visualization.rendering import Camera
import sys
import os
import numpy as np
import time
import utility as U
from convex_hull_animation import quickhull, jarvis, graham_scan
import threading


def window_to_scene_coords(x, y, scene):

    # x = x - scene.frame.x
    # y = y - scene.frame.y

    world = scene.scene.camera.unproject(
        x, y, 0, scene.frame.width, scene.frame.height
    )

    return world

def sphere_at(xy):

    return o3d.geometry.TriangleMesh.create_sphere(
        radius = 0.02
    ).translate(np.array([xy[0], xy[1], 0]))


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
        self._scene.force_redraw()

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

        #Load initial geometry
        m = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
        # self.vertices = U.unit_sphere_normalization(U.recenter(U.subsample(np.asarray(m.vertices)[:,:2], 4098)))
        self.vertices = U.unit_sphere_normalization(U.recenter(np.asarray(m.vertices)[:,:2]))

        # m = o3d.geometry.TriangleMesh.create_sphere(resolution=30)
        # self.vertices = U.recenter(U.subsample(np.asarray(m.vertices)[:,:2], 2048))
        self.init_shape = U.o3d_pointcloud(self.vertices, center=True)

        #initializing iterator for quickhull
        self.qhull_it = None
        self.convex_hull_points = None

    def _on_layout(self, layout_context):

        r = self.window.content_rect
        self._scene.frame = r

    def _run_convex_hull_step(self, geometries):

        #clear scene
        self._scene.scene.clear_geometry()

        pointMat = rendering.MaterialRecord()
        pointMat.shader = "defaultUnlit"
        lineMat = rendering.MaterialRecord()
        lineMat.shader = "unlitLine"
        lineMat.line_width = 10

        #add new geometries
        for i, g in enumerate(geometries["point"]):
            self._scene.scene.add_geometry(
                f"_p{i}", g, pointMat
            )

        for i, g in enumerate(geometries["line"]):
            self._scene.scene.add_geometry(
                f"_l{i}", g, lineMat
            )

    def _run_convex_hull_animation_step(self):

        #clear scene
        self._scene.scene.clear_geometry()

        pointMat = rendering.MaterialRecord()
        pointMat.shader = "defaultUnlit"
        lineMat = rendering.MaterialRecord()
        lineMat.shader = "unlitLine"
        lineMat.line_width = 10

        #add new geometries
        for i, g in enumerate(self.geometries["point"]):
            self._scene.scene.add_geometry(
                f"_p{i}", g, pointMat
            )

        for i, g in enumerate(self.geometries["line"]):
            self._scene.scene.add_geometry(
                f"_l{i}", g, lineMat
            )

    #TASK 3: Call jarvis method and watch it happen
    def _run_convex_hull_animation(self):

        # self.qhull_it = graham_scan(self.vertices)
        # self.qhull_it = quickhull(self.vertices)
        self.qhull_it = jarvis(self.vertices)
        dt = 0.02
        prev_t = time.time()

        while True:

            if time.time() - prev_t > dt:
                try:
                    self.geometries = next(self.qhull_it)
                    gui.Application.instance.post_to_main_thread(self.window, self._run_convex_hull_animation_step)

                except StopIteration:
                    break

                finally:
                    prev_t = time.time()

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

                #adding the initial shape
                self._scene.scene.add_geometry("init", self.init_shape, defaultUnlit)

            #if the convex hull is empty, run the step by step algorithm
            elif self.convex_hull_points is None:

                #initializing generator if not initialized already
                if self.qhull_it is None:
                    #begin quickhull
                    self.qhull_it = graham_scan(self.vertices)
                    # self.qhull_it = quickhull(self.vertices)
                    # self.qhull_it = jarvis(self.vertices)



                #grabbing geometries from the next step
                try:
                    geometries = next(self.qhull_it)
                    self._run_convex_hull_step(geometries)

                #if finished, get the return value
                except StopIteration:
                    pass

                # xy = window_to_scene_coords(event.x, event.y, self._scene)

            #at this point the convex hull has already been created
            else:

                #convert click coordinates to screen coordinates
                xy = window_to_scene_coords(event.x, event.y, self._scene)

                #remove existing position
                self._scene.scene.remove_geometry("qpoint")

                self._on_point_placed(xy)

            return gui.Widget.EventCallbackResult.CONSUMED

        elif event.type == MouseEvent.Type.BUTTON_UP:
            print("mouse button lifted")
            return gui.Widget.EventCallbackResult.CONSUMED

        else:
            return gui.Widget.EventCallbackResult.IGNORED

    #TASK 4: WRITE YOUR CODE HERE!
    def _on_point_placed(self, xy):

        #print the coordinates you clicked at.
        #only use the x and y for your implementation
        print(f"clicked at: {xy}")

        self._scene.scene.clear_geometry()

        #add a marker to indicate where the user clicked (do not modify.)
        self._scene.scene.add_geometry("qpoint", sphere_at(xy), defaultUnlit)

        #self.vertices contains the entire point cloud
        #self.convex_hull_points contains the points that are part of the convex hull

        #after sorting the convex hull points by angle
        #you can create a lineset using g = U.chull_to_lineset()
        #then you can add the geometry to the scene as self._scene.scene.add_geometry(name_str, geometry, material)
        #read the rest of the code if you want to find examples of usage

        sorted_chull = U.sort_angle(self.convex_hull_points)[:, :2]

        #find the points, or line segments (whatever you want) that are visible by the point
        #color them and create open3d objects. Display them on the scene.
        #feel free to use or create any utility functions you want.
        #visit the docs http://www.open3d.org/docs/release/ if you want to find further information

        geometries = {
            "line": [U.chull_to_lineset(sorted_chull, color=U.black)],
            "point": [U.o3d_pointcloud(self.vertices)]
        }

        visible_points = self.get_visible_points(xy, sorted_chull)

        visible_points.append(xy[:2])
        visible_points = np.array(visible_points)

        geometries["line"].append(U.chull_to_lineset(visible_points, color=U.blue))

        pointMat = rendering.MaterialRecord()
        pointMat.shader = "defaultUnlit"
        lineMat = rendering.MaterialRecord()
        lineMat.shader = "unlitLine"
        lineMat.line_width = 10

        #add new geometries
        for i, g in enumerate(geometries["point"]):
            self._scene.scene.add_geometry(
                f"_p{i}", g, pointMat
            )

        for i, g in enumerate(geometries["line"]):
            self._scene.scene.add_geometry(
                f"_l{i}", g, lineMat
            )


    def get_visible_points(self, viewpoint, convexhull):
        visible_points = []
        viewpoint = viewpoint[:2]

        for point in convexhull:
            view_vector = point - viewpoint
            normal_vector = point + np.array((view_vector[1], -view_vector[0]))

            _, front = U.separate_points_by_line(point, normal_vector, convexhull)

            left, right = U.separate_points_by_line(viewpoint, point, front)

            if left.shape[0] == 0 or right.shape[0] == 0:
                visible_points.append(point)
        
        return visible_points


    def _on_key_pressed(self, event):

        print(event.key)

        #C key - start the animation
        if event.key == 99:
            threading.Thread(target=self._run_convex_hull_animation).start()

            return gui.Widget.EventCallbackResult.HANDLED

        #M key to begin point selection
        elif event.key == 109:

            #
            # print(type(np.asarray(self.geometries["line"])))
            # print(type(np.asarray(self.geometries["line"][0])))
            # print(np.asarray(self.geometries["line"][0].points) )
            self.convex_hull_points = np.asarray(self.geometries["line"][0].points)

            return gui.Widget.EventCallbackResult.HANDLED

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