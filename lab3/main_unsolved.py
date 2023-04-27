import os
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent, KeyEvent
from open3d.visualization.rendering import Camera
import numpy as np
import utility as U
import time


defaultUnlit = rendering.MaterialRecord()
defaultUnlit.shader = "defaultLit"

duration = 0
times = 0

class AppWindow:

    def __init__(self, width, height, window_name="Lab"):

        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window(window_name, width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)

        # basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        # self._scene.set_on_mouse(self._on_mouse_pressed)

        #set up camera
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.look_at(center, center - [0, 0, 2], [0, 1, 0])
        
        self.geometries = {}
        self.wireframe_on = False
        self.aabb_on = False
        self.pr_comp_on = False

    def _on_layout(self, layout_context):
        
        r = self.window.content_rect
        self._scene.frame = r

    def add_geometry(self, geometry, name):

        self._scene.scene.add_geometry(name, geometry, defaultUnlit)
        self.geometries[name] = geometry

    def remove_geometry(self, name):

        self._scene.scene.remove_geometry(name)

    def wireframe(self):
        if not self.wireframe_on:

            for k, mesh in self.geometries.items():

                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                lineset.paint_uniform_color(U.black)

                self._scene.scene.add_geometry("wireframe_" + k, lineset, defaultUnlit)


        else:

            self._scene.scene.clear_geometry()

            for k, v in self.geometries.items():
                self._scene.scene.add_geometry(k, v, defaultUnlit)

        self.wireframe_on = not self.wireframe_on

    def pop_plane_and_intersection(self):
        plane = self.geometries.pop('plane')
        self.remove_geometry('plane')
        self.geometries.pop('intersection')
        self.remove_geometry('intersection')

        self.geometries.pop('above')
        self.remove_geometry('above')
        self.geometries.pop('below')
        self.remove_geometry('below')

        return plane

    def _on_key_pressed(self, event):

        if event.type == KeyEvent.Type.UP:
            return gui.Widget.EventCallbackResult.HANDLED

        #C key
        if event.key == 99:
            print("C pressed")
            self.wireframe()
            return gui.Widget.EventCallbackResult.HANDLED

        #V key
        elif event.key == 118:
            print(self.geometries)
            return gui.Widget.EventCallbackResult.HANDLED

        #B key
        elif event.key == 98:
            self.aabb()
            return gui.Widget.EventCallbackResult.HANDLED
        
        #G key
        elif event.key == 103:
            self.principal_component()
            return gui.Widget.EventCallbackResult.HANDLED
        
        #P key
        elif event.key == 112:
            self.add_plane()
            return gui.Widget.EventCallbackResult.HANDLED

        #Up arrow
        elif event.key == 265:  
            self.move_plane(self.plane_speed)
            self.mesh_plane_intersection()
            return gui.Widget.EventCallbackResult.HANDLED

        #Down arrow
        elif event.key == 266:
            self.move_plane(-self.plane_speed)
            self.mesh_plane_intersection()
            return gui.Widget.EventCallbackResult.HANDLED

        print(event.key)
        return gui.Widget.EventCallbackResult.HANDLED


########################################################################
############################# LAB CONTENTS ############################# 
########################################################################
     
    # Task 2: Find the axis aligned bounding box
    def aabb(self):
        
        # if aabb is not currently displayed
        if not self.aabb_on:
            aabb = U.find_AABB(self.geometries["mesh"])
            self.add_geometry(aabb, "aabb")

        else:
            self.geometries.pop("aabb")
            self.remove_geometry("aabb")

        self.aabb_on = not self.aabb_on


    # Task 3: Calculate the principal component of the mesh
    def principal_component(self):
        if not self.pr_comp_on:
            principal_components = U.find_principal_components(self.geometries["mesh"])

            for i, pr in enumerate(principal_components):
                self.add_geometry(pr, f"principal_component{i}")

        else:
            for i in range(3):
                self.geometries.pop(f"principal_component{i}")
                self.remove_geometry(f"principal_component{i}")

        self.pr_comp_on = not self.pr_comp_on


    # Task 4.1: Create and visualize a plane
    def add_plane(self):
        # checking if a plane already exists in the scene
        if not 'plane' in self.geometries.keys():
            # Create a vector that will store the plane coefficients
            self.plane_vec = np.array([0.0, 1.0, 0.0, 0.0])
            self.plane_angle = 0.0
            self.plane_y_offset = 0.0

            self.plane_speed = 0.01
            self.plane_w = 1

            # Create a plane model
            plane = U.get_xz_plane()
            self.add_geometry(plane, 'plane')
            
            # calculate intersection with the mesh
            self.mesh_plane_intersection()
    

    def position_plane(self, plane):
        plane = U.translate(plane, np.array([0.0, self.plane_y_offset, 0.0]))
        return plane


    # Task 4.2: Give the ability to the plane to move up/down across the y-axis
    def move_plane(self, step):

        # remove plane and intersection from existing scene
        self.pop_plane_and_intersection()
        
        # update the y offset of the plane
        self.plane_y_offset += step 

        # Update the plane coefficient
        d = - self.plane_y_offset * self.plane_vec[1] 
        self.plane_vec[-1] = d

        # create a new plane (use get_zx_plane from utility.py)
        plane = U.get_xz_plane()
        # Translate the plane that is being visualized
        plane = self.position_plane(plane)

        # Add plane to the scene
        self.add_geometry(plane, 'plane')


    # Task 4.2: Calculate the mesh/plane intersection
    def mesh_plane_intersection(self):
        global duration, times

        mesh = self.geometries['mesh']
        plane_vec = self.plane_vec

        # get the mesh vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # get the indices of the triangles that intersect with the plane
        start = time.time()
        # intersect_idxs = U.fing_mesh_plane_intersection_triangles(vertices, triangles, plane_vec)
        intersect_idxs, above_idxs, below_idxs = U.fing_mesh_plane_intersection_triangles_efficient(vertices, triangles, plane_vec)
        # intersect_idxs = U.fing_mesh_plane_intersection_triangles_loop(vertices, triangles, plane_vec)
        # intersect_idxs = U.fing_mesh_plane_intersection_triangles_loop_efficient(vertices, triangles, plane_vec)
        duration += time.time() - start
        times += 1

        intersect_triangles = triangles[intersect_idxs, :]  
        above_triangles = triangles[above_idxs, :]  
        below_triangles = triangles[below_idxs, :]  

        ## RENDERING ##
        # create a new mesh containing only these triangles
        intersection = o3d.geometry.TriangleMesh()
        intersection.vertices = o3d.utility.Vector3dVector(vertices)
        intersection.triangles = o3d.utility.Vector3iVector(intersect_triangles)
        intersection.compute_triangle_normals()
        intersection.paint_uniform_color(U.blue)

        above = o3d.geometry.TriangleMesh()
        above.vertices = o3d.utility.Vector3dVector(vertices)
        above.triangles = o3d.utility.Vector3iVector(above_triangles)
        above.compute_triangle_normals()
        above.paint_uniform_color(U.red)

        below = o3d.geometry.TriangleMesh()
        below.vertices = o3d.utility.Vector3dVector(vertices)
        below.triangles = o3d.utility.Vector3iVector(below_triangles)
        below.compute_triangle_normals()
        below.paint_uniform_color(U.green)
        
        translation_vector = np.array((0, 0.5, 0))
        above.translate(translation_vector)
        below.translate(-translation_vector)

        self.add_geometry(intersection, "intersection")
        self.add_geometry(above, "above")
        self.add_geometry(below, "below")

        self.remove_geometry("mesh")







if __name__=="__main__":

    try:
        gui.Application.instance.initialize()

        # initialize GUI
        app = AppWindow(1280, 720)

        # load a mesh for manipulation
        mesh = o3d.io.read_triangle_mesh("lab3/armadillo_005.ply")
        mesh.compute_vertex_normals()
        

        # Task 1: Preprocess the mesh before adding it to the GUI
        mesh_center = U.get_center(mesh)
        mesh = U.translate(mesh, -mesh_center)
        mesh = U.unit_sphere_normalization(mesh)

        app.add_geometry(mesh, "mesh")

        gui.Application.instance.run()
    
    except KeyboardInterrupt:
        # print()
        # if times != 0:
        #     average_time = duration / times
        #     with open("times.txt", "a") as f:
        #         f.write(f"{average_time}\n")
        raise KeyboardInterrupt
