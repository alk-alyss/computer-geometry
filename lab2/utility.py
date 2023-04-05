# from __future__ import annotations
import open3d as o3d
import numpy as np
from meta import *

dims = lambda x: len(x.shape) 
di = lambda x, i: x.shape[i]

red = np.array([1,0,0])
green = np.array([0,1,0])
blue = np.array([0,0,1])
yellow = np.array([1,1,0])
magenta = np.array([1,0,1])
cyan = np.array([0,1,1])
black = np.array([0,0,0])
white = np.array([1,1,1])

class Circle2D(metaclass = MultipleMeta):

    def __init__(self, center: np.ndarray, radius: float, res: int = 40, color: np.ndarray = np.array([0,0,0])):

        self.center = center
        self.radius = radius
        self.res = res
        self.color = color

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, res: int = 40, color: np.ndarray = np.array([0,0,0])):

        A = np.array([[v2[0] - v1[0], v2[1] - v1[1]],
                      [v3[0] - v2[0], v3[1] - v2[1]]])

        b = 0.5 * np.array([np.dot(v2, v2) - np.dot(v1, v1), np.dot(v3, v3) - np.dot(v2, v2)])

        self.center = np.linalg.solve(A, b)
        self.radius = np.sqrt(np.dot(v1 - self.center, v1 - self.center))
        self.res = res
        self.color = color

    def contains(self, v: np.ndarray):

        d = v-self.center
        return np.dot(d, d) < self.radius * self.radius

    @property
    def center3d(self):
        return np.array([self.center[0], self.center[1], 0.5])

    @property
    def as_o3d_lineset(self):

        samples = np.linspace(0, 2*np.pi, self.res)
        points = np.array([
            self.radius * np.cos(samples),
            self.radius * np.sin(samples),
            np.zeros(self.res),
        ]).T

        pairs = np.array([np.arange(self.res), np.arange(self.res)+1]).T
        pairs[-1,0] = 0
        pairs[-1,1] = self.res-1

        return o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(pairs)).translate(self.center3d)

    @property
    def as_o3d_mesh(self):
        
        samples = np.linspace(0, 2*np.pi, self.res)
        points = np.array([
            self.radius * np.cos(samples),
            self.radius * np.sin(samples),
            np.zeros(self.res),
        ]).T

        c = np.array([[0, 0, 0.5]])
        points = np.concatenate((c, points))
        triangles = np.stack((np.arange(self.res-1)+1, np.zeros(self.res-1), np.arange(self.res-1)+2)).T
        triangles = np.concatenate((triangles, np.flip(triangles, -1)), 0)

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(triangles)
        ).translate(self.center3d).paint_uniform_color(self.color)

class Line2D:

    '''
        Line representation from 2 vertices
        v1: np.array(2)
        v2: np.array(2)
    '''

    def __init__(self, v1, v2):

        assert dims(v1) == 1 and dims(v2) == 1
        assert di(v1, 0) == 2 and di(v2, 0) == 2 

        self.v1 = v1
        self.v2 = v2
    
    #Checks for intersection between self and another given 'Line2D' object
    def _interesects(self, l1):

        pass

    @property
    def length(self):
        x = self.v1 = self.v2
        return np.sqrt((x*x).sum())

    @property
    def _o3d(self):

        vertices = np.ones((2, 3))
        vertices[:,:2] = np.array([self.v1, self.v2])

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector2iVector(np.array([[0,1]]))
        )
    
class Triangle2D(metaclass = MultipleMeta):

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, 
                 color: np.ndarray = np.array([0,0,0]), id: str = "0"):

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.color = color
        self.id = id

    def __init__(self, verts: np.ndarray, color: np.ndarray = np.array([0,0,0]),
                 id: str = "0"):

        assert dims(verts) == 2
        assert di(verts, 0) == 3 and di(verts, 1) == 2

        self.v1 = verts[0]
        self.v2 = verts[1]
        self.v3 = verts[2]

        self.color = color
        self.id = id

    def set_color(self, color: np.ndarray):
        self.color = color

    def set_id(self, id: str):
        self.id = id

    def contains(self, v: np.ndarray):

        a12 = Triangle2D(self.v1, self.v2, v).area
        a23 = Triangle2D(self.v2, self.v3, v).area
        a31 = Triangle2D(self.v3, self.v1, v).area

        if a12 + a23 + a31 - self.area < 1.e-8:
            return True
        
        return False

    def has_vertex(self, v: np.ndarray):

        if len(v.shape) == 2:

            target_vertices = self.vertices.reshape(-1, 1, 2)
            query_vertices = v.reshape(1, -1, 2)

            dist = (target_vertices - query_vertices)
            dist = (dist * dist).sum(-1).reshape(-1) < 1e-8

            return dist.any()

        elif len(v.shape) == 1:

            dist = self.vertices - v
            dist = (dist * dist).sum(-1).reshape(-1) < 1e-8

            return dist.any()

    def has_common_edge(self, tri):

        target_vertices = self.vertices.reshape(-1, 1, 2)
        query_vertices = tri.vertices.reshape(1, -1, 2)

        dist = (target_vertices - query_vertices)
        dist = (dist * dist).sum(-1).reshape(-1) < 1e-8

        #must have two common vertices
        return dist.sum() == 2

    def non_common_vertex(self, t, return_common=False):

        if not t.has_vertex(self.v1):
            if return_common:
                return self.v1, (self.v2, self.v3)
            else:
                return self.v1

        if not t.has_vertex(self.v2):
            if return_common:
                return self.v2, (self.v1, self.v3)
            else:
                return self.v2

        if not t.has_vertex(self.v3):
            if return_common:
                return self.v3, (self.v1, self.v2)
            else:
                return self.v3

    def split(self, v: np.ndarray):

        return [Triangle2D(self.v1, self.v2, v, self.color, self.id + "1"),
                Triangle2D(self.v2, self.v3, v, self.color, self.id + "2"),
                Triangle2D(self.v3, self.v1, v, self.color, self.id + "3")]

    def __repr__(self):
        return f"Triangle object with vertices v1{self.v1} v2{self.v2} v3{self.v3}"

    @property
    def vertices(self):

        return np.vstack((
            self.v1,
            self.v2,
            self.v3
        ))

    @property
    def circumcircle(self):
        return Circle2D(self.v1, self.v2, self.v3)

    @property
    def area(self):

        return abs((self.v2[0] - self.v1[0])*(self.v3[1] - self.v1[1]) - (self.v3[0] - self.v1[0])*(self.v2[1] - self.v1[1])) / 2.0

    @property
    def o3d(self):

        vertices = np.ones((3, 3))
        vertices[:,:2] = np.array([self.v1, self.v2, self.v3])

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector2iVector(np.array([[0,1], [1,2], [2,0]]))
        ).paint_uniform_color(self.color)

    @property
    def as_o3d_lineset(self):

        vertices = np.ones((3, 3))
        vertices[:,:2] = np.array([self.v1, self.v2, self.v3])

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 0]]))
        ).paint_uniform_color(self.color)

    @property
    def as_o3d_mesh(self):

        vertices = np.ones((3, 3))
        vertices[:,:2] = np.array([self.v1, self.v2, self.v3])

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
        ).paint_uniform_color(self.color)
    
    @property
    def as_o3d_mesh_fb(self):

        vertices = np.ones((3, 3))
        vertices[:,:2] = np.array([self.v1, self.v2, self.v3])

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 1]]))
        ).paint_uniform_color(self.color)

class Triangle3D(metaclass = MultipleMeta):

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, color: np.ndarray = np.array([0,0,0])):

        assert dims(v1) == dims(v2) == dims(v3) == 1, "Vertices must be one-dimensional arrays"
        assert di(v1, 0) == di(v2, 0) == di(v3, 0) == 3, "Vertices must have a shape of (3,)"

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.color = color
    
    def __init__(self, verts: np.ndarray, color: np.ndarray = np.array([0,0,0])):

        assert dims(verts) == 2, "Vertex array must be two-dimensional"
        assert di(verts, 0) == di(verts, 1) == 3, "Vertex array must have a shape of (3, 3)"

        self.v1 = verts[0]
        self.v2 = verts[1]
        self.v3 = verts[2]

        self.color = color

    def contains(self, v):
        pass

    def set_color(self, color: np.ndarray):
        self.color = color   

    @property
    def area(self):
        return 0.5 * np.linalg.norm(np.cross(self.v2 - self.v1, self.v3 - self.v1))

    @property
    def as_o3d_lineset(self):

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.array([self.v1, self.v2, self.v3])),
            o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 0]]))
        ).paint_uniform_color(self.color)

    @property
    def as_o3d_mesh(self):

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.array([self.v1, self.v2, self.v3])),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
        ).paint_uniform_color(self.color)
    
    @property
    def as_o3d_mesh_fb(self):

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.array([self.v1, self.v2, self.v3])),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 1]]))
        ).paint_uniform_color(self.color)


def flip_edge(t1: Triangle2D, t2: Triangle2D):

    nc1, c = t1.non_common_vertex(t2, return_common=True)
    nc2 = t2.non_common_vertex(t1)

    return Triangle2D(nc1, c[0], nc2, white, t1.id), Triangle2D(nc2, c[1], nc1, white, t2.id)

def vertices_of(l: list, unique=True):

    #input: list of Triangle2D
    #output: 2d numpy array containing all of their vertices

    verts = np.stack([
        t.vertices for t in l
    ]).reshape(-1, 2)

    if unique:
        verts = np.unique(verts, axis=0)

    return verts

if __name__ == "__main__":

    #-------------------------TESTS-------------------------

    v1 = np.array([1,0])
    v2 = np.array([0,1])
    v3 = np.array([0.5,1])
    v4 = np.array([-0.5,1])

    #-----------------------CIRCLE--------------------------

    c = Circle2D(v1,v2,v3, 40, black)
    qpoint = Circle2D(np.array([0.2,0.2]), 0.1, 40)

    o3d.visualization.draw_geometries([c.as_o3d_lineset, qpoint.as_o3d_lineset])

    print(c.contains(np.array([0.2,0.2])))

    #-----------------------TRIANGLE------------------------
    L = [Triangle2D(v1, v2, v3), Triangle2D(v1, v2, v4)]

    print(L[0].area)
    print(L[0].contains((v1+v2+v3)/3))
    print(L[0].has_vertex(v1))
    print(L[0].has_common_edge(L[1]))
