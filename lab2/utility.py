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

def polar2cart(radius, theta):
    x = np.cos(theta)*radius
    y = np.sin(theta)*radius
    return x, y

def cart2polar(x, y):
    r = (x**2 + y**2)**0.5
    theta = x/r
    return r, theta

class Circle2D(metaclass = MultipleMeta):

    def __init__(self, center: np.ndarray, radius: float, res: int = 100, color: np.ndarray = np.array([0,0,0])):

        self.center = center
        self.radius = radius
        self.res = res
        self.color = color

    #TASK - 1.1
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, res: int = 40, color: np.ndarray = np.array([0,0,0])):

        #calculate the circle's center and radius
        A = 2 * np.array([
            [v2[0]-v1[0], v2[1]-v1[1]],
            [v3[0]-v2[0], v3[1]-v2[1]]
        ]) 


        b = np.array([
            np.dot(v2, v2) - np.dot(v1, v1),
            np.dot(v3, v3) - np.dot(v2, v2),
        ])

        self.center = np.linalg.solve(A, b)
        self.radius = np.linalg.norm(v1 - self.center)

        self.res = res
        self.color = color

    #TASK - 1.2
    def contains(self, v: np.ndarray):

        return np.linalg.norm(v - self.center) <= self.radius

    @property
    def center3d(self):
        return np.array([self.center[0], self.center[1], 0.5])

    #TASK - 1.3
    @property
    def as_o3d_lineset(self):

        theta = np.linspace(0, 2*np.pi, self.res)

        points = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            np.zeros(self.res)
        ]).T

        segments = np.array([
            np.arange(self.res),
            np.arange(self.res) + 1
        ]).T

        segments[-1, 1] = 0

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector2iVector(segments)
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

    #TASK - 2.2
    def contains(self, v: np.ndarray):
        t1 = Triangle2D(self.v1, self.v2, v)
        t2 = Triangle2D(self.v2, self.v3, v)
        t3 = Triangle2D(self.v3, self.v1, v)

        return t1.area + t2.area + t3.area - self.area < 1e-8

    #TASK - 2.3
    def has_vertex(self, v: np.ndarray):
        dist = self.vertices - v
        dist = np.linalg.norm(dist, axis=1)

        return (dist < 1e-8).any()

    #TASK - 2.4
    def has_common_edge(self, t):
        return sum(map(self.has_vertex, t.vertices)) >= 2

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

    #TASK
    def split(self, v: np.ndarray):
        return (
            Triangle2D(self.v1, self.v2, v, self.color, self.id+"1"),
            Triangle2D(self.v2, self.v3, v, self.color, self.id+"2"),
            Triangle2D(self.v3, self.v1, v, self.color, self.id+"3")
        )

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

    #TASK - 2.1
    @property
    def area(self):
        return 0.5 * np.abs(np.cross(self.v2 - self.v1, self.v3 - self.v1))

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

    c = Circle2D(v1,v2,v3, 40, red)
    qpoint = Circle2D(np.array([0.2,0.2]), 0.1, 40)

    o3d.visualization.draw_geometries([c.as_o3d_lineset, qpoint.as_o3d_lineset])

    print(c.contains(np.array([0.2,0.2])))

    #-----------------------TRIANGLE------------------------
    L = [Triangle2D(v1, v2, v3), Triangle2D(v1, v2, v4)]

    print(L[0].area)
    print(L[0].contains((v1+v2+v3)/3))
    print(L[0].has_vertex(v1))
    print(L[0].has_common_edge(L[1]))