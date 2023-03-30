import open3d as o3d
import numpy as np
from functools import wraps

#constants
red = np.array([1,0,0])
green = np.array([0,1,0])
blue = np.array([0,0,1])
cyan = np.array([0,1,1])
yellow = np.array([1,1,0])
magenta = np.array([1,0,1])
black = np.array([0,0,0])
white = np.array([1,1,1])

def unit_sphere_normalization(P):

    return P / (np.sqrt(np.max((P * P).sum(-1))))

#TASK 2.2
#given a directed line represented as two points, splits the given point set as left and right from the line
def separate_points_by_line(A, B, P):
    
    A = np.append(A, 0)
    B = np.append(B, 0)
    P = np.append(P, np.zeros((P.shape[0], 1)), axis=1)
    

    AB = B - A
    AP = P - A

    cross = np.cross(AB, AP)
    left = P[cross[:, 2] > 0, :2]
    right = P[cross[:, 2] < 0, :2]

    return left, right

#TASK 1.4
#sorts a set of points according to their angle from the x axis
def sort_angle(points, descending=False):

    centered_points = recenter(points)

    angles = np.arctan2(centered_points[:,1], centered_points[:,0])

    indices = np.argsort(-angles if descending else angles)
    sorted_points = points[indices]

    return sorted_points

#Given two sets of points P and E, returns P - E (exclusion set operation)
def exclude_points(pcloud, exclude):

    assert len(exclude.shape) == len(pcloud.shape)
    assert exclude.shape[1] == pcloud.shape[1]

    for point in exclude:
        d = np.abs(pcloud - point).sum(-1)
        pcloud = pcloud[d > 1e-7]

    return pcloud

#Converts a 2d set of points representing the convex hull to a lineset
#assumes the points are sorted
def chull_to_lineset(points, color=black):

    points = pad_2d(points, None, -1, 0)

    indices1 = np.arange(points.shape[0])
    indices2 = np.arange(points.shape[0])+1
    indices2[-1] = 0

    indices = np.vstack((indices1, indices2)).T

    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(indices)
    ).paint_uniform_color(color)

#utility function for padding a 2d array with a row and/or column of a specific value
def pad_2d(arr, row=None, col=None, value=0):

    '''
        arr: np.array with shape (N, M)
    '''

    if len(arr.shape) != 2:
        raise ValueError(f"Only 2d numpy arrays are accepted. You gave {len(arr.shape)}d")

    if (not isinstance(row, int)) and (row is not None):
        raise ValueError("Only integer values -1 <= row <= num_rows are accepted")
    
    if (not isinstance(col, int)) and (col is not None):
        raise ValueError("Only integer values -1 <= row <= num_cols are accepted")

    #trivial case, no change is required
    if row is None and col is None:
        return arr

        
    zero_row = np.ones((1, arr.shape[1])) * value

    if row is not None:
        if row < -1 or row > arr.shape[0]:
            raise ValueError("Only integer values -1 <= row <= num_rows are accepted")
        elif row > 0:
            arr = np.concatenate(
                (arr[:row,:], zero_row, arr[row:, :]), axis=0
            )
        elif row == -1:
            arr = np.concatenate(
                (arr, zero_row), axis=0
            )

    zero_col = np.ones((arr.shape[0], 1)) * value

    if col is not None:
        if col < -1 or col > arr.shape[1]:
            raise ValueError("Only integer values -1 <= row <= num_rows are accepted")
        elif col > 0:
            arr = np.concatenate(
                (arr[:,:col], zero_col, arr[:, col:]), axis=1
            )
        elif col == -1:
            arr = np.concatenate(
                (arr, zero_col), axis=1
            )

    return arr

def edges_to_lineset(points, edges, color=black):

    points = pad_2d(points, col=-1)

    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(np.array(edges))
    ).paint_uniform_color(color)

#returns an open3d line from two points
def o3d_line2d(A, B, color=black):

    assert A.shape[0] == B.shape[0]

    if A.shape[0] == 3:
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.vstack((A, B))),
            o3d.utility.Vector2iVector(np.array([[0, 1]]))
        ).paint_uniform_color(color)
    else:
        A, B = np.append(A, 0), np.append(B, 0) 
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.vstack((A, B))),
            o3d.utility.Vector2iVector(np.array([[0, 1]]))
        ).paint_uniform_color(color)

#returns an open3d triangle from 3 points
def o3d_triangle2d(A, B, C, color=black):

    assert A.shape[0] == B.shape[0] == C.shape[0]

    if A.shape[0] == 3:
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.vstack((A, B, C))),
            o3d.utility.Vector2iVector(np.array([[0, 1],[1, 2],[2, 0]]))
        )
    else:
        A, B, C = np.append(A, 0), np.append(B, 0), np.append(C, 0) 
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.vstack((A, B, C))),
            o3d.utility.Vector2iVector(np.array([[0, 1],[1, 2],[2, 0]]))
        )

#EXTRA TASK 
#samples N points from the given M x d point cloud
def subsample(PC, N):

    assert len(PC.shape) == 2

    K, d = PC.shape

    if N > K:
        return PC
    else:
        p = np.random.permutation(N)
        return PC[p,:]

#unimplemented
def furthest_point_sample(PC, N):
    raise NotImplementedError()

#returns an open3d point cloud from a point array
def o3d_pointcloud(verts, center=False, color=black):

    assert verts.shape[-1] in (2, 3)
    assert len(verts.shape) == 2

    if verts.shape[-1] == 2:
        verts = pad_2d(verts, row = None, col=-1, value=0)

    if center:
        centroid = verts.mean(0)
        verts = verts - centroid

    if len(color.shape) == 1:
        return o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(verts)
        ).paint_uniform_color(color)
    else:
        assert len(color.shape) == 2
        assert color.shape[0] == verts.shape[0]

        pcloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(verts)
        )

        pcloud.colors = o3d.utility.Vector3dVector(color)
        return pcloud

#centers a pointcloud by subtracting the centroid from each point
def recenter(P):

    centroid = P.mean(0)

    return P - centroid

if __name__ == "__main__":

    pass