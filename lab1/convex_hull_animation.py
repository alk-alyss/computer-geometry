import numpy as np
import open3d as o3d
from tqdm import tqdm
import utility as U

def quickhull(pointcloud):

    if pointcloud.shape[0] < 3:
        raise ValueError("No convex hull can be defined with less than 3 points")

    elif pointcloud.shape[0] == 3:
        return pointcloud

    whole = pointcloud

    def query_line(pointcloud, A, B):

        nonlocal chull
        nonlocal whole

        #base case, empty pointcloud
        if pointcloud.shape[0] == 0:
            return

        #--------------------------------------------------------------------------
        # finding the furthest point C from the line A B
        #--------------------------------------------------------------------------

        #TASK 2.1
        #projecting
        AP = pointcloud - A
        AB = B - A

        proj = AB * (AP @ AB.reshape(-1, 1) / np.dot(AB, AB))

        #finding distances between points and their projection (which is the distance from the line)
        dist = np.linalg.norm(pointcloud - A - proj, axis=1)

        #the furthest point is the one with the maximum distance
        C = pointcloud[np.argmax(dist)]

        #adding C to the convex hull
        chull.append(C)

        #--------------------------------------------------------------------------
        # forming the lines CA, CB that constitute a triangle
        #--------------------------------------------------------------------------

        #separating the points on the right and on the left of AC
        ACleft, ACunused = U.separate_points_by_line(A, C, pointcloud)

        #separating the points on the right and on the left of CB
        CBleft, CBunused = U.separate_points_by_line(C, B, pointcloud)

        #--------------------------------------------------------------------------
        #---------------------Triangle Split visualization-------------------------
        #--------------------------------------------------------------------------
        #region tsplit
        geometries = {"line": [], "point": []}
        geometries["point"].append(U.o3d_pointcloud(ACleft, color=U.red))
        geometries["point"].append(U.o3d_pointcloud(CBleft, color=U.blue))
        geometries["point"].append(U.o3d_pointcloud(U.exclude_points(ACunused, CBleft), color=U.black))
        geometries["point"].append(U.o3d_pointcloud(U.exclude_points(whole, pointcloud), color=U.black))
        geometries["line"].append(U.o3d_line2d(A, C, color=U.red))
        geometries["line"].append(U.o3d_line2d(B, C, color=U.blue))
        geometries["line"].append(U.o3d_line2d(A, B, color=U.black))
        yield geometries
        #endregion

        #Recursively process each set
        yield from query_line(ACleft, A, C)
        yield from query_line(CBleft, C, B)

    #Finding extreme points
    A, B = pointcloud[np.argmin(pointcloud[:,0])], pointcloud[np.argmax(pointcloud[:,0])]

    #list to keep track of convex hull points
    chull = []

    #extreme points necessarily belong to the convex hull
    chull.append(A)
    chull.append(B)

    #splitting the pointcloud along the line into 2 sets
    P1, P2 = U.separate_points_by_line(A, B, pointcloud)

    #-----------------------------------------------------------
    #--------------Initial split visualization------------------
    #-----------------------------------------------------------
    #region INIT_VIS
    geometries = {"line": [], "point": []}
    geometries["point"].append(U.o3d_pointcloud(P1, color=U.red))
    geometries["point"].append(U.o3d_pointcloud(P2, color=U.blue))
    geometries["line"].append(U.o3d_line2d(A, B))
    yield geometries
    #endregion

    #recusrively processing each point set
    yield from query_line(P1, A, B)
    yield from query_line(P2, B, A)

    #sorting and creating lineset
    chull = U.sort_angle(np.array(chull))
    geometries = {"line": [], "point": []}
    geometries["line"].append(U.chull_to_lineset(chull, color=U.blue))
    geometries["point"].append(U.o3d_pointcloud(pointcloud))
    yield geometries

#TASK 3 - Implement the gift wrapping / jarvis algorithm for convex hull creation
#yield the geometry at every step in order to watch it happen
#follow the example of quickhull and graham_scan for how the geometry dictionary is structured.
def jarvis(pointcloud):

    if pointcloud.shape[0] < 3:
        raise ValueError("No convex hull can be defined with less than 3 points")

    elif pointcloud.shape[0] == 3:
        return pointcloud

    chull = []

    #given a query point Q and a pointcloud P, finds S in P, such that
    #the line QS is to the left of all points in P
    def find_leftmost_point(qpoint, pointcloud):
        while True:
            spoint = pointcloud[0]
            if np.all(spoint == qpoint):
                spoint = pointcloud[1]

            pointcloud, _ = U.separate_points_by_line(qpoint, spoint, pointcloud)

            if pointcloud.shape[0] == 0:  return spoint


    #finding the extreme left point, used as initial position
    #adding it to the convex hull
    left_point = pointcloud[np.argmin(pointcloud[:,0])]

    #keeping track of the latest point found
    chull.append(left_point)

    #repeating till the endpoint loops back around
    while True:

        endpoint = find_leftmost_point(left_point, pointcloud)

        if np.all(endpoint == chull[0]): break
            # return np.array(chull)

        chull.append(endpoint)

        left_point = endpoint

        geometries = {
            "line": [
                U.chull_to_lineset(np.array(chull))
            ],
            "point": [U.o3d_pointcloud(pointcloud)]
        }

        yield geometries

    geometries = {
        "line": [
            U.chull_to_lineset(np.array(chull))
        ],
        "point": [U.o3d_pointcloud(pointcloud)]
    }

    yield geometries

def graham_scan(pointcloud):

    if pointcloud.shape[0] < 3:
        raise ValueError("No convex hull can be defined with less than 3 points")

    elif pointcloud.shape[0] == 3:
        return pointcloud

    #TASK 1.1
    def is_ccw(v1, v2, v3):

        current_dir = v2-v1
        next_dir = v3-v2

        ccw = ((next_dir[0] * current_dir[1]) - (next_dir[1]*current_dir[0])) < 0
        return ccw

    #TASK 1.2
    #initial step
    # sorting according to angle
    p0 = pointcloud[np.argmin(pointcloud[:,1])]

    angles = np.arctan2(pointcloud[:,1] - p0[1], pointcloud[:,0] - p0[0])

    # Sort the points by angle
    sorted_idx = np.argsort(angles)
    pointcloud = pointcloud[sorted_idx]


    stack = [0, 1]
    ids = np.arange(2, pointcloud.shape[0])

    #iterating the rest of the points
    for id in ids:

        #TASK 1.3
        #pop elements from the stack until graham condition is satisfied
        while len(stack) > 1 and not is_ccw(
            pointcloud[stack[-2]], pointcloud[stack[-1]], pointcloud[id]
        ):
            stack.pop()


        #append current point
        stack.append(id)

        geometries = {
            "line": [
                U.chull_to_lineset(pointcloud[np.array(stack)])
            ],
            "point": [U.o3d_pointcloud(pointcloud)]
        }

        yield geometries

    geometries = {
        "line": [
            U.chull_to_lineset(pointcloud[np.array(stack)])
        ],
        "point": [U.o3d_pointcloud(pointcloud)]
    }

    yield geometries


if __name__ == "__main__":

    pcloud = np.random.random((100, 2))

    hull = graham_scan(pcloud)

    for geometry in quickhull(pcloud):
        o3d.visualization.draw_geometries(geometry)
