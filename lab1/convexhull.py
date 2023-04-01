import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
import utility as U
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#TASK 2
def quickhull(pointcloud):

    if pointcloud.shape[0] < 3:
        raise ValueError("No convex hull can be defined with less than 3 points")

    elif pointcloud.shape[0] == 3:
        return pointcloud

    def query_line(pointcloud, A, B):

        nonlocal chull

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
        ACleft, _ = U.separate_points_by_line(A, C, pointcloud)

        #separating the points on the right and on the left of CB
        CBleft, _ = U.separate_points_by_line(C, B, pointcloud)

        #Recursively process each set
        query_line(ACleft, A, C)
        query_line(CBleft, C, B)

    #Finding extreme points
    A, B = pointcloud[np.argmin(pointcloud[:,0])], pointcloud[np.argmax(pointcloud[:,0])]

    #list to keep track of convex hull points
    chull = []

    #extreme points necessarily belong to the convex hull
    chull.append(A)
    chull.append(B)

    #splitting the pointcloud along the line into 2 sets
    P1, P2 = U.separate_points_by_line(A, B, pointcloud)

    #recusrively processing each point set
    query_line(P1, A, B)
    query_line(P2, B, A)

    return np.array(chull)

#TASK 3
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

        if np.all(endpoint == chull[0]):
            return np.array(chull)

        chull.append(endpoint)

        left_point = endpoint


#TASK 1
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
    for i in ids:

        #TASK 1.3
        #pop elements from the stack until graham condition is satisfied
        while len(stack) > 1 and not is_ccw(
            pointcloud[stack[-2]], pointcloud[stack[-1]], pointcloud[i]
        ):
            stack.pop()


        #append current point
        stack.append(i)

    return pointcloud[np.array(stack)]

if __name__ == "__main__":
    m = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    vertices = np.asarray(m.vertices)[:,:2]
    totalPoints = vertices.shape[0]

    samples = list(map(int, np.linspace(1024, totalPoints, 50)))

    durations = []

    for i, algorithm in enumerate([graham_scan, quickhull, jarvis]):
        match i:
            case 0:
                label = "graham scan"
            case 1:
                label = "quickhull"
            case 2:
                label = "jarvis match"

        print(label)

        duration = []

        for sample in tqdm(samples):
            #load stanford bunny

            subvertices = U.subsample(vertices, sample)

            #calculate convex hull using scipy
            # hull = ConvexHull(vertices)

            # #convert the point cloud to open3d
            # pointcloud = U.o3d_pointcloud(vertices)
            # #convert the convex hull to a lineset
            # hull = U.chull_to_lineset(vertices[hull.vertices])

            #visualize
            # o3d.visualization.draw_geometries([pointcloud, hull])

            start = time.time()
            chull = U.sort_angle(algorithm(subvertices))
            duration.append(time.time() - start)

            # ls = U.chull_to_lineset(chull)
            # pts = U.o3d_pointcloud(vertices)

            # o3d.visualization.draw_geometries([pts, ls])

        durations.append(duration)

        plt.plot(samples, duration, label=label)

    plt.xlabel("Number of points")
    plt.ylabel("Seconds")
    plt.legend()
    plt.show()