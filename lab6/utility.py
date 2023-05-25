import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def disp_image_and_rectangle(img, rect_start, template_rows, template_cols):
    # Display the original image
    plt.imshow(img, cmap='gray')

    # Get the current reference
    ax = plt.gca()

    # Create a Rectangle patch
    rect = Rectangle(rect_start, template_cols, template_rows, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

def visualize_stereo_frames(left_frame, right_frame):
    # Concatenate the left and right frames horizontally
    concatenated_frames = np.hstack((left_frame, right_frame))

    # Convert to RGB if the input is in BGR format
    if left_frame.ndim == 3 and left_frame.shape[2] == 3:
        concatenated_frames = cv2.cvtColor(concatenated_frames, cv2.COLOR_BGR2RGB)

    # Display the concatenated frames using Matplotlib
    plt.figure(figsize=(12, 5))
    plt.imshow(concatenated_frames, cmap='gray' if left_frame.ndim == 2 else None)
    plt.axis('off')
    plt.show()

def visualize_disparity_map(disparity_map, cmap='jet'):
    # Normalize the disparity map for visualization
    #normalized_disparity = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min())
    normalized_disparity = disparity_map
    # Display the disparity map using Matplotlib
    plt.figure(figsize=(12, 5))
    plt.imshow(normalized_disparity, cmap=cmap)
    plt.colorbar()
    plt.axis('off')
    plt.show()


def visualize_point_cloud(pc, colors=None):
    pc = o3d.utility.Vector3dVector(pc)
    pc = o3d.geometry.PointCloud(pc)
    # pc.paint_uniform_color(np.array([0.0, 0.0, 0.0]))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])