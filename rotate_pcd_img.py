# script.py
import open3d as o3d
import os
import numpy as np

def load_point_cloud(file_path):
    # Load the point cloud from the given file path
    pcd = o3d.io.read_point_cloud(file_path)

    # Normalize the point cloud
    # Compute the centroid
    centroid = pcd.get_center()

    # Translate the point cloud to the origin
    pcd.translate(-centroid, relative=False)

    # Get the max extent for scaling
    max_extent = np.max(pcd.get_max_bound() - pcd.get_min_bound())

    # Scale the point cloud to fit in a unit cube
    pcd.scale(1 / max_extent, center=(0, 0, 0))

    return pcd

def rotate_point_cloud(pcd, angle):
    # Rotate the point cloud by the specified angle around the Y-axis
    R = pcd.get_rotation_matrix_from_xyz((0, np.radians(angle), 0))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def capture_image(pcd, folder_path, file_name):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Render the point cloud and capture it as an image
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Invisible window
    vis.add_geometry(pcd)
    #set_view_point(vis, pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('/usr/mvl2/lgsm2n/ULProj/rotated_imgs/image.png', do_render=True)
    vis.destroy_window()

def main():
    # Example file path and folder path
    file_path = '/usr/mvl2/lgsm2n/ULProj/workflow_steps/removed_base.ply'
    folder_path = '/usr/mvl2/lgsm2n/ULProj/rotated_imgs'
    file_name = 'rotated_point_cloud.png'

    pcd = load_point_cloud(file_path)
    pcd = rotate_point_cloud(pcd, 90)
    capture_image(pcd, folder_path, file_name)

if __name__ == "__main__":
    main()
