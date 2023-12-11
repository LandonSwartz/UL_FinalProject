import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys

def create_rotation_gif(point_cloud, output_path, num_frames=36, duration=0.1):
    """
    Create a GIF animation of an Open3D point cloud rotating.

    Args:
    - point_cloud: An Open3D point cloud object.
    - output_path: The path to save the output GIF.
    - num_frames: The number of frames in the animation.
    - duration: The duration (in seconds) of each frame in the GIF.

    Returns:
    - None
    """
    # Create a directory to store temporary frame images
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Get the bounding box of the point cloud
    min_bound = np.min(np.asarray(point_cloud.points), axis=0)
    max_bound = np.max(np.asarray(point_cloud.points), axis=0)
    center = (min_bound + max_bound) / 2
    radius = np.linalg.norm(max_bound - center)

    # Generate frames for the rotation animation
    for i in range(num_frames):
        angle_deg = i * (360 / num_frames)
        angle_rad = np.radians(angle_deg)

        # Set the camera view parameters for the current frame
        view_matrix = np.eye(4)
        view_matrix[:3, 3] = center
        view_matrix[:3, :3] = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        view_matrix[:3, 3] -= view_matrix[:3, :3].dot(center)

        # Create a temporary point cloud with the rotated view
        # Create a temporary point cloud with the rotated view
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = point_cloud.points
        temp_pcd.colors = point_cloud.colors
        temp_pcd.transform(view_matrix)

        # Convert points to a NumPy array
        temp_points = np.asarray(temp_pcd.points)
        temp_colors= np.asarray(temp_pcd.colors)

        # Render the point cloud using Matplotlib
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(temp_points[:, 0], temp_points[:, 1], temp_points[:, 2], s=1, c=temp_colors) # change colors
        ax.axis('off')
        
        # Save the frame as an image
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=80, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


    # Convert the frames to a GIF using imageio
    frame_paths = [os.path.join(temp_dir, f"frame_{i:03d}.png") for i in range(num_frames)]
    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimsave(output_path, images, duration=duration)

    # Clean up temporary frame images
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)

# Main execution block
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python create_rotation_gif.py <point_cloud_file> <output_gif> <num_frames> <duration>")
        sys.exit(1)

    point_cloud_file = sys.argv[1]
    output_gif = sys.argv[2]
    num_frames = int(sys.argv[3])
    duration = float(sys.argv[4])

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)

    # Create the rotation GIF
    create_rotation_gif(point_cloud, output_gif, num_frames, duration)
