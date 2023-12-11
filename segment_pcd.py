# class for doing the algorithm and include the different methods

import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


class SegmentLeavesPCD:


    def __init__(self):
        print("Creating Class")

    def CopyPointCloud(self, input_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        copy_point_cloud = o3d.geometry.PointCloud()

        # copying contents
        copy_point_cloud.points = input_pcd.points
        copy_point_cloud.colors = input_pcd.colors
        copy_point_cloud.normals = input_pcd.normals

        return copy_point_cloud

    def WritePointCloud(self, file_path: str, pcd: o3d.geometry.PointCloud) -> None: 
        o3d.io.write_point_cloud(file_path, pcd)

    def ReadCleanPointCloud(self, file_path: str, nb_neighbors=20, std_ratio=2.0) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(file_path)
        pcd = pcd.remove_duplicated_points()
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors,
                                                        std_ratio)
        
        return pcd

    def RemoveBase(self, pcd: o3d.geometry.PointCloud, z_level: float = 0.2):
        # Load your point cloud (replace 'your_point_cloud.pcd' with the actual path)
        point_cloud = self.CopyPointCloud(pcd) # nor normalizing now

        # Get the Z-coordinates of all points in the point cloud
        pts = point_cloud.points
        pts = np.asarray(pts)
        z_coordinates = pts[:, 2]

        # Find the highest Z value
        highest_z = z_coordinates.max()

        # Find the lowest Z value
        lowest_z = z_coordinates.min()

        # print("Highest Z value:", highest_z)
        # print("Lowest Z value:", lowest_z)

        threshold_z =lowest_z+ z_level # Replace with your desired threshold value

        # Create a binary mask for points below the threshold Z value
        below_threshold_mask = z_coordinates > threshold_z

        # Apply the mask to filter out points below the threshold Z value
        filtered_point_cloud = point_cloud.select_by_index(np.where(below_threshold_mask)[0])

        return filtered_point_cloud
    
    def GMM_Clustering(self, pcd: o3d.geometry.PointCloud, num_clusters: int = 7, ):

        gmm_pcd = self.CopyPointCloud(pcd)
        gm = GaussianMixture(n_components=num_clusters, init_params='k-means++', random_state=42) # could play with covariance_type='diag'
        pts = gmm_pcd.points
        pts = np.array(pts)
        # print(pts.shape)
        colors = gmm_pcd.colors
        colors = np.array(colors)
        # normals = filtered_point_cloud.normals
        # normals = np.array(normals)
        # print(normals.shape)
        #pts_colors = np.column_stack([pts, colors])
        labels_gm = gm.fit_predict(pts)
        # print(labels_gm)

        return labels_gm
    
    def KMeans_Clustering(self, pcd: o3d.geometry.PointCloud, num_clusters: int = 3):
        kmeans_pcd = self.CopyPointCloud(pcd)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        pts = np.array(kmeans_pcd.points)
        labels_kmeans = kmeans.fit_predict(pts)
        return labels_kmeans
    
    def DBSCAN_Clustering(self, pcd: o3d.geometry.PointCloud, eps: float = 0.1, min_samples: int = 10):
        dbscan_pcd = self.CopyPointCloud(pcd)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        pts = np.array(dbscan_pcd.points)
        colors = np.array(dbscan_pcd.colors)
        normals = np.array(dbscan_pcd.normals)
        pts_colors = np.column_stack([pts, colors, normals])
        labels_dbscan = dbscan.fit_predict(pts_colors)
        print(labels_dbscan)
        return labels_dbscan
    
    def MeanShift_Clustering(self, pcd: o3d.geometry.PointCloud, bandwidth: float = 0.1):
        ms_pcd = self.CopyPointCloud(pcd)
        pts = np.array(ms_pcd.points)
        colors = np.array(ms_pcd.colors)
        features = np.hstack([pts, colors])  # Combine point locations and colors into features

        # Create and fit a Mean Shift clustering model
        ms = MeanShift(bandwidth=bandwidth)
        labels_ms = ms.fit_predict(features)

        return labels_ms

    def estimate_normals(self, point_cloud: o3d.geometry.PointCloud, search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
        """
        Estimate normals for the point cloud.

        :param point_cloud: Open3D point cloud object.
        :param search_param: Parameters for KDTree search in normal estimation.
        :return: Point cloud with estimated normals.
        """
        point_cloud.estimate_normals(search_param=search_param)
        point_cloud.orient_normals_consistent_tangent_plane(k=30)
        return point_cloud

    def calculate_average_normal(self, point_cloud: o3d.geometry.PointCloud):
        """
        Calculate the average normal vector of the point cloud.

        :param point_cloud: Point cloud with normals.
        :return: Average normal vector.
        """
        normals = np.asarray(point_cloud.normals)
        average_normal = np.mean(normals, axis=0)
        return average_normal
    
    # def CompareHistograms(self, segmented_pcd: List[o3d.geometry.PointCloud]):
    #         # Create an empty similarity matrix to store cosine similarity scores
    #         num_clusters = len(segmented_pcd)
    #         similarity_matrix = np.zeros((num_clusters, num_clusters))

    #         # Iterate through each pair of clusters and calculate cosine similarity
    #         for i in range(num_clusters):
    #             hist_i = self.ComputeHistogram(segmented_pcd[i])
    #             for j in range(i+1, num_clusters):
    #                 hist_j = self.ComputeHistogram(segmented_pcd[j])
    #                 similarity_score = cosine_similarity([hist_i], [hist_j])[0][0]
    #                 similarity_matrix[i][j] = similarity_score
    #                 similarity_matrix[j][i] = similarity_score

    #         return similarity_matrix

    # def ComputeHistogram(self, cluster: o3d.geometry.PointCloud):
    #     normals = np.asarray(cluster.normals)
    #     magnitudes = np.linalg.norm(normals, axis=1)
    #     hist, _ = np.histogram(magnitudes, bins=20, density=True)
    #     return hist
    
    def GMMHist(self, pcd: o3d.geometry.PointCloud, num_clusters_initial: int = 7):
        # Read and preprocess the point cloud
        # input_pcd = self.ReadCleanPointCloud(file_path=pcd_path)
        # removed_base_pcd = self.RemoveBase(input_pcd)
        pcd = self.CopyPointCloud(pcd)

        # Fit a Gaussian Mixture Model to the data
        gmm_labels = self.GMM_Clustering(pcd, num_clusters=num_clusters_initial)

        # Compute the histograms and cosine similarity
        histograms = self.ComputeNormalsHistograms(pcd, gmm_labels)
        similarity_matrix = cosine_similarity(histograms)

        # Perform hierarchical clustering based on cosine similarity
        linkage_matrix = linkage(1 - similarity_matrix, method='average')
        cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')

        # Segment the point cloud based on the hierarchical clustering result
        #segmented_pcd = self.segment_point_cloud(pcd, cluster_labels)

        return cluster_labels

    def ComputeNormalsHistograms(self, point_cloud: o3d.geometry.PointCloud, labels: List[int]):
        unique_labels = np.unique(labels)
        histograms = []

        for label in unique_labels:
            mask = labels == label
            cluster = self.segment_point_cloud(point_cloud, mask)[0]

            # Estimate normals for the cluster
            cluster = self.estimate_normals(cluster)

            # Compute normals histograms
            hist = self.ComputeHistogram(cluster)
            histograms.append(hist)

        return histograms

    def ComputeHistogram(self, point_cloud: o3d.geometry.PointCloud, num_bins=30):
        normals = np.asarray(point_cloud.normals)
        hist, _ = np.histogramdd(normals, bins=num_bins, range=[[-1, 1], [-1, 1], [-1, 1]])
        hist = hist / np.sum(hist)  # Normalize the histogram
        return hist

    def HierarchicalClustering(self, input_pcd: o3d.geometry.PointCloud, labels: List, sim_metric: str = 'cosine', num_clusters: int = 3):
        segments = self.segment_point_cloud(input_pcd, labels)

        # finding normals
        normals = []
        for segment in segments:
           # segment = self.estimate_normals(segment)
            avg_normal = self.calculate_average_normal(segment)
            normals.append(avg_normal)

        # Calculate pairwise distances using pdist
        distances = pdist(normals, metric=sim_metric)

        # Convert the pairwise distances to a square distance matrix using squareform
        distance_matrix = squareform(distances)

        # Specify the number of clusters (in this case, 3)
        num_clusters = num_clusters

        # Compute the linkage matrix
        linkage_matrix = linkage(distance_matrix, method='ward')

        # Perform hierarchical clustering and assign nodes to clusters
        cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

        # Print cluster assignments
        #for node, cluster in enumerate(cluster_labels):
            #print(f"Node {node + 1} belongs to Cluster {cluster}")

        heirarchical_labels = cluster_labels[labels]
        #print(heirarchical_labels)

        return heirarchical_labels
    
    ### seeing if similar normals allow for combining clusters
    def segment_point_cloud(self, point_cloud: o3d.geometry.PointCloud, labels):
        clouds = []
        # Convert the labels list to a NumPy array
        labels = np.asarray(labels)

        # Get unique labels
        unique_labels = np.unique(labels)

        # Iterate over unique labels and extract points belonging to each segment
        for label in unique_labels:
            mask = labels == label
            segment_points = np.asarray(point_cloud.points)[mask]
            segment_colors = np.asarray(point_cloud.colors)[mask]
            segment_normals = np.asarray(point_cloud.normals)[mask]
            segment_cloud = o3d.geometry.PointCloud()
            segment_cloud.points = o3d.utility.Vector3dVector(segment_points)
            segment_cloud.colors = o3d.utility.Vector3dVector(segment_colors)
            segment_cloud.normals = o3d.utility.Vector3dVector(segment_normals)
            clouds.append(segment_cloud)

        return clouds
    
    def SegmentLeaves(self, pcd_path: str):
        # read pcd 
        input_pcd = self.ReadCleanPointCloud(file_path=pcd_path)

        # step 1 remove the base of the pcd
        removed_base_pcd = self.RemoveBase(input_pcd)

        # Kmeans clustering
        # kmeans_labels = self.KMeans_Clustering(removed_base_pcd, num_clusters=11)
        # gmm_labels = kmeans_labels

        # GMM clustering 
        gmm_labels = self.GMM_Clustering(removed_base_pcd, num_clusters=7)
        # hierarchical_labels = gmm_labels

        # DBSCAN clustering
        # dbscan_labels = self.DBSCAN_Clustering(removed_base_pcd)
        # gmm_labels = dbscan_labels

        # Mean Shift
       # ms_labels = self.MeanShift_Clustering(removed_base_pcd)
        #hierarchical_labels = ms_labels

        # hierarchical clustering by avg normals
        hierarchical_labels = self.HierarchicalClustering(removed_base_pcd, gmm_labels, sim_metric='cosine', num_clusters=3) # could be cosine sim as well 

        # Clustering with histograms
        # gmm_hist_labels = self.GMMHist(removed_base_pcd)
        # hierarchical_labels = gmm_hist_labels

        # segmenting the point clouds
        segmented_pcd = self.segment_point_cloud(removed_base_pcd, hierarchical_labels)

        return segmented_pcd
