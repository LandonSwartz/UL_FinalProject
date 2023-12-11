# class for evaluating the point clouds

import open3d as o3d
import numpy as np

class PointCloudMetricsCalculator:
    def __init__(self, predicted_pc, ground_truth_pc, tolerance=0.001):
        self.predicted_pc = predicted_pc
        self.ground_truth_pc = ground_truth_pc
        self.tolerance = tolerance

    def find_matching_and_non_matching_points(self):
        """
        Find matching and non-matching points between the predicted point cloud and the ground truth point cloud.

        Returns:
        - matching_points: Indices of matching points in the predicted point cloud.
        - non_matching_predicted_points: Indices of non-matching points in the predicted point cloud.
        - non_matching_ground_truth_points: Indices of non-matching points in the ground truth point cloud.
        """
        predicted_points = np.asarray(self.predicted_pc.points)
        ground_truth_points = np.asarray(self.ground_truth_pc.points)

        matching_points = []
        non_matching_predicted_points = []
        non_matching_ground_truth_points = []

        for i, pred_point in enumerate(predicted_points):
            closest_distance = np.min(np.linalg.norm(ground_truth_points - pred_point, axis=1))
            if closest_distance <= self.tolerance:
                matching_points.append(i)
            else:
                non_matching_predicted_points.append(i)

        for j, gt_point in enumerate(ground_truth_points):
            closest_distance = np.min(np.linalg.norm(predicted_points - gt_point, axis=1))
            if closest_distance > self.tolerance:
                non_matching_ground_truth_points.append(j)

        return matching_points, non_matching_predicted_points, non_matching_ground_truth_points

    @staticmethod
    def calculate_iou(matching, non_matching_pred, non_matching_gt):
        intersection = len(matching)
        union = len(non_matching_pred) + len(non_matching_gt) + intersection
        iou = intersection / union
        return iou

    @staticmethod
    def calculate_precision(matching, non_matching_pred):
        true_positives = len(matching)
        false_positives = len(non_matching_pred)
        precision = true_positives / (true_positives + false_positives)
        return precision

    @staticmethod
    def calculate_recall(matching, non_matching_gt):
        true_positives = len(matching)
        false_negatives = len(non_matching_gt)
        recall = true_positives / (true_positives + false_negatives)
        return recall

    @staticmethod
    def calculate_f1_score(matching, non_matching_pred, non_matching_gt):
        precision = PointCloudMetricsCalculator.calculate_precision(matching, non_matching_pred)
        recall = PointCloudMetricsCalculator.calculate_recall(matching, non_matching_gt)
        
        if precision + recall == 0:
            # Handle the case where both precision and recall are zero.
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


    def CalculateMetrics(self):
        matching, non_matching_pred, non_matching_gt = self.find_matching_and_non_matching_points()
        iou = self.calculate_iou(matching, non_matching_pred, non_matching_gt)
        precision = self.calculate_precision(matching, non_matching_pred)
        recall = self.calculate_recall(matching, non_matching_gt)
        f1_score = self.calculate_f1_score(matching, non_matching_pred, non_matching_gt)
        return {
            'IOU': iou,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }