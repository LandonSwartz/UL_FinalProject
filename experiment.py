# from segment_pcd import SegmentLeavesPCD
# from evaluation import PointCloudMetricsCalculator
# import open3d as o3d
# import os
# import csv
from segment_pcd import SegmentLeavesPCD
from evaluation import PointCloudMetricsCalculator
import open3d as o3d
import os
import csv

class ULExperiment:

    def __init__(self, gt_dir_filepath: str, input_filepath: str, csv_path: str) -> None:
        print('Starting experiment')
        self.gt_dir_filepath = gt_dir_filepath
        self.input_pcd_path = input_filepath
        self.csv_path = csv_path

        # getting paths
        self.gt_paths = []
        self.gt_paths = self.ReadDir(self.gt_dir_filepath)
        print(f'The gt file paths are {self.gt_paths}')
        self.input_path = []
        self.input_path = self.ReadDir(self.input_pcd_path)
        self.input_path = self.input_path[0]
        print(f'The input pcd path is {self.input_path}')

        # Initialize a list to store experiment metrics
        self.all_metrics = []

        # setting up csv file output
        self.header = ['Segment', 'IOU', 'Precision', 'Recall', 'F1 Score']

    # getting all of the file paths for the directory
    def ReadDir(self, dir_path: str):
        # Get a list of all files in the directory
        files = os.listdir(dir_path)

        # Initialize an empty list to store full paths
        full_paths = []

        # Iterate through the list of files
        for file in files:
            # Get the full path of each file
            full_path = os.path.join(dir_path, file)
            # Append the full path to the list
            full_paths.append(full_path)

        return full_paths

    def RunFullExperiment(self):
        for file_path in self.gt_paths:
            print(f'Running experiment on gt model {file_path}')
            metrics = self.RunExperiment(file_path)

            if metrics is not None:
                self.all_metrics.append([file_path] + list(metrics.values()))

        # Write all metrics to the CSV file
        with open(self.csv_path, mode='a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if os.path.getsize(self.csv_path) == 0:
                writer.writerow(self.header)  # Write the header if the file is empty

            writer.writerows(self.all_metrics)

        # Calculate and append average metrics after all experiments
        avg_metrics = self.CalculateAverageMetrics()
        with open(self.csv_path, mode='a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Average'] + avg_metrics)  # Write the average metrics as a new row

        print(f'The experiment is finished, and the file output is at {self.csv_path}')

    def RunExperiment(self, gt_path: str):
        seg_class = SegmentLeavesPCD()
        ground_truth_pc = o3d.io.read_point_cloud(gt_path)
        segments = seg_class.SegmentLeaves(self.input_path)

        highest_f1 = 0.0
        highest_metrics = None

        for i, segment in enumerate(segments):
            eval = PointCloudMetricsCalculator(segment, ground_truth_pc)

            metrics = eval.CalculateMetrics()

            current_f1 = metrics['F1 Score']

            if current_f1 > highest_f1:
                highest_f1 = current_f1
                highest_metrics = metrics

        print(f'The most likely segment had an F1 score of {highest_f1 * 100:.2f}%')

        if highest_metrics is not None:
            print('The metrics are:')
            for metric_name, metric_value in highest_metrics.items():
                print(f'{metric_name}: {metric_value}')
        else:
            print('No suitable segment found.')

        return highest_metrics

    def CalculateAverageMetrics(self):
        if not self.all_metrics:
            return [0.0] * 4  # Return zeros for IOU, Precision, Recall, and F1 Score

        num_metrics = len(self.all_metrics)
        avg_metrics = [0.0] * 4

        for metrics in self.all_metrics:
            for i in range(4):
                avg_metrics[i] += metrics[i + 1]  # Skip the 'Segment' column

        avg_metrics = [round(avg / num_metrics, 4) for avg in avg_metrics]

        return avg_metrics





# class ULExperiment:

#     def __init__(self, gt_dir_filepath:str, input_filepath: str, csv_path: str) -> None:
#         print('Starting experiment')
#         self.gt_dir_filepath = gt_dir_filepath
#         self.input_pcd_path = input_filepath
#         self.csv_path = csv_path

#         # getting paths
#         self.gt_paths = []
#         self.gt_paths  = self.ReadDir(self.gt_dir_filepath) 
#         print(f'The gt file paths are {self.gt_paths}')
#         self.input_path = []
#         self.input_path = self.ReadDir(self.input_pcd_path)
#         self.input_path = self.input_path[0]
#         print(f'The input pcd path is {self.input_path}')

#     # getting all of the file paths for the directory
#     def ReadDir(self, dir_path: str):
#         # Get a list of all files in the directory
#         files = os.listdir(dir_path)

#         # Initialize an empty list to store full paths
#         full_paths = []

#         # Iterate through the list of files
#         for file in files:
#             # Get the full path of each file
#             full_path = os.path.join(dir_path, file)
#             # Append the full path to the list
#             full_paths.append(full_path)

#         return full_paths
    
#     def RunFullExperiment(self):
#         # setting up csv file output 
#         header = ['Segment', 'IOU', 'Precision', 'Recall', 'F1 Score']

#         with open(self.csv_path, mode='a', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow(header)

#             for file_path in self.gt_paths:
#                 print(f'Running experiment on gt model {file_path}')
#                 metrics = self.RunExperiment(file_path)
#                 #metrics['file_path'] = file_path

#                 if metrics is not None:
#                         metrics_list = list(metrics.values())
#                         metrics_list.insert(0, file_path)
#                         writer.writerow(metrics_list)
#                 else:
#                     print(f'No suitable segment found for gt model {file_path}')
        
#         print(f'The experiment is finished and the file output is at {self.csv_path}')

#     def RunExperiment(self, gt_path: str):
#         seg_class = SegmentLeavesPCD()
#         ground_truth_pc = o3d.io.read_point_cloud(gt_path)
#         segments = seg_class.SegmentLeaves(self.input_path)

#         highest_f1 = 0.0
#         highest_metrics = None

#         for i, segment in enumerate(segments):
#             #print(f'The metrics for segment {i} is: \n')
#             # Create an instance of the PointCloudMetricsCalculator
#             eval =  PointCloudMetricsCalculator(segment, ground_truth_pc)

#             # Calculate all metrics
#             metrics = eval.CalculateMetrics()

#             # Print the metrics
#             # for metric_name, metric_value in metrics.items():
#             #     print(f'{metric_name}: {metric_value}')

#             current_f1 = metrics['F1 Score']

#             if current_f1 > highest_f1:
#                 highest_f1 = current_f1
#                 highest_metrics = metrics
#                 predicted_pcd = segment

#         print(f'The most likely segment had a F1 score of {highest_f1*100:.2f}%')

#         if highest_metrics is not None:
#             # Print the metrics
#             print('The metrics are:')
#             for metric_name, metric_value in highest_metrics.items():
#                 print(f'{metric_name}: {metric_value}')
#         else:
#             print('No suitable segment found.')

#         # Print the metrics
#         # print('The metrics are:')
#         # for metric_name, metric_value in highest_metrics.items():
#         #     print(f'{metric_name}: {metric_value}')

#         return highest_metrics

