from experiment import ULExperiment

# Define the base directory path
base_path = '/usr/mvl2/lgsm2n/ULProj/pcd_dataset/'

# Define the list of S values you want to iterate through
s_values = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']  # Add more values as needed

# Iterate through the S values and generate the file paths
for s_value in s_values:
    # Create the file path using string formatting
    gt_file_path = f"{base_path}{s_value}/gt_pcd/"
    input_file_path = f"{base_path}{s_value}/input_pcd/"
    #csv_file_path = f"{base_path}{s_value}/output.csv"
    csv_file_path = '/usr/mvl2/lgsm2n/ULProj/gmm7_HierConsineBaseline.csv'
    
    # Now 'file_path' contains the complete path for each S value
    print(gt_file_path)
    print(input_file_path)
    print(csv_file_path)


    exp = ULExperiment(gt_dir_filepath=gt_file_path, input_filepath=input_file_path,
                       csv_path =csv_file_path)
    exp.RunFullExperiment()