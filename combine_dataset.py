import os
import csv


# directory containing the txt files
dir_path = r"C:\Users\ad_xleong\Desktop\coral-sleuth\images\MCR_LTER_ComputerVision_LabeledCorals_2008_2009_2010\2010"

# path to the existing excel file
csv_path = r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\combined_annotations.csv"

for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        image_name = filename.replace('.txt', '')
        txt_file_path = os.path.join(dir_path, filename)

        # read txt file
        with open(txt_file_path, 'r') as file:
            # skip the header
            next(file)

            # read remaining lines
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                for line in file:
                    row_col_label = line.strip().split('; ')
                    
                    # append to csv
                    writer.writerow([image_name] + row_col_label)


