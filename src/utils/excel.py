import pandas as pd
from openpyxl import load_workbook


def adjust_cell_widths_in_excel(filename):
    """
    Adjusts the width of cells in an Excel file based on their content.

    Parameters:
    filename (str): The path to the Excel file.

    Returns:
    None
    """
    # Load workbook
    book = load_workbook(filename)

    # Iterate through each sheet in the workbook
    for sheet in book:
        # Iterate through each column in the sheet
        for column in sheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    # Update the max_length if cell length is greater
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            # Adjust the width of the column based on the max_length
            adjusted_width = (max_length + 2)
            sheet.column_dimensions[column[0].column_letter].width = adjusted_width

    # Save the workbook
    book.save(filename)
    
    
def dict_to_excel(data, filename, first_column_name):
    """
    Converts a dictionary to an Excel file with columns auto-fit.

    Parameters:
    data (dict): The dictionary data to convert.
    filename (str): The output Excel file name.
    first_column_name (str): The name of the first column header

    Returns:
    None
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Transpose the DataFrame so that models are rows and metrics are columns
    df = df.transpose()

    # Name the index
    df.index.name = first_column_name

    # Write the DataFrame to an Excel file
    df.to_excel(filename)

    adjust_cell_widths_in_excel(filename)


def append_label_distribution_to_excel(annotation_file, filename):
    """
    Appends a new sheet to an Excel workbook with label distribution data.

    Parameters:
    annotation_file (str): The path to the annotation CSV file.
    filename (str): The output Excel file name.

    Returns:
    None
    """
    # Load the annotation CSV file
    df = pd.read_csv(annotation_file)

    # Count the number of occurrences of each label
    label_counts = df['Label'].value_counts()

    # Convert the Series to DataFrame for easier Excel output
    label_counts_df = pd.DataFrame(label_counts).reset_index()
    label_counts_df.columns = ['Label', 'Count']

    # Specify the ExcelWriter to use the 'openpyxl' engine
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
        # Write the DataFrame to an Excel file, into the 'Label Distribution' sheet
        label_counts_df.to_excel(writer, sheet_name='Label Distribution', index=False)

    adjust_cell_widths_in_excel(filename)
    

if __name__ == "__main__":
    import json 
    
    test_file = r"C:\Users\ad_xleong\Desktop\coral-sleuth\models\coral_reef_classifier_epoch_1_1_batchsize_16_metrics_combined_annotations_about_40k_png_only_remapped.json"
    annotation_file = r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\annotations\combined_annotations_about_40k_png_only_remapped.csv"
    # Load the data from the JSON file
    with open(test_file, 'r') as file:
        data = json.load(file)
        
    dict_to_excel(data, "output.xlsx", "model_name")
    append_label_distribution_to_excel(annotation_file, "output.xlsx")