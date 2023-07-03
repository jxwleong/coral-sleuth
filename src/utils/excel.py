import pandas as pd


def dict_to_excel(data, filename, first_column_name):
    """
    Converts a dictionary to an Excel file.

    The dictionary keys are considered as row indices and its inner dictionary keys are
    considered as column headers in the Excel sheet.

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



if __name__ == "__main__":
    import json 
    
    test_file = r"C:\Users\ad_xleong\Desktop\coral-sleuth\models\coral_reef_classifier_epoch_2_1_batchsize_16_metrics_combined_annotations_about_40k_png_CCA_Sand_SAND_TURF_Turf.json"
    
    # Load the data from the JSON file
    with open(test_file, 'r') as file:
        data = json.load(file)
        
        
    dict_to_excel(data, "output.xlsx", "model_name")