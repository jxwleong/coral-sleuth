import pandas as pd
from openpyxl import load_workbook


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

    # Load workbook
    book = load_workbook(filename)
    sheet = book.active

    for column in sheet.columns:
        max_length = 0
        column = [cell for cell in column]
        
        for cell in column:
            try:
                # update the max_length if cell length is greater
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        
        # Consider the header cell length as well
        if len(str(column[0].value)) > max_length:
            max_length = len(column[0].value)
            
        adjusted_width = (max_length + 2)
        sheet.column_dimensions[column[0].column_letter].width = adjusted_width

    book.save(filename)



if __name__ == "__main__":
    import json 
    
    test_file = r"C:\Users\ad_xleong\Desktop\coral-sleuth\models\coral_reef_classifier_epoch_2_1_batchsize_16_metrics_combined_annotations_about_40k_png_CCA_Sand_SAND_TURF_Turf.json"
    
    # Load the data from the JSON file
    with open(test_file, 'r') as file:
        data = json.load(file)
        
    dict_to_excel(data, "output.xlsx", "model_name")