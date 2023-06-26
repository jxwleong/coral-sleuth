import os
import re
import requests
import pandas as pd

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

percent_covers_path = os.path.join(DATA_DIR, "coralnet_source_2091_percent_covers.csv")
# Create 'images' folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Read the annotations CSV
annotations = pd.read_csv(percent_covers_path)
image_list = annotations['Image name'].drop_duplicates()

# Extract Image IDs from the CSV
image_ids = annotations['Image ID'].drop_duplicates()

file_type=".jpg"
found_images = 0
print(image_ids) 
print(image_list)

for index, image_id in enumerate(image_ids):
    page_url = f'https://coralnet.ucsd.edu/image/{image_id}/view/'
    r = requests.get(page_url)
    source = r.text

    img_url = re.search(r'coralnet-production.s3.amazonaws.com:443(.*?)>', source).group(1)
    url = 'https://coralnet-production.s3.amazonaws.com' + img_url
    url = url.replace('&amp;', '&')
    url = url.replace('" /', '')
    
    r = requests.get(url, allow_redirects=True)
    if not os.path.exists(f"{image_list[index]}{file_type}"):
        open(f"images/{image_list[index]}{file_type}", 'wb').write(r.content)
    
    found_images += 1
 

print(f'Found {found_images} images.')