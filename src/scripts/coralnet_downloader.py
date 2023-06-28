import os
import re
import requests
import pandas as pd

import csv
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR, IMAGE_DIR
from config.proxy import proxies

"""
Please use your own proxies. Example as follow:
 
proxies = {
  "http": "http://10.10.1.10:3128",
  "https": "http://10.10.1.10:1080",
}
"""

percent_covers_path = os.path.join(ANNOTATION_DIR, "coralnet_source_2091_percent_covers.csv")

# Create 'images' folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Read the annotations CSV
annotations = pd.read_csv(percent_covers_path)

# Extract unique 'Image name' and 'Image ID'
unique_annotations = annotations.drop_duplicates(subset=['Image name', 'Image ID'])

image_list = unique_annotations['Image name'].tolist()[:-1]
image_ids = unique_annotations['Image ID'].tolist()[:-1]
print(f"Image ID Count: {len(image_ids)}")

found_images = 0
#print(image_ids) 
#print(image_list)



for image_name, image_id in zip(image_list, image_ids):
    page_url = f'https://coralnet.ucsd.edu/image/{image_id}/view/'
    print(f"Request to {page_url}", end=' ')
    r = requests.get(page_url, proxies=proxies)
    source = r.text
    print("......DONE")

    img_url = re.search(r'coralnet-production.s3.amazonaws.com:443(.*?)>', source).group(1)
    url = 'https://coralnet-production.s3.amazonaws.com' + img_url
    url = url.replace('&amp;', '&')
    url = url.replace('" /', '')
    
    print(f"Request to {url}", end=' ')
    r = requests.get(url, allow_redirects=True, proxies=proxies)
    print("......DONE")
    image_filepath = os.path.join(IMAGE_DIR, f"{image_name}")
    if not os.path.exists(image_filepath):
        open(image_filepath, 'wb').write(r.content)
        print(f"Image saved: {image_filepath}")
    
    found_images += 1

print(f'Found {found_images} images.')
