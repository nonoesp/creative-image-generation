from glob import glob
import os
import csv

DATASET_DIR = 'dataset'
NO_CAPTIONS_DIR = 'no_captions'

csv_path = os.path.join(DATASET_DIR, 'metadata.csv')
images_paths = glob(os.path.join(DATASET_DIR, '*.jpg'))

# Utils

def get_image_names_from_csv_file(filename):
    images = set()
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            images.add(row["file_name"])
    return images

# Remove images with no metadata

images_with_metadata = get_image_names_from_csv_file(csv_path)

if not os.path.exists(NO_CAPTIONS_DIR):
    os.makedirs(NO_CAPTIONS_DIR)

for image_path in images_paths:
    image_name = image_path.split('/')[1]

    if not image_name in images_with_metadata:
        os.rename(image_path, os.path.join(NO_CAPTIONS_DIR, image_name))