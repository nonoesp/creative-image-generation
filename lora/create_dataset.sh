#!/bin/bash
DATASET_DIR=dataset
ZIP_PATH=~/Desktop/dataset.zip

rm -rf $DATASET_DIR
mkdir -p $DATASET_DIR
curl -s -o $DATASET_DIR/metadata.csv "https://nono.ma/dataset-sketches.csv"
cp -r images/* $DATASET_DIR
python remove_images_without_metadata.py
[ -f "$ZIP_PATH" ] && rm "$ZIP_PATH"
zip -qr9 $ZIP_PATH $DATASET_DIR