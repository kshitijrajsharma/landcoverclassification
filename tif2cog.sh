#!/bin/bash

[ -z "$1" ] && { echo "Usage: $0 <input_folder>"; exit 1; }
[ ! -d "$1" ] && { echo "Error: Directory not found"; exit 1; }

INPUT_FOLDER="$1"
COG_FOLDER="${INPUT_FOLDER}/cog"

mkdir -p "$COG_FOLDER"

mapfile -d '' files < <(find "$INPUT_FOLDER" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) -print0)
total=${#files[@]}
count=0

for tif_file in "${files[@]}"; do
    filename=$(basename "$tif_file")
    base_name="${filename%.*}"
    extension="${filename##*.}"
    output_file="${COG_FOLDER}/${base_name}.${extension}"
    
    ((count++))
    echo "$count/$total Converting: $filename"
    
    gdal_translate -q -of COG -co COMPRESS=LZW -co BIGTIFF=YES -ot Byte -scale "$tif_file" "$output_file"
done
