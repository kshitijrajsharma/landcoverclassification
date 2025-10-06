#!/bin/bash

[ -z "$1" ] && { echo "Usage: $0 <input_folder>"; exit 1; }
[ ! -d "$1" ] && { echo "Error: Directory not found"; exit 1; }

INPUT_FOLDER="$1"

mapfile -d '' files < <(find "$INPUT_FOLDER" -type f \( -iname "*.tif" -o -iname "*.tiff" \) -print0)

for tif_file in "${files[@]}"; do
    filename=$(basename "$tif_file")
    
    if gdalinfo "$tif_file" 2>/dev/null | grep -q "LAYOUT=COG"; then
        continue
    fi

    echo " Converting: $filename"
    dir=$(dirname "$tif_file")
    base_name="${filename%.*}"
    extension="${filename##*.}"
    output_file="${dir}/${base_name}_cog.${extension}"
    
    gdal_translate -q -of COG -co COMPRESS=LZW -co BIGTIFF=YES "$tif_file" "$output_file"
done
