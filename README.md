## Dataset

About the dataset :
In the attribute table, there is a field called OpenAerialMap (OAM) , which contains the OpenAerialMap link or the link to the image which was used to generate the polygon or coverage sample.
In some cases, the image didn’t come directly from OAM, so there is a the Drive folder link in that same field.
The codigo field corresponds to the different land cover samples found in the area. To see what each code means, check the Observ field, where it’s described.

### Category Distribution (Label Classes)

- Building: 683
- Clean pasture: 320
- Dense forest: 155
- Bare areas: 145
- Urban not continuous: 142
- Weedy pasture: 128
- Wooded pasture: 64
- Permanent crops: 57
- Fragmented forest: 56
- Seasonal crops: 42
- Highway: 33
- Rivers: 21
- Urban: 17
- Open forest: 17
- Lakes: 14
- Secondary vegetation: 14
- Open grassland: 13
- Shrubland: 10
- Sandy areas: 6
- Burnt areas: 4
- Commercial or industrial area: 4
- Dense grassland: 3
- Canals: 3
- Forest: 1

<img width="815" height="696" alt="image" src="https://github.com/user-attachments/assets/ee838726-6d5b-4690-9c45-b6c9e470604e" />



### Spatial Distribution 

<img width="927" height="511" alt="image" src="https://github.com/user-attachments/assets/9baa3f9d-ca02-47dc-bd57-47c30bdaf150" />



**Total samples in the dataset: 1,953 , 24 distinct classes**

You can check the [data centroid file](./data/data_centroid.geojson) to visualize the  distribution interactively.


### Project Setup

Land cover classification using drone imagery and polygon features with statistical analysis.

### Project Structure

```
├── src/                    
│   ├── preprocess.py               # Utilities to build dataset from OAM files
│   ├── stats.py                    # Functions to extract image features
│   └── utils.py          
├── notebooks/             
│   ├── preprocess.ipynb            # Notebook to run the dataset build
│   ├── stats_computation.ipynb     # Notebook to run the feature extraction
│   └── distribution.ipynb          # Notebook to explore data distribution
├── data/
│   └── geojson/                    # Labeled data
├── tif2cog.sh                      # Script to convert tif files to COG
└── main.py               
```

## Setup

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

## Usage

### Notebook Workflow

1. **Dataset Distribution** - First, see the dataset distribution:
   ```bash
   jupyter notebook notebooks/distribution.ipynb
   ```

2. **Preprocessing** - Preprocess the data (download images, build URLs):
   ```bash
   jupyter notebook notebooks/preprocess.ipynb
   ```

3. **Stats Computation** - After preprocessing, build the features (compute raster statistics):
   ```bash
   jupyter notebook notebooks/stats_computation.ipynb
   ```
   This will build the features for your model.

### Convert to COG
Standardize images to Cloud Optimized GeoTIFF: ( This is part of preprocess notebook)
```bash
./tif2cog.sh data/images
```
