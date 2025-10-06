import numpy as np
import rasterio
from rasterio.mask import mask
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def compute_stats(gdf, image_uid_col='image_uid', base_url='https://files.krschap.tech/api/public/dl/RGNv3CL4', max_workers=8):
    def process_polygon(idx, row):
        try:
            cog_url = f"{base_url}/{row[image_uid_col]}.tif"
            geom = [row.geometry.__geo_interface__]
            
            with rasterio.open(cog_url) as src:
                masked_data, _ = mask(src, geom, crop=True, all_touched=False)
                
                stats = {}
                for band_idx, band_name in enumerate(['r', 'g', 'b'], start=1):
                    band_data = masked_data[band_idx - 1]
                    valid = band_data[band_data != src.nodata] if not np.ma.isMaskedArray(band_data) else band_data[~band_data.mask]
                    
                    if valid.size > 0:
                        stats[f'{band_name}_mean'] = float(np.mean(valid))
                        stats[f'{band_name}_std'] = float(np.std(valid))
                    else:
                        stats[f'{band_name}_mean'] = np.nan
                        stats[f'{band_name}_std'] = np.nan
                
                return idx, stats
        except Exception:
            return idx, {f'{b}_{s}': np.nan for b in ['r', 'g', 'b'] for s in ['mean', 'std']}
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_polygon, idx, row): idx for idx, row in gdf.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating stats for polygons"):
            idx, stats = future.result()
            results[idx] = stats
    
    for col in ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std']:
        gdf[col] = [results[idx][col] for idx in gdf.index]
    
    return gdf
