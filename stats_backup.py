import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm
from skimage import color
from scipy.stats import circmean, circstd


def calculate_stats(data_1d: np.ndarray) -> dict:
    """Calculates basic statistics for a 1D array of valid band data."""
    if data_1d.size == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'var': np.nan, 'cv': np.nan}
    
    mean = np.mean(data_1d)
    std = np.std(data_1d)

    stats = {
        'mean': float(mean),
        'median': float(np.median(data_1d)),
        'std': float(std),
        'var': float(np.var(data_1d)),
        'cv': float(std / mean) if mean != 0 else np.nan
    }
    return stats

def compute_raster_stats(gdf, base_url):
    """
    Extracts comprehensive features (RGB stats, Ratios, Differences, HSV, LAB, VARI)
    for each geometry in the GeoDataFrame by reading rasters from COG URLs.
    """
    gdf = gdf.copy()
    
    ALL_FEATURE_KEYS = [
        'r_mean', 'r_median', 'r_std', 'r_var', 'r_cv',
        'g_mean', 'g_median', 'g_std', 'g_var', 'g_cv',
        'b_mean', 'b_median', 'b_std', 'b_var', 'b_cv',
        'r_g_mean', 'r_g_std', 'r_b_mean', 'r_b_std', 'g_b_mean', 'g_b_std',
        'r_minus_g_mean', 'r_minus_g_std', 'g_minus_b_mean', 'g_minus_b_std',
        'h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std',
        'lab_l_mean', 'lab_a_mean', 'lab_b_mean',
        'vari_mean', 'vari_std'
    ]
    for col in ALL_FEATURE_KEYS:
        gdf[col] = np.nan

    for idx in tqdm(gdf.index, desc="Processing AOIs and extracting features"):
        row = gdf.loc[idx]
        cog_url = f"{base_url}/{row['image_uid']}.tif"
        
        try:
            with rasterio.open(cog_url) as src:
                geom_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs)
                geom_reprojected = geom_gdf.to_crs(src.crs)
                geom_transformed = [geom_reprojected.geometry.iloc[0].__geo_interface__]
                
                masked_data, _ = mask(src, geom_transformed, crop=True, all_touched=False)
                r_band, g_band, b_band = masked_data[0], masked_data[1], masked_data[2]
                
                common_valid_mask = np.ones(r_band.shape, dtype=bool) 
                nodata = src.nodata if src.nodata is not None else -9999
                
                for band in masked_data:
                    if np.ma.isMaskedArray(band):
                        common_valid_mask &= ~band.mask
                    common_valid_mask &= (band != nodata)
                    common_valid_mask &= (band != 0)

                if not np.any(common_valid_mask):
                    continue

                r_valid = r_band[common_valid_mask].astype(float)
                g_valid = g_band[common_valid_mask].astype(float)
                b_valid = b_band[common_valid_mask].astype(float)
                
                row_features = {}

                for name, data in zip(['r', 'g', 'b'], [r_valid, g_valid, b_valid]):
                    band_stats = calculate_stats(data)
                    for stat_key, value in band_stats.items():
                        row_features[f'{name}_{stat_key}'] = value
                
                # Ratios (R/G, R/B, G/B)
                ratio_r_g = np.divide(r_valid, g_valid, where=g_valid != 0, out=np.full_like(r_valid, np.nan))
                ratio_r_b = np.divide(r_valid, b_valid, where=b_valid != 0, out=np.full_like(r_valid, np.nan))
                ratio_g_b = np.divide(g_valid, b_valid, where=b_valid != 0, out=np.full_like(g_valid, np.nan))
                
                for name, data in zip(['r_g', 'r_b', 'g_b'], [ratio_r_g, ratio_r_b, ratio_g_b]):
                    row_features[f'{name}_mean'] = float(np.nanmean(data))
                    row_features[f'{name}_std'] = float(np.nanstd(data))
                    
                # Differences (R-G, G-B)
                row_features['r_minus_g_mean'] = float(np.mean(r_valid - g_valid))
                row_features['r_minus_g_std'] = float(np.std(r_valid - g_valid))
                row_features['g_minus_b_mean'] = float(np.mean(g_valid - b_valid))
                row_features['g_minus_b_std'] = float(np.std(g_valid - b_valid))
                
                rgb_stack = np.stack([r_valid, g_valid, b_valid], axis=-1)
                rgb_norm = rgb_stack / 255.0 

                # HSV
                hsv = color.rgb2hsv(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
                h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

                hue_rad = h * 2 * np.pi
                row_features['h_mean'] = float(circmean(hue_rad))
                row_features['h_std'] = float(circstd(hue_rad))
                
                row_features['s_mean'] = float(np.mean(s))
                row_features['s_std'] = float(np.std(s))
                row_features['v_mean'] = float(np.mean(v))
                row_features['v_std'] = float(np.std(v))

                lab = color.rgb2lab(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
                row_features['lab_l_mean'] = float(np.mean(lab[:, 0]))
                row_features['lab_a_mean'] = float(np.mean(lab[:, 1]))
                row_features['lab_b_mean'] = float(np.mean(lab[:, 2]))
                
                # Visual Atmospheric Resistance Index (VARI)
                denom = g_valid + r_valid - b_valid
                vari = np.divide(g_valid - r_valid, denom, where=denom != 0, out=np.full_like(g_valid, np.nan))
                
                row_features['vari_mean'] = float(np.nanmean(vari))
                row_features['vari_std'] = float(np.nanstd(vari))
                
                # Store all extracted features back into the GeoDataFrame
                for key, value in row_features.items():
                    gdf.at[idx, key] = value

        except rasterio.RasterioIOError:
            print(f"\nError reading raster {cog_url}. Skipping index {idx}.")
        except Exception as e:
            print(f"\nAn unexpected error occurred for index {idx}: {e}")
            pass
            
    return gdf