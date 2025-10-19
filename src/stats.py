import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import geopandas as gpd
# from skimage import color
# from scipy.stats import circmean, circstd

def compute_raster_stats(gdf, image_uid_col='image_uid', base_url='https://files.krschap.tech/api/public/dl/RGNv3CL4'):
    stat_cols = ['r_mean', 'r_median', 'r_std', 'r_var', 'g_mean', 'g_median', 'g_std', 'g_var',
                 'b_mean', 'b_median', 'b_std', 'b_var']
    # stat_cols += ['r_g_mean', 'r_g_std', 'r_b_mean', 'r_b_std', 'g_b_mean', 'g_b_std',
    #               'hsv_s_mean', 'hsv_v_mean', 'hsv_h_mean', 'hsv_h_std',
    #               'lab_l_mean', 'lab_a_mean', 'lab_b_mean', 'vari_mean', 'vari_std']
    
    for col in stat_cols:
        gdf[col] = np.nan
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing polygons"):
        try:
            with rasterio.open(f"{base_url}/{row[image_uid_col]}.tif") as src:
                geom = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs).to_crs(src.crs)
                masked_data, _ = mask(src, [geom.geometry.iloc[0].__geo_interface__], crop=True, all_touched=False)
                
                valid_mask = ~np.logical_or.reduce([masked_data[i].mask if np.ma.isMaskedArray(masked_data[i]) 
                                                     else masked_data[i] == (src.nodata or -9999) for i in range(3)])
                valid_mask &= (masked_data[0] > 0) & (masked_data[1] > 0) & (masked_data[2] > 0)
                
                bands = {c: masked_data[i][valid_mask].astype(float) for i, c in enumerate(['r', 'g', 'b'])}
                
                if bands['r'].size > 0:
                    for c in ['r', 'g', 'b']:
                        gdf.at[idx, f'{c}_mean'] = np.mean(bands[c])
                        gdf.at[idx, f'{c}_median'] = np.median(bands[c])
                        gdf.at[idx, f'{c}_std'] = np.std(bands[c])
                        gdf.at[idx, f'{c}_var'] = np.var(bands[c])
                    
                    # r, g, b = bands['r'], bands['g'], bands['b']
                    # for num, den, name in [(r, g, 'r_g'), (r, b, 'r_b'), (g, b, 'g_b')]:
                    #     ratio = np.divide(num, den, where=den!=0, out=np.full_like(num, np.nan))
                    #     gdf.at[idx, f'{name}_mean'] = np.nanmean(ratio)
                    #     gdf.at[idx, f'{name}_std'] = np.nanstd(ratio)
                    
                    # rgb_norm = np.stack([r, g, b], axis=-1) / (255.0 if max(r.max(), g.max(), b.max()) > 1 else 1.0)
                    # hsv = color.rgb2hsv(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
                    # hue_rad = hsv[:, 0] * 2 * np.pi
                    # gdf.at[idx, 'hsv_s_mean'] = np.mean(hsv[:, 1])
                    # gdf.at[idx, 'hsv_v_mean'] = np.mean(hsv[:, 2])
                    # gdf.at[idx, 'hsv_h_mean'] = circmean(hue_rad)
                    # gdf.at[idx, 'hsv_h_std'] = circstd(hue_rad)
                    
                    # lab = color.rgb2lab(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
                    # for i, c in enumerate(['l', 'a', 'b']):
                    #     gdf.at[idx, f'lab_{c}_mean'] = np.mean(lab[:, i])
                    
                    # denom = g + r - b
                    # vari = np.divide(g - r, denom, where=denom!=0, out=np.full_like(g, np.nan))
                    # gdf.at[idx, 'vari_mean'] = np.nanmean(vari)
                    # gdf.at[idx, 'vari_std'] = np.nanstd(vari)
        except Exception as e:
            print(f"Failed to process polygon at index {idx}: {e}")
            pass

    return gdf