import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import geopandas as gpd
from skimage import color
from scipy.stats import circmean, circstd

def compute_raster_stats(gdf, image_uid_col='image_uid', base_url='https://files.krschap.tech/api/public/dl/RGNv3CL4'):
    def process_polygon(idx, row):
        try:
            cog_url = f"{base_url}/{row[image_uid_col]}.tif"
            
            with rasterio.open(cog_url) as src:
                geom_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs)
                geom_reprojected = geom_gdf.to_crs(src.crs)
                geom_transformed = [geom_reprojected.geometry.iloc[0].__geo_interface__]
                
                masked_data, _ = mask(src, geom_transformed, crop=True, all_touched=False)
                
                r_band = masked_data[0].copy()
                g_band = masked_data[1].copy()
                b_band = masked_data[2].copy()
                
                is_masked = np.ma.isMaskedArray(r_band)
                
                if is_masked:
                    valid_mask = ~(r_band.mask | g_band.mask | b_band.mask)
                else:
                    nodata = src.nodata if src.nodata is not None else -9999
                    valid_mask = (r_band != nodata) & (g_band != nodata) & (b_band != nodata)
                
                r_valid = r_band[valid_mask].copy()
                g_valid = g_band[valid_mask].copy()
                b_valid = b_band[valid_mask].copy()
            
            stats = {}
            
            if r_valid.size > 0:
                stats['r_mean'] = float(np.mean(r_valid))
                stats['r_median'] = float(np.median(r_valid))
                stats['r_std'] = float(np.std(r_valid))
                stats['r_var'] = float(np.var(r_valid))
                
                stats['g_mean'] = float(np.mean(g_valid))
                stats['g_median'] = float(np.median(g_valid))
                stats['g_std'] = float(np.std(g_valid))
                stats['g_var'] = float(np.var(g_valid))
                
                stats['b_mean'] = float(np.mean(b_valid))
                stats['b_median'] = float(np.median(b_valid))
                stats['b_std'] = float(np.std(b_valid))
                stats['b_var'] = float(np.var(b_valid))
                
                r_g = np.divide(r_valid, g_valid, where=g_valid != 0, out=np.full_like(r_valid, np.nan, dtype=float))
                r_b = np.divide(r_valid, b_valid, where=b_valid != 0, out=np.full_like(r_valid, np.nan, dtype=float))
                g_b = np.divide(g_valid, b_valid, where=b_valid != 0, out=np.full_like(g_valid, np.nan, dtype=float))
                
                stats['r_g_mean'] = float(np.nanmean(r_g))
                stats['r_g_std'] = float(np.nanstd(r_g))
                stats['r_b_mean'] = float(np.nanmean(r_b))
                stats['r_b_std'] = float(np.nanstd(r_b))
                stats['g_b_mean'] = float(np.nanmean(g_b))
                stats['g_b_std'] = float(np.nanstd(g_b))
                
                rgb_stack = np.stack([r_valid, g_valid, b_valid], axis=-1)
                rgb_norm = rgb_stack / 255.0 if rgb_stack.max() > 1 else rgb_stack
                
                # https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2hsv
                hsv = color.rgb2hsv(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
                stats['hsv_s_mean'] = float(np.mean(hsv[:, 1]))
                stats['hsv_v_mean'] = float(np.mean(hsv[:, 2]))
                
                hue_rad = hsv[:, 0] * 2 * np.pi
                stats['hsv_h_mean'] = float(circmean(hue_rad))
                stats['hsv_h_std'] = float(circstd(hue_rad))
                
                # https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab
                lab = color.rgb2lab(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
                stats['lab_l_mean'] = float(np.mean(lab[:, 0]))
                stats['lab_a_mean'] = float(np.mean(lab[:, 1]))
                stats['lab_b_mean'] = float(np.mean(lab[:, 2]))
                
                denom = g_valid + r_valid - b_valid
                vari = np.divide(g_valid - r_valid, denom, where=denom != 0, out=np.full_like(g_valid, np.nan, dtype=float))
                
                stats['vari_mean'] = float(np.nanmean(vari))
                stats['vari_std'] = float(np.nanstd(vari))
            else:
                stats = {k: np.nan for k in [
                    'r_mean', 'r_median', 'r_std', 'r_var',
                    'g_mean', 'g_median', 'g_std', 'g_var',
                    'b_mean', 'b_median', 'b_std', 'b_var',
                    'r_g_mean', 'r_g_std', 'r_b_mean', 'r_b_std', 'g_b_mean', 'g_b_std',
                    'hsv_s_mean', 'hsv_v_mean', 'hsv_h_mean', 'hsv_h_std',
                    'lab_l_mean', 'lab_a_mean', 'lab_b_mean',
                    'vari_mean', 'vari_std'
                ]}
            
            return idx, stats
        except Exception as ex:
            raise ex
    
    stat_cols = ['r_mean', 'r_median', 'r_std', 'r_var', 'g_mean', 'g_median', 'g_std', 'g_var',
                 'b_mean', 'b_median', 'b_std', 'b_var', 'r_g_mean', 'r_g_std', 'r_b_mean', 
                 'r_b_std', 'g_b_mean', 'g_b_std', 'hsv_s_mean', 'hsv_v_mean', 'hsv_h_mean', 
                 'hsv_h_std', 'lab_l_mean', 'lab_a_mean', 'lab_b_mean', 'vari_mean', 'vari_std']
    
    for col in stat_cols:
        gdf[col] = np.nan
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing polygons"):
        _, stats = process_polygon(idx, row)
        for col in stat_cols:
            gdf.at[idx, col] = stats[col]
    
    return gdf
