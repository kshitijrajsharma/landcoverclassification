import hashlib

import pandas as pd


def assign_image_uids(gdf, url_column="download_url"):
    url_to_uid = {}

    def get_uid(url):
        if pd.isna(url) or url is None:
            return None
        if url not in url_to_uid:
            hash_suffix = hashlib.md5(url.encode()).hexdigest()[:8]
            uid_count = len(url_to_uid)
            url_to_uid[url] = f"img_{uid_count:04d}_{hash_suffix}"
        return url_to_uid[url]

    gdf["image_uid"] = gdf[url_column].apply(get_uid)

    return gdf, url_to_uid
