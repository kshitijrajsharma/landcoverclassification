import asyncio
from pathlib import Path
import gdown
import httpx
from tqdm.asyncio import tqdm as atqdm


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


async def build_image_download_uri(gdf, url_column, max_workers=50):
    semaphore = asyncio.Semaphore(max_workers)

    async def fetch_oam_data(url):
        async with semaphore:
            if not isinstance(url, str):
                return None, None

            if "map.openaerialmap.org" in url:
                uuid = url.rstrip('/').split('/')[-1].split('?')[0]
                api_url = f"https://api.openaerialmap.org/meta/{uuid}"

                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.get(api_url)
                        response.raise_for_status()
                        data = response.json()
                        return "oam", data.get("results", {}).get("uuid")
                except (httpx.HTTPError, ValueError, KeyError):
                    return "oam", None

            elif "drive.google.com" in url or "docs.google.com" in url:
                return "gdrive", url

            return "other", None

    urls = gdf[url_column].tolist()
    tasks = [fetch_oam_data(url) for url in urls]
    
    results = []
    for task in atqdm(tasks, desc="Processing imagery links"):
        results.append(await task)

    gdf["url_type"] = [r[0] for r in results]
    gdf["download_url"] = [r[1] for r in results]

    return gdf


def build_image_download_uri_sync(gdf, url_column, max_workers=50):
    return asyncio.run(build_image_download_uri(gdf, url_column, max_workers))


async def download_images(gdf, output_dir, max_workers=20):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_workers)
    unique_images = gdf[["image_uid", "download_url", "url_type"]].drop_duplicates(
        "image_uid"
    )

    async def download_single(row):
        async with semaphore:
            image_uid = row["image_uid"]
            download_url = row["download_url"]
            url_type = row["url_type"]

            if not image_uid or not download_url:
                return None

            output_file = output_path / f"{image_uid}.tif"

            if output_file.exists():
                return str(output_file)

            try:
                if url_type == "oam":
                    async with httpx.AsyncClient(timeout=300) as client:
                        response = await client.get(download_url)
                        response.raise_for_status()
                        output_file.write_bytes(response.content)
                        return str(output_file)

                elif url_type == "gdrive":
                    await asyncio.to_thread(
                        gdown.download,
                        download_url,
                        str(output_file),
                        quiet=False,
                        fuzzy=True,
                    )
                    return str(output_file)

            except Exception as e:
                print(f"Failed {image_uid}: {e}")
                return None

    tasks = [download_single(row) for _, row in unique_images.iterrows()]
    results = []

    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Downloading images"):
        results.append(await coro)

    return [r for r in results if r]


def download_images_sync(gdf, output_dir, max_workers=20):
    return asyncio.run(download_images(gdf, output_dir, max_workers))
