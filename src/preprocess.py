import asyncio
from urllib.parse import urlparse

import httpx
from tqdm.asyncio import tqdm as atqdm


async def build_image_download_uri(gdf, url_column, max_workers=50):
    semaphore = asyncio.Semaphore(max_workers)

    async def fetch_oam_data(url):
        async with semaphore:
            if not isinstance(url, str):
                return None, None, None

            parsed = urlparse(url)

            if "openaerialmap.org" in parsed.netloc:
                image_id = url.rstrip("/").split("/")[-1].split("?")[0]
                api_url = f"https://api.openaerialmap.org/meta/{image_id}/"

                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.get(api_url)
                        response.raise_for_status()
                        data = response.json()
                        results = data.get("results", {})

                        return (
                            "oam",
                            results.get("uuid"),
                            results.get("properties", {}).get("tms"),
                        )
                except (httpx.HTTPError, ValueError, KeyError):
                    return "oam", None, None

            elif (
                "drive.google.com" in parsed.netloc
                or "docs.google.com" in parsed.netloc
            ):
                return "gdrive", url, None

            return "other", None, None

    urls = gdf[url_column].tolist()

    tasks = [fetch_oam_data(url) for url in urls]
    results = []

    for coro in atqdm.as_completed(
        tasks, total=len(tasks), desc="Processing imagery links"
    ):
        results.append(await coro)

    gdf["url_type"] = [r[0] for r in results]
    gdf["download_url"] = [r[1] for r in results]
    gdf["tms_url"] = [r[2] for r in results]

    return gdf


def build_image_download_uri_sync(gdf, url_column, max_workers=50):
    return asyncio.run(build_image_download_uri(gdf, url_column, max_workers))
