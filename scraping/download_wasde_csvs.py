#!/usr/bin/env python3
"""
Download all CSV files from the USDA Historical WASDE Report Data page.
https://www.usda.gov/historical-wasde-report-data-3

Note: The 2010-2020 data is provided as two ZIP archives containing CSVs.
      This script downloads those ZIPs as well and optionally extracts them.
"""

import os
import re
import time
import zipfile
import requests
from pathlib import Path
from urllib.parse import urljoin

# --- Config ---
PAGE_URL = "https://www.usda.gov/historical-wasde-report-data-3"
OUTPUT_DIR = Path("wasde_data")
EXTRACT_ZIPS = True       # Set False to keep ZIPs without extracting
DELAY_SECONDS = 0.5       # Polite delay between requests
# --------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WASDE-downloader/1.0)"
}

def get_all_file_links(page_url: str) -> list[dict]:
    """Fetch the page and extract all .csv and .zip links."""
    print(f"Fetching page: {page_url}")
    resp = requests.get(page_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Match href attributes pointing to .csv or .zip files
    pattern = re.compile(r'href="(https?://[^"]+\.(?:csv|zip))"', re.IGNORECASE)
    urls = pattern.findall(resp.text)

    # Also match link text for labeling (month/year)
    label_pattern = re.compile(
        r'href="(https?://[^"]+\.(?:csv|zip))"[^>]*>\s*([^<]+)',
        re.IGNORECASE
    )
    links = []
    seen = set()
    for match in label_pattern.finditer(resp.text):
        url, label = match.group(1).strip(), match.group(2).strip()
        if url not in seen:
            seen.add(url)
            links.append({"url": url, "label": label})

    # Catch any URLs the label pattern missed
    for url in urls:
        if url not in seen:
            seen.add(url)
            links.append({"url": url, "label": Path(url).name})

    return links


def download_file(url: str, dest_path: Path) -> bool:
    """Download a single file. Returns True on success."""
    if dest_path.exists():
        print(f"  [skip] already exists: {dest_path.name}")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_kb = dest_path.stat().st_size // 1024
        print(f"  [ok]   {dest_path.name}  ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"  [err]  {url} — {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    links = get_all_file_links(PAGE_URL)
    print(f"\nFound {len(links)} file(s) to download.\n")

    success, failed = 0, 0
    for item in links:
        url, label = item["url"], item["label"]
        filename = Path(url).name          # e.g. oce-wasde-report-data-2026-04.csv
        dest = OUTPUT_DIR / filename

        print(f"Downloading: {label}")
        if download_file(url, dest):
            success += 1
            # Extract ZIPs if requested
            if EXTRACT_ZIPS and filename.lower().endswith(".zip"):
                zip_dir = OUTPUT_DIR / filename.replace(".zip", "")
                print(f"  Extracting → {zip_dir}/")
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(zip_dir)
        else:
            failed += 1

        time.sleep(DELAY_SECONDS)

    print(f"\nDone. {success} downloaded, {failed} failed.")
    print(f"Files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
