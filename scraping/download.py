"""
Agricultural Futures Project — Unstructured Data Downloader
Covers ~2006-2026 (15-20 years)

Sources:
  1. NOAA CPC Climate Discussions (weekly narrative text, scraped from CPC website)
  2. USDA NASS Crop Progress & Condition Reports (weekly PDF/text, via NASS publications)
  3. USDA WASDE Reports (monthly PDF, via USDA direct download)
  4. CBOT Futures Prices for Wheat, Corn, Soybeans (via yfinance — structured, for labels)

Requirements:
  pip install requests beautifulsoup4 yfinance pandas tqdm
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─── Output directories ───────────────────────────────────────────────────────
BASE_DIR = "./data"
DIRS = {
    "cpc":    os.path.join(BASE_DIR, "noaa_cpc_discussions"),
    "nass":   os.path.join(BASE_DIR, "usda_crop_progress"),
    "wasde":  os.path.join(BASE_DIR, "usda_wasde"),
    "prices": os.path.join(BASE_DIR, "futures_prices"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; contact: your@email.com)"}
START_YEAR = 2006
END_YEAR   = 2026


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NOAA CPC Climate Discussions
#     URL pattern: https://www.cpc.ncep.noaa.gov/products/predictions/long_range/
#     Archive discussions are plain-text files organized by year on the CPC FTP.
# ═══════════════════════════════════════════════════════════════════════════════

def download_noaa_cpc_discussions():
    """
    Downloads NOAA CPC 6-10 day and 8-14 day outlook discussion text files.
    Files live on the CPC FTP at:
      https://ftp.cpc.ncep.noaa.gov/GFS/discussions/

    Each file is a plain-text climate narrative tagged by date.
    """
    print("\n=== [1/4] NOAA CPC Climate Discussions ===")

    # CPC stores archived seasonal/long-range discussion text files here:
    base_url = "https://ftp.cpc.ncep.noaa.gov/GFS/discussions/"

    # Also scrape the CPC drought monitor narratives (richer agricultural signal):
    drought_url = "https://droughtmonitor.unl.edu/CurrentMap/StateDroughtMonitor.aspx"

    # ── Approach: scrape the CPC "Prognostic Discussion" archive index ─────────
    # These weekly files are at: https://www.cpc.ncep.noaa.gov/products/predictions/
    # long_range/tools/bfcst_discuss.YYYYMM  (older format)
    # Newer: https://www.cpc.ncep.noaa.gov/products/predictions/long_range/fxus07.html

    # We'll download from the publicly available archive index:
    archive_index = "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/text_archives/"

    try:
        resp = requests.get(archive_index, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True)
                 if a["href"].endswith(".txt") or "discuss" in a["href"].lower()]
        print(f"  Found {len(links)} discussion text links in archive index.")

        saved = 0
        for link in tqdm(links, desc="  CPC discussions"):
            url  = link if link.startswith("http") else archive_index + link
            fname = os.path.join(DIRS["cpc"], os.path.basename(link))
            if os.path.exists(fname):
                continue
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    with open(fname, "w", encoding="utf-8", errors="replace") as f:
                        f.write(r.text)
                    saved += 1
                time.sleep(0.3)  # polite delay
            except Exception as e:
                print(f"    WARN: {link} — {e}")

        print(f"  Saved {saved} new files to {DIRS['cpc']}")

    except Exception as e:
        print(f"  ERROR reaching CPC archive index: {e}")
        print("  Fallback: manually browse https://www.cpc.ncep.noaa.gov/products/predictions/long_range/")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  USDA NASS Crop Progress & Condition Reports
#     Weekly PDF/text reports released every Monday during the growing season.
#     National release index: https://usda.library.cornell.edu/concern/publications/
#     8336h188j  (Cornell USDA digital library — most reliable archive)
# ═══════════════════════════════════════════════════════════════════════════════

def download_usda_crop_progress():
    """
    Downloads USDA NASS Crop Progress & Condition weekly reports (PDFs) from
    the Cornell USDA digital library, which archives reports back to the 1980s.

    Direct API endpoint:
      https://usda.library.cornell.edu/concern/publications/8336h188j.json
    Each record has a list of file URLs for each release date.
    """
    print("\n=== [2/4] USDA Crop Progress & Condition Reports ===")

    api_url = "https://usda.library.cornell.edu/concern/publications/8336h188j.json"

    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        releases = data.get("items", [])
        print(f"  Found {len(releases)} release records in Cornell archive.")

        saved = 0
        for item in tqdm(releases, desc="  Crop Progress PDFs"):
            # Each item has a 'label' (date string) and 'files' list
            label = item.get("label", "unknown")
            try:
                year = int(label[:4])
            except Exception:
                continue
            if not (START_YEAR <= year <= END_YEAR):
                continue

            for file_info in item.get("files", []):
                file_url = file_info.get("url", "")
                if not file_url.endswith(".pdf"):
                    continue
                fname = os.path.join(DIRS["nass"], f"crop_progress_{label}.pdf")
                if os.path.exists(fname):
                    continue
                try:
                    r = requests.get(file_url, headers=HEADERS, timeout=30)
                    if r.status_code == 200:
                        with open(fname, "wb") as f:
                            f.write(r.content)
                        saved += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    WARN: {label} — {e}")
                break  # one PDF per release date

        print(f"  Saved {saved} new PDFs to {DIRS['nass']}")

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fallback: manually browse https://usda.library.cornell.edu/concern/publications/8336h188j")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  USDA WASDE Reports (monthly)
#     Official historical CSV:  https://www.usda.gov/historical-wasde-report-data-3
#     PDF archive (text):       https://usda.library.cornell.edu/concern/publications/j3860694x
#
#     The CSV covers April 2010–present (structured).
#     The PDF archive at Cornell covers older issues with full narrative text.
# ═══════════════════════════════════════════════════════════════════════════════

def download_wasde():
    """
    Downloads:
      (a) USDA historical WASDE CSV (structured, 2010-present)
      (b) WASDE PDF reports (narrative text, 2006-present) from Cornell archive
    """
    print("\n=== [3/4] USDA WASDE Reports ===")

    # ── (a) Structured CSV from USDA ──────────────────────────────────────────
    wasde_csv_url = "https://apps.fas.usda.gov/psdonline/downloads/psd_grains_pulses_csv.zip"
    wasde_csv_path = os.path.join(DIRS["wasde"], "wasde_historical.zip")

    # Official consolidated WASDE data download link:
    wasde_direct = "https://www.usda.gov/sites/default/files/documents/wasde-data.zip"

    print("  Downloading WASDE structured CSV archive...")
    for url in [wasde_direct, wasde_csv_url]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            if r.status_code == 200:
                with open(wasde_csv_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Saved WASDE ZIP to {wasde_csv_path}")
                break
        except Exception as e:
            print(f"    WARN trying {url}: {e}")

    # ── (b) PDF narrative text from Cornell USDA digital library ──────────────
    api_url = "https://usda.library.cornell.edu/concern/publications/j3860694x.json"

    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data  = resp.json()
        items = data.get("items", [])
        print(f"  Found {len(items)} WASDE release records in Cornell archive.")

        saved = 0
        for item in tqdm(items, desc="  WASDE PDFs"):
            label = item.get("label", "unknown")
            try:
                year = int(label[:4])
            except Exception:
                continue
            if not (START_YEAR <= year <= END_YEAR):
                continue

            for file_info in item.get("files", []):
                file_url = file_info.get("url", "")
                if not file_url.endswith(".pdf"):
                    continue
                fname = os.path.join(DIRS["wasde"], f"wasde_{label}.pdf")
                if os.path.exists(fname):
                    continue
                try:
                    r = requests.get(file_url, headers=HEADERS, timeout=30)
                    if r.status_code == 200:
                        with open(fname, "wb") as f:
                            f.write(r.content)
                        saved += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    WARN: {label} — {e}")
                break

        print(f"  Saved {saved} new WASDE PDFs to {DIRS['wasde']}")

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fallback: manually browse https://usda.library.cornell.edu/concern/publications/j3860694x")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CBOT Futures Prices — Wheat (ZW), Corn (ZC), Soybeans (ZS)
#     via yfinance (Yahoo Finance); used as prediction targets / labels.
#     Continuous front-month contract tickers on Yahoo Finance:
#       ZW=F (wheat), ZC=F (corn), ZS=F (soybeans)
# ═══════════════════════════════════════════════════════════════════════════════

def download_futures_prices():
    """
    Downloads daily CBOT futures close prices for wheat, corn, and soybeans
    from Yahoo Finance using yfinance. Saves each as a CSV.
    """
    print("\n=== [4/4] CBOT Futures Prices ===")

    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Run: pip install yfinance")
        return

    tickers = {
        "wheat":    "ZW=F",
        "corn":     "ZC=F",
        "soybeans": "ZS=F",
    }

    start = f"{START_YEAR}-01-01"
    end   = f"{END_YEAR}-12-31"

    for name, ticker in tickers.items():
        fname = os.path.join(DIRS["prices"], f"{name}_futures.csv")
        print(f"  Downloading {name} ({ticker}) {start} → {end}...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"    WARN: No data returned for {ticker}.")
            else:
                df.to_csv(fname)
                print(f"    Saved {len(df)} rows to {fname}")
        except Exception as e:
            print(f"    ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(" Agricultural Futures Project — Data Downloader")
    print(f" Coverage: {START_YEAR}–{END_YEAR}")
    print("=" * 60)

    download_noaa_cpc_discussions()
    download_usda_crop_progress()
    download_wasde()
    download_futures_prices()

    print("\n✓ Download complete. Check ./data/ for all files.")
    print("\nNext steps:")
    print("  • Parse PDFs with pdfplumber or pypdf2 to extract text")
    print("  • Embed text with sentence-transformers or OpenAI embeddings")
    print("  • Align all sources by ISO week number for walk-forward training")