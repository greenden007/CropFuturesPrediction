#!/usr/bin/env python3
"""
Download all Crop Progress .txt files from 2006 onward
from https://esmis.nal.usda.gov/publication/crop-progress

Usage:
    python download_crop_progress.py [--output-dir ./crop_progress_txt] [--dry-run]
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)


BASE_URL = "https://esmis.nal.usda.gov/publication/crop-progress"
CUTOFF_YEAR = 2006
DELAY_SECONDS = 1.0  # polite crawl delay between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; crop-progress-downloader/1.0; "
        "+https://github.com/your-repo)"
    )
}


def parse_date(date_str: str) -> datetime | None:
    """Parse a date string like 'Apr 13 2026' into a datetime object."""
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def get_page(session: requests.Session, page: int) -> BeautifulSoup | None:
    """Fetch a single paginated results page and return parsed HTML."""
    url = BASE_URL if page == 0 else f"{BASE_URL}?page={page}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch page {page}: {e}")
        return None


def extract_txt_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """
    Extract (date_str, txt_url) pairs from a results page table.
    Returns an empty list if no rows are found.
    """
    results = []
    table = soup.find("table")
    if not table:
        return results

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_cell = cells[0].get_text(strip=True)

        # Find all links in the row; keep only .txt ones
        for a in row.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".txt"):
                full_url = urljoin("https://esmis.nal.usda.gov", href)
                results.append((date_cell, full_url))

    return results


def get_last_page(soup: BeautifulSoup) -> int:
    """Determine the last page number from the pagination widget."""
    # Look for a link labelled 'Last' which points to ?page=N
    last_link = soup.find("a", string=re.compile(r"Last", re.I))
    if last_link and last_link.get("href"):
        m = re.search(r"page=(\d+)", last_link["href"])
        if m:
            return int(m.group(1))

    # Fallback: find all page= links and take the max
    page_nums = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]page=(\d+)", a["href"])
        if m:
            page_nums.append(int(m.group(1)))
    return max(page_nums) if page_nums else 0


def collect_all_links(session: requests.Session) -> list[tuple[datetime, str, str]]:
    """
    Scrape every paginated results page and collect txt links from 2006+.
    Returns a list of (date, date_str, url) tuples, newest first.
    Stops scraping when all entries on a page are before CUTOFF_YEAR.
    """
    print(f"Fetching page 0 to determine total pages …")
    soup0 = get_page(session, 0)
    if soup0 is None:
        print("[FATAL] Could not load the first page.")
        sys.exit(1)

    last_page = get_last_page(soup0)
    print(f"Total pages found: {last_page + 1}")

    all_links: list[tuple[datetime, str, str]] = []

    for page in range(0, last_page + 1):
        if page > 0:
            soup = get_page(session, page)
            time.sleep(DELAY_SECONDS)
        else:
            soup = soup0

        if soup is None:
            print(f"  [WARN] Skipping page {page} due to fetch error.")
            continue

        links = extract_txt_links(soup)
        if not links:
            print(f"  Page {page:3d}: no table rows found, stopping.")
            break

        page_dates = []
        for date_str, url in links:
            dt = parse_date(date_str)
            if dt is None:
                # Try pulling year from URL as a last resort
                m = re.search(r"(\d{4})", date_str)
                year = int(m.group(1)) if m else 0
                if year < CUTOFF_YEAR:
                    continue
                all_links.append((datetime(year, 1, 1), date_str, url))
                page_dates.append(year)
            else:
                page_dates.append(dt.year)
                if dt.year >= CUTOFF_YEAR:
                    all_links.append((dt, date_str, url))

        oldest_on_page = min(page_dates) if page_dates else 9999
        newest_on_page = max(page_dates) if page_dates else 0
        kept = len(links)  # count before filter
        print(
            f"  Page {page:3d}: years {oldest_on_page}–{newest_on_page}, "
            f"{len([x for x in page_dates if x >= CUTOFF_YEAR])} links kept"
        )

        # Once every entry on the page predates the cutoff, we're done
        if oldest_on_page < CUTOFF_YEAR and newest_on_page < CUTOFF_YEAR:
            print(f"  All entries before {CUTOFF_YEAR}, stopping pagination.")
            break

    return all_links


def download_file(
    session: requests.Session,
    url: str,
    dest_path: Path,
    dry_run: bool = False,
) -> bool:
    """Download a single file. Returns True on success."""
    if dest_path.exists():
        print(f"    [SKIP] Already exists: {dest_path.name}")
        return True

    if dry_run:
        print(f"    [DRY RUN] Would download: {url}")
        return True

    try:
        resp = session.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"    [ERROR] {url}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download USDA Crop Progress .txt files from {CUTOFF_YEAR} onward."
    )
    parser.add_argument(
        "--output-dir",
        default="./crop_progress_txt",
        help="Directory to save downloaded files (default: ./crop_progress_txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_SECONDS,
        help=f"Seconds to wait between page requests (default: {DELAY_SECONDS})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving files to: {output_dir.resolve()}")
    else:
        print("DRY RUN — no files will be written.")

    session = requests.Session()

    # ── Step 1: Collect all txt links ────────────────────────────────────────
    print(f"\nScraping release pages (cutoff: {CUTOFF_YEAR}+) …\n")
    all_links = collect_all_links(session)
    print(f"\nFound {len(all_links)} .txt file(s) from {CUTOFF_YEAR} onward.\n")

    if not all_links:
        print("Nothing to download.")
        return

    # Sort newest → oldest so progress is intuitive
    all_links.sort(key=lambda x: x[0], reverse=True)

    # ── Step 2: Download each file ────────────────────────────────────────────
    success = 0
    skipped = 0
    failed = 0

    for i, (dt, date_str, url) in enumerate(all_links, start=1):
        filename = url.split("/")[-1]
        # Prefix with ISO date for easy sorting on disk
        iso_date = dt.strftime("%Y-%m-%d")
        dest_path = output_dir / f"{iso_date}_{filename}"

        print(f"[{i:4d}/{len(all_links)}] {date_str:15s}  {filename}")
        ok = download_file(session, url, dest_path, dry_run=args.dry_run)

        if ok:
            if dest_path.exists() or args.dry_run:
                # Check if it was already there (size would be > 0 before)
                success += 1
            else:
                success += 1
        else:
            failed += 1

        if not args.dry_run and i % 10 == 0:
            time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Done.  Total: {len(all_links)}  |  OK: {success}  |  Failed: {failed}")
    if not args.dry_run:
        print(f"Files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
