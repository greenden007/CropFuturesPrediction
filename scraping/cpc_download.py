#!/usr/bin/env python3
"""
CPC 6-10 Day & 8-14 Day Outlook Archive Downloader
====================================================
Downloads temperature and precipitation outlook data/graphics from:
https://www.cpc.ncep.noaa.gov/products/archives/short_range/srarc.ind.php

Iterates week-by-week from 2006-01-02 to today, saving files in an
organized directory structure: output_dir/YYYY/MM/YYYYMMDD/

Usage:
    python cpc_outlook_downloader.py
    python cpc_outlook_downloader.py --start 2010-01-03 --end 2015-12-28
    python cpc_outlook_downloader.py --output ./cpc_data --dry-run
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.cpc.ncep.noaa.gov/products/archives/short_range"

DEFAULT_OUTPUT_DIR = "./cpc_outlook_data"
DEFAULT_START_DATE = date(2006, 1, 2)   # first full week of 2006
WEEK_DELTA = timedelta(days=7)

# Pause between requests to be polite to the server (seconds)
REQUEST_DELAY = 1.0

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CPC-Archive-Downloader/1.0; "
        "Research use; contact: your@email.com)"
    )
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cpc_downloader.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_with_retry(session: requests.Session, url: str, **kwargs) -> requests.Response | None:
    """GET a URL with automatic retries on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            if status == 404:
                log.debug("404 – not found: %s", url)
                return None          # don't retry 404s
            log.warning("HTTP %s on attempt %d/%d: %s", status, attempt, MAX_RETRIES, url)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d/%d: %s – %s", attempt, MAX_RETRIES, url, exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    log.error("All %d attempts failed for: %s", MAX_RETRIES, url)
    return None


def save_file(content: bytes, dest: Path) -> bool:
    """Write bytes to dest, creating parent dirs as needed."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return True
    except OSError as exc:
        log.error("Could not save %s: %s", dest, exc)
        return False


def construct_urls(forecast_date: date) -> list[str]:
    """
    Construct direct URLs for 610 and 814 outlook files for a given date.
    Pattern: BASE_URL/YYYY/MM/DD/{610,814}{temp,prcp,hghts}.YYYYMMDD.fcst.gif
    """
    year = forecast_date.strftime("%Y")
    month = forecast_date.strftime("%m")
    day = forecast_date.strftime("%d")
    yyyymmdd = forecast_date.strftime("%Y%m%d")
    
    base_path = f"{BASE_URL}/{year}/{month}/{day}"
    
    file_types = ["temp", "prcp", "hghts"]
    outlooks = ["610", "814"]
    
    urls = []
    for outlook in outlooks:
        for ftype in file_types:
            filename = f"{outlook}{ftype}.{yyyymmdd}.fcst.gif"
            url = f"{base_path}/{filename}"
            urls.append(url)
    
    return urls


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def download_day(
    session: requests.Session,
    forecast_date: date,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Download all outlook files for a single forecast date.

    Returns a summary dict with counts.
    """
    date_str = forecast_date.strftime("%Y%m%d")
    day_dir = output_dir / forecast_date.strftime("%Y") / forecast_date.strftime("%m") / date_str

    summary = {"date": date_str, "found": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    urls = construct_urls(forecast_date)
    summary["found"] = len(urls)

    for url in urls:
        filename = os.path.basename(url)
        dest = day_dir / filename

        if dest.exists():
            log.debug("%s – already exists, skipping: %s", date_str, filename)
            summary["skipped"] += 1
            continue

        if dry_run:
            log.info("[DRY-RUN] %s – would download: %s", date_str, url)
            summary["downloaded"] += 1
            continue

        file_resp = fetch_with_retry(session, url)
        if file_resp is None:
            summary["errors"] += 1
            continue

        if save_file(file_resp.content, dest):
            log.debug("%s – saved: %s", date_str, filename)
            summary["downloaded"] += 1
        else:
            summary["errors"] += 1

        time.sleep(REQUEST_DELAY)

    log.info(
        "%s – found=%d  downloaded=%d  skipped=%d  errors=%d",
        date_str,
        summary["found"],
        summary["downloaded"],
        summary["skipped"],
        summary["errors"],
    )
    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CPC 6-10 & 8-14 day outlook archives week by week."
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE.isoformat(),
        help=f"Start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds to wait between file downloads (default: {REQUEST_DELAY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without saving anything",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_date = date.fromisoformat(args.start)
    end_date   = date.fromisoformat(args.end)
    output_dir = Path(args.output)
    global REQUEST_DELAY
    REQUEST_DELAY = args.delay

    if start_date > end_date:
        log.error("--start must be before --end")
        sys.exit(1)

    # Build the list of weekly dates
    weekly_dates: list[date] = []
    current = start_date
    while current <= end_date:
        weekly_dates.append(current)
        current += WEEK_DELTA

    total_weeks = len(weekly_dates)
    log.info("=" * 60)
    log.info("CPC Outlook Archive Downloader")
    log.info("  Date range : %s → %s", start_date, end_date)
    log.info("  Weeks      : %d", total_weeks)
    log.info("  Output dir : %s", output_dir.resolve())
    log.info("  Dry run    : %s", args.dry_run)
    log.info("=" * 60)

    session = requests.Session()

    totals = {"found": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    for idx, forecast_date in enumerate(weekly_dates, start=1):
        log.info("── Week %d/%d  (%s) ──", idx, total_weeks, forecast_date)
        summary = download_day(session, forecast_date, output_dir, dry_run=args.dry_run)

        for key in totals:
            totals[key] += summary[key]

        # Polite pause between date queries (not just file downloads)
        if idx < total_weeks:
            time.sleep(REQUEST_DELAY)

    log.info("=" * 60)
    log.info("DONE – totals across %d weeks:", total_weeks)
    log.info("  Links found  : %d", totals["found"])
    log.info("  Downloaded   : %d", totals["downloaded"])
    log.info("  Skipped      : %d", totals["skipped"])
    log.info("  Errors       : %d", totals["errors"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()