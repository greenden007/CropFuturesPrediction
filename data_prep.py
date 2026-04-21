"""
Aggregate futures CSVs and weekly text reports into a single weekly dataset.

Usage:
    python data_prep.py --data-dir ./data --out ./data/dataset.parquet

The script attempts to:
- load `data/futures_prices/*_futures.csv` (expects a Date index or `Date` column)
- produce weekly closes (W-MON) for each commodity
- read text files from `data/usda_crop_progress/` and `data/noaa_cpc_discussions/` and concat per week
- output a merged Parquet dataset with `week_start`, `commodity_close`, `text`, and forward label `return_4w`
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def load_futures(data_dir: Path) -> pd.DataFrame:
    prices_dir = data_dir / "futures_prices"
    dfs = []
    if not prices_dir.exists():
        print(f"No futures directory found at {prices_dir}")
        return pd.DataFrame()

    for fname in ["corn_futures.csv", "soybeans_futures.csv", "wheat_futures.csv"]:
        p = prices_dir / fname
        if not p.exists():
            print(f"  Missing: {p.name}")
            continue
        name = p.stem.replace("_futures", "")
        # Some CSVs have a multi-row header (Price / Ticker / Date rows).
        # Detect the header row that contains the literal 'Date' and read from there.
        header_row = 0
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if line.strip().startswith("Date") or line.split(",")[0].strip() == "Date":
                        header_row = i
                        break
        except Exception:
            header_row = 0

        try:
            df = pd.read_csv(p, header=header_row)
        except Exception:
            # fallback to a simple read
            df = pd.read_csv(p, skiprows=2, header=0)

        # normalize Date column/index
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            # try parsing first column as dates
            try:
                df.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
                df = df.drop(df.columns[0], axis=1)
            except Exception:
                # give up for this file
                print(f"  Could not parse dates in {p.name}")
                continue

        if "Close" in df.columns:
            close = df["Close"].rename(name)
        elif "Adj Close" in df.columns:
            close = df["Adj Close"].rename(name)
        else:
            close = df.iloc[:, 0].rename(name)

        # weekly resample to Monday (week start)
        weekly = close.resample("W-MON").last().to_frame()
        dfs.append(weekly)

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, axis=1)
    merged.index.name = "week_start"
    return merged


def load_texts(data_dir: Path) -> pd.DataFrame:
    # Read .txt and .pdf-extracted text (assume plain .txt for now)
    texts_dir = data_dir / "usda_crop_progress"
    rows = []
    if texts_dir.exists():
        for p in sorted(texts_dir.glob("*.txt")):
            # guess date from filename or from first line
            try:
                # many files like '2006-04-03_CropProg-04-03-2006.txt'
                parts = p.name.split("_")
                datepart = parts[0]
                dt = datetime.fromisoformat(datepart)
            except Exception:
                # fallback to file mtime
                dt = datetime.fromtimestamp(p.stat().st_mtime)
            week = pd.to_datetime(dt).to_period("W").start_time
            text = p.read_text(encoding="utf-8", errors="replace")
            rows.append({"week_start": week, "text_crop_progress": text})

    # CPC discussions
    cpc_dir = data_dir / "noaa_cpc_discussions"
    if cpc_dir.exists():
        for p in sorted(cpc_dir.glob("*.txt")):
            try:
                # try to parse date from filename
                name = p.stem
                # often contains YYYYMMDD
                import re

                m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
                if m:
                    dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                else:
                    dt = datetime.fromtimestamp(p.stat().st_mtime)
            except Exception:
                dt = datetime.fromtimestamp(p.stat().st_mtime)
            week = pd.to_datetime(dt).to_period("W").start_time
            text = p.read_text(encoding="utf-8", errors="replace")
            rows.append({"week_start": week, "text_cpc": text})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Build aggregation dict only for columns that exist to avoid KeyError
    agg_dict = {}
    if "text_crop_progress" in df.columns:
        agg_dict["text_crop_progress"] = lambda s: "\n\n".join(x for x in s if isinstance(x, str))
    if "text_cpc" in df.columns:
        agg_dict["text_cpc"] = lambda s: "\n\n".join(x for x in s if isinstance(x, str))

    if not agg_dict:
        return pd.DataFrame()

    df = df.groupby("week_start").agg(agg_dict)

    # ensure both columns present in the returned frame
    for c in ["text_crop_progress", "text_cpc"]:
        if c not in df.columns:
            df[c] = ""

    return df


def build_dataset(data_dir: Path) -> pd.DataFrame:
    prices = load_futures(data_dir)
    texts = load_texts(data_dir)

    if prices.empty:
        print("No price data available — run scrapers and place CSVs in data/futures_prices/")
    if texts.empty:
        print("No text data found in data/usda_crop_progress or data/noaa_cpc_discussions")

    # Align by week_start index
    if prices.empty and texts.empty:
        return pd.DataFrame()

    parts = [prices]
    if not texts.empty:
        parts.append(texts)

    df = pd.concat(parts, axis=1)
    df = df.sort_index()

    # Create forward 4-week return label for corn if present
    if "corn" in df.columns:
        df["return_4w"] = df["corn"].pct_change(periods=4).shift(-4)
    else:
        df["return_4w"] = np.nan

    # Drop rows with NaN target
    df = df.dropna(subset=["return_4w"]) if "return_4w" in df.columns else df

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="Root data directory")
    parser.add_argument("--out", default="./data/dataset.parquet", help="Output dataset path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)

    ds = build_dataset(data_dir)
    if ds.empty:
        print("No dataset produced. Populate `data/` with scraped files first.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out.suffix.lower() == ".csv":
            ds.to_csv(out)
        else:
            ds.to_parquet(out)
        print(f"Wrote dataset with {len(ds)} rows to {out}")
    except ImportError:
        fallback = out.with_suffix(".csv")
        ds.to_csv(fallback)
        print("Parquet engine not available; wrote CSV instead:", fallback)


if __name__ == "__main__":
    main()
