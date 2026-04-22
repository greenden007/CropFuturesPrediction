#!/usr/bin/env python3
"""
Data Cleaning Script for Crop Futures Prediction Project

Cleans and processes data from:
1. futures_prices/ - Commodity futures OHLCV data
2. usda_wasde/ - USDA supply/demand reports
3. usda_crop_progress/ - Weekly crop progress text reports
4. noaa_cpc_discussions/ - Weather outlook image filenames (references only)

Output: Cleaned CSV files in a processed_data/ directory
"""

import os
import re
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed_data")


def ensure_output_dir():
    """Create output directory structure."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "futures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "wasde").mkdir(exist_ok=True)
    (OUTPUT_DIR / "crop_progress").mkdir(exist_ok=True)
    (OUTPUT_DIR / "weather_outlooks").mkdir(exist_ok=True)


# =============================================================================
# 1. FUTURES PRICES CLEANING
# =============================================================================

def clean_futures_prices():
    """Clean corn, soybeans, and wheat futures CSV files."""
    futures_dir = DATA_DIR / "futures_prices"
    output_subdir = OUTPUT_DIR / "futures"
    
    futures_files = {
        "corn": "corn_futures.csv",
        "soybeans": "soybeans_futures.csv", 
        "wheat": "wheat_futures.csv"
    }
    
    cleaned_data = {}
    
    for commodity, filename in futures_files.items():
        filepath = futures_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue
            
        print(f"Cleaning {commodity} futures...")
        
        # Read raw CSV
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse structure: header row, ticker row, then data
        # Row 1: Price,Close,High,Low,Open,Volume
        # Row 2: Ticker,ZC=F,ZC=F,ZC=F,ZC=F,ZC=F
        # Row 3: Date,,,,,
        # Row 4+: data
        
        data_rows = []
        for line in lines[3:]:  # Skip header/ticker/date rows
            parts = line.strip().split(',')
            if len(parts) >= 6 and parts[0]:
                try:
                    date = pd.to_datetime(parts[0])
                    row = {
                        'date': date.strftime('%Y-%m-%d'),
                        'close': float(parts[1]) if parts[1] else None,
                        'high': float(parts[2]) if parts[2] else None,
                        'low': float(parts[3]) if parts[3] else None,
                        'open': float(parts[4]) if parts[4] else None,
                        'volume': int(float(parts[5])) if parts[5] else None,
                        'commodity': commodity
                    }
                    data_rows.append(row)
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data_rows)
        
        # Clean: remove rows with missing critical values
        df = df.dropna(subset=['date', 'close'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Save cleaned data
        output_file = output_subdir / f"{commodity}_cleaned.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved {len(df)} rows to {output_file}")
        
        cleaned_data[commodity] = df
    
    return cleaned_data


# =============================================================================
# 2. USDA WASDE DATA CLEANING
# =============================================================================

def clean_wasde_historical():
    """Clean historical WASDE report CSV files."""
    wasde_dir = DATA_DIR / "usda_wasde" / "usda_historical_wasde_report_data"
    output_subdir = OUTPUT_DIR / "wasde"
    
    if not wasde_dir.exists():
        print(f"Warning: {wasde_dir} not found")
        return None
    
    # Find all CSV files
    csv_files = sorted(wasde_dir.glob("oce-wasde-report-data-*.csv"))
    
    all_records = []
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Standardize column names
            df.columns = [c.strip().strip('"') for c in df.columns]
            
            # Keep key columns
            key_cols = ['WasdeNumber', 'ReportDate', 'ReportTitle', 'Attribute', 
                       'Commodity', 'Region', 'MarketYear', 'ProjEstFlag',
                       'Value', 'Unit', 'ReleaseDate', 'ForecastYear', 'ForecastMonth']
            
            existing_cols = [c for c in key_cols if c in df.columns]
            df = df[existing_cols]
            
            # Clean numeric values
            if 'Value' in df.columns:
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            # Parse dates
            if 'ReleaseDate' in df.columns:
                df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
            
            all_records.append(df)
            
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            continue
    
    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        
        # Remove duplicates
        combined = combined.drop_duplicates()
        
        # Sort by release date and WASDE number
        if 'ReleaseDate' in combined.columns:
            combined = combined.sort_values(['ReleaseDate', 'WasdeNumber'], ascending=[False, False])
        
        output_file = output_subdir / "wasde_historical_cleaned.csv"
        combined.to_csv(output_file, index=False)
        print(f"Saved {len(combined)} WASDE records to {output_file}")
        
        return combined
    
    return None


def clean_psd_grains():
    """Clean the large PSD grains/pulses CSV."""
    psd_file = DATA_DIR / "usda_wasde" / "psd_grains_pulses.csv"
    output_subdir = OUTPUT_DIR / "wasde"
    
    if not psd_file.exists():
        print(f"Warning: {psd_file} not found")
        return None
    
    print(f"Cleaning PSD grains/pulses data (this may take a moment)...")
    
    # Read in chunks due to large file size
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv(psd_file, chunksize=chunk_size, low_memory=False):
        # Filter for major commodities of interest
        commodities_of_interest = ['Corn', 'Soybeans', 'Wheat', 'Barley', 'Sorghum']
        
        mask = chunk['Commodity_Description'].isin(commodities_of_interest)
        filtered = chunk[mask].copy()
        
        if len(filtered) > 0:
            chunks.append(filtered)
    
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        
        # Clean and standardize
        df = df.drop_duplicates()
        
        # Sort by country, commodity, market year
        df = df.sort_values(['Country_Name', 'Commodity_Description', 'Market_Year', 'Month'])
        
        output_file = output_subdir / "psd_grains_cleaned.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} PSD records to {output_file}")
        
        return df
    
    return None


# =============================================================================
# 3. USDA CROP PROGRESS CLEANING
# =============================================================================

def parse_crop_progress_text(filepath: Path) -> List[Dict]:
    """Parse a single crop progress text file and extract structured data."""
    
    records = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract date from filename or content
    # Filename format: YYYY-MM-DD_CropProg-... or YYYY-MM-DD_progXXXX.txt
    filename = filepath.name
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
    
    if date_match:
        report_date = date_match.group(1)
    else:
        # Try to extract from content
        date_patterns = [
            r'Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]
        report_date = None
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    parsed = pd.to_datetime(match.group(1))
                    report_date = parsed.strftime('%Y-%m-%d')
                    break
                except:
                    continue
    
    if not report_date:
        return records
    
    # Parse crop-specific sections
    # Look for state-by-state progress tables
    
    crops = ['Corn', 'Soybeans', 'Wheat', 'Cotton', 'Sorghum', 'Rice', 'Oats', 'Barley']
    
    for crop in crops:
        # Find sections mentioning this crop
        crop_pattern = rf'{crop}.*?:(?:\s*Percent)?\s*(Planted|Harvested|Emerging|Booting|Heading|Coloring|Mature|Dropped|Complete)'
        
        # Extract table data - look for state abbreviations followed by percentages
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Match state lines with percentages
            # Pattern: State abbrev, then numbers/NA
            state_match = re.match(r'^(AR|CA|LA|MS|MO|TX|IA|IL|IN|OH|MN|WI|MI|NE|KS|SD|ND|CO|OK|NC|SC|GA|FL|AL|TN|KY|WV|VA|PA|NY|VT|NH|ME|MA|RI|CT|NJ|DE|MD|MT|WY|ID|WA|OR|NV|UT|AZ|NM|TX|AK|HI)\s+', line)
            
            if state_match:
                state = state_match.group(1)
                
                # Extract numbers from the line
                parts = line.split()
                if len(parts) >= 2:
                    # Try to find percentages
                    percentages = []
                    for part in parts[1:]:
                        if part.replace('.', '').isdigit():
                            percentages.append(float(part))
                        elif part == 'NA':
                            percentages.append(None)
                    
                    if percentages:
                        record = {
                            'date': report_date,
                            'crop': crop,
                            'state': state,
                            'current_pct': percentages[0] if len(percentages) > 0 else None,
                            'prev_week_pct': percentages[1] if len(percentages) > 1 else None,
                            'prev_year_pct': percentages[2] if len(percentages) > 2 else None,
                            'avg_pct': percentages[3] if len(percentages) > 3 else None,
                        }
                        records.append(record)
    
    return records


def clean_crop_progress():
    """Clean all crop progress text files."""
    progress_dir = DATA_DIR / "usda_crop_progress"
    output_subdir = OUTPUT_DIR / "crop_progress"
    
    if not progress_dir.exists():
        print(f"Warning: {progress_dir} not found")
        return None
    
    # Find all text files
    txt_files = list(progress_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} crop progress files")
    
    all_records = []
    
    for txt_file in sorted(txt_files):
        try:
            records = parse_crop_progress_text(txt_file)
            all_records.extend(records)
        except Exception as e:
            print(f"  Error parsing {txt_file.name}: {e}")
            continue
    
    if all_records:
        df = pd.DataFrame(all_records)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by date
        df = df.sort_values('date')
        
        output_file = output_subdir / "crop_progress_cleaned.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} crop progress records to {output_file}")
        
        return df
    
    return None


# =============================================================================
# 4. NOAA CPC OUTLOOK - FILENAME INDEXING
# =============================================================================

def index_weather_outlooks():
    """Index all weather outlook image filenames without loading images."""
    outlook_dir = DATA_DIR / "noaa_cpc_discussions" / "cpc_outlook_data"
    output_subdir = OUTPUT_DIR / "weather_outlooks"
    
    if not outlook_dir.exists():
        print(f"Warning: {outlook_dir} not found")
        return None
    
    outlook_records = []
    
    # Walk through year/month/day structure
    for year_dir in sorted(outlook_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        
        year = year_dir.name
        
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            
            month = month_dir.name
            
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                
                day = day_dir.name
                outlook_date = f"{year}-{month}-{day[6:8]}" if len(day) >= 8 else f"{year}-{month}-{day}"
                
                # List all GIF files (image filenames only)
                gif_files = list(day_dir.glob("*.gif"))
                
                for gif_file in gif_files:
                    # Parse filename components
                    # Format: {forecast}{var}.{date}.fcst.gif
                    # e.g., 610prcp.20250407.fcst.gif, 814temp.20250407.fcst.gif
                    
                    filename = gif_file.name
                    parts = filename.split('.')
                    
                    if len(parts) >= 3:
                        forecast_var = parts[0]  # e.g., "610prcp", "814temp"
                        file_date = parts[1] if len(parts) > 1 else ""
                        
                        # Parse forecast type and variable
                        forecast_type = ""  # 6-10 day (610) or 8-14 day (814)
                        variable = ""  # prcp, temp, hghts
                        
                        if forecast_var.startswith('610'):
                            forecast_type = "6-10_day"
                            variable = forecast_var[3:]
                        elif forecast_var.startswith('814'):
                            forecast_type = "8-14_day"
                            variable = forecast_var[3:]
                        
                        record = {
                            'outlook_date': outlook_date,
                            'filename': filename,
                            'forecast_type': forecast_type,
                            'variable': variable,  # prcp, temp, hghts
                            'file_date': file_date,
                            'year': year,
                            'month': month,
                            'full_path': str(gif_file.relative_to(DATA_DIR))
                        }
                        outlook_records.append(record)
    
    if outlook_records:
        df = pd.DataFrame(outlook_records)
        
        # Sort by date
        df = df.sort_values('outlook_date')
        
        output_file = output_subdir / "weather_outlook_index.csv"
        df.to_csv(output_file, index=False)
        print(f"Indexed {len(df)} weather outlook files to {output_file}")
        
        # Also create a summary JSON
        summary = {
            'total_files': len(df),
            'date_range': {
                'start': df['outlook_date'].min(),
                'end': df['outlook_date'].max()
            },
            'forecast_types': df['forecast_type'].value_counts().to_dict(),
            'variables': df['variable'].value_counts().to_dict(),
            'years': sorted(df['year'].unique().tolist())
        }
        
        summary_file = output_subdir / "weather_outlook_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")
        
        return df
    
    return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all data cleaning operations."""
    print("=" * 60)
    print("Crop Futures Prediction - Data Cleaning")
    print("=" * 60)
    print()
    
    # Create output directories
    ensure_output_dir()
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()
    
    results = {}
    
    # 1. Clean futures prices
    print("-" * 40)
    print("1. FUTURES PRICES")
    print("-" * 40)
    results['futures'] = clean_futures_prices()
    print()
    
    # 2. Clean WASDE data
    print("-" * 40)
    print("2. USDA WASDE REPORTS")
    print("-" * 40)
    results['wasde_historical'] = clean_wasde_historical()
    results['psd_grains'] = clean_psd_grains()
    print()
    
    # 3. Clean crop progress
    print("-" * 40)
    print("3. USDA CROP PROGRESS")
    print("-" * 40)
    results['crop_progress'] = clean_crop_progress()
    print()
    
    # 4. Index weather outlooks (filenames only)
    print("-" * 40)
    print("4. NOAA WEATHER OUTLOOKS (Filename Index)")
    print("-" * 40)
    results['weather_outlooks'] = index_weather_outlooks()
    print()

    # 5. Extract image features from outlook GIFs
    print("-" * 40)
    print("5. WEATHER OUTLOOK IMAGE FEATURES")
    print("-" * 40)
    outlook_index = OUTPUT_DIR / "weather_outlooks" / "weather_outlook_index.csv"
    image_features_file = OUTPUT_DIR / "weather_outlooks" / "image_features.csv"

    if outlook_index.exists():
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
            from image_encoder import extract_image_features_to_csv

            extract_image_features_to_csv(
                outlook_index_path=outlook_index,
                data_dir=DATA_DIR,
                output_path=image_features_file,
                feature_dim=64,
                batch_size=32
            )
            results['image_features'] = image_features_file
        except Exception as e:
            print(f"Warning: Could not extract image features: {e}")
            print("  Image features will be skipped. Ensure images are downloaded and torch/PIL are installed.")
    else:
        print("  No outlook index found, skipping image feature extraction")
    print()

    # Summary
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print()
    print("Output files created:")
    for output_file in OUTPUT_DIR.rglob("*"):
        if output_file.is_file():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  {output_file.relative_to(OUTPUT_DIR)} ({size_mb:.2f} MB)")
    
    return results


if __name__ == "__main__":
    main()
