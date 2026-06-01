#!/usr/bin/env python3
"""
Merge all processed data into unified dataframes.

Creates two merged datasets:
1. daily_unified.csv - Daily granularity with futures + weather outlooks + aggregated WASDE/crop progress
2. weekly_unified.csv - Weekly granularity with all data sources aligned to weekly periods
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


PROCESSED_DIR = Path("processed_data")
OUTPUT_DIR = Path("merged_data")


def ensure_output_dir():
    """Create output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_futures_data() -> Optional[pd.DataFrame]:
    """Load and combine all futures data."""
    futures_dir = PROCESSED_DIR / "futures"
    
    if not futures_dir.exists():
        print("Futures data not found")
        return None
    
    all_futures = []
    for commodity in ["corn", "soybeans", "wheat"]:
        filepath = futures_dir / f"{commodity}_cleaned.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            all_futures.append(df)
    
    if not all_futures:
        return None
    
    combined = pd.concat(all_futures, ignore_index=True)
    
    # Pivot to wide format: one row per date, columns for each commodity
    pivot_data = {}
    for commodity in combined['commodity'].unique():
        commodity_df = combined[combined['commodity'] == commodity].copy()
        commodity_df = commodity_df.sort_values('date')
        
        # Rename columns with commodity prefix
        col_prefix = commodity[:3].lower()  # corn -> cor, soybeans -> soy, wheat -> whe
        rename_map = {
            'close': f'{col_prefix}_close',
            'high': f'{col_prefix}_high',
            'low': f'{col_prefix}_low',
            'open': f'{col_prefix}_open',
            'volume': f'{col_prefix}_volume'
        }
        commodity_df = commodity_df.rename(columns=rename_map)
        pivot_data[commodity] = commodity_df[['date'] + list(rename_map.values())]
    
    # Merge all commodities on date
    result = None
    for commodity, df in pivot_data.items():
        if result is None:
            result = df
        else:
            result = pd.merge(result, df, on='date', how='outer')
    
    result = result.sort_values('date')
    print(f"Loaded futures: {len(result)} rows, {len(result.columns)} columns")
    return result


def load_wasde_data() -> Optional[pd.DataFrame]:
    """Load and process WASDE data for merging."""
    wasde_dir = PROCESSED_DIR / "wasde"
    
    if not wasde_dir.exists():
        print("WASDE data not found")
        return None
    
    # Load historical WASDE
    historical_file = wasde_dir / "wasde_historical_cleaned.csv"
    wasde_records = []
    
    if historical_file.exists():
        df = pd.read_csv(historical_file, low_memory=False)
        df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
        
        # Filter for relevant commodities
        relevant_commodities = ['Corn', 'Soybeans', 'Wheat', 'Coarse Grain']
        df = df[df['Commodity'].isin(relevant_commodities)]
        
        # Focus on key attributes
        key_attributes = ['Production', 'Exports', 'Imports', 'Domestic', 
                         'Ending Stocks', 'Beginning Stocks']
        df = df[df['Attribute'].isin(key_attributes)]
        
        # Create pivot: date x commodity_attribute
        for _, row in df.iterrows():
            if pd.isna(row['ReleaseDate']) or pd.isna(row['Value']):
                continue
            
            commodity = row['Commodity'].lower().replace(' ', '_')
            attribute = row['Attribute'].lower().replace(' ', '_')
            col_name = f"wasde_{commodity}_{attribute}"
            
            wasde_records.append({
                'date': row['ReleaseDate'],
                'column': col_name,
                'value': row['Value']
            })
    
    if not wasde_records:
        return None
    
    # Convert to wide format
    wasde_df = pd.DataFrame(wasde_records)
    wasde_pivot = wasde_df.pivot_table(
        index='date', 
        columns='column', 
        values='value',
        aggfunc='first'
    ).reset_index()
    
    print(f"Loaded WASDE: {len(wasde_pivot)} rows, {len(wasde_pivot.columns)} columns")
    return wasde_pivot


def load_crop_progress_data() -> Optional[pd.DataFrame]:
    """Load and aggregate crop progress data."""
    progress_file = PROCESSED_DIR / "crop_progress" / "crop_progress_cleaned.csv"
    
    if not progress_file.exists():
        print("Crop progress data not found")
        return None
    
    df = pd.read_csv(progress_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate: national averages and key states
    # Key corn states: IA, IL, IN, NE, MN
    # Key soybean states: IA, IL, IN, MN, NE
    # Key wheat states: KS, ND, TX, MT, WA
    
    key_states = {
        'corn': ['IA', 'IL', 'IN', 'NE', 'MN'],
        'soybeans': ['IA', 'IL', 'IN', 'MN', 'NE'],
        'wheat': ['KS', 'ND', 'TX', 'MT', 'WA']
    }
    
    records = []
    
    # Aggregate by crop and date
    for crop in df['crop'].unique():
        crop_df = df[df['crop'] == crop].copy()
        
        # National average (all states)
        national = crop_df.groupby('date').agg({
            'current_pct': 'mean',
            'prev_year_pct': 'mean'
        }).reset_index()
        national['metric'] = f'{crop.lower()}_national_pct'
        
        # Key states average
        states = key_states.get(crop.lower(), [])
        if states:
            key_states_df = crop_df[crop_df['state'].isin(states)]
            key_avg = key_states_df.groupby('date').agg({
                'current_pct': 'mean',
                'prev_year_pct': 'mean'
            }).reset_index()
            key_avg['metric'] = f'{crop.lower()}_keystates_pct'
            national = pd.concat([national, key_avg])
        
        records.append(national)
    
    if not records:
        return None
    
    combined = pd.concat(records, ignore_index=True)
    
    # Pivot to wide format
    pivot = combined.pivot_table(
        index='date',
        columns='metric',
        values=['current_pct', 'prev_year_pct'],
        aggfunc='first'
    )
    
    # Flatten column names
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reset_index()
    
    print(f"Loaded crop progress: {len(pivot)} rows, {len(pivot.columns)} columns")
    return pivot


def load_weather_outlook_data() -> Optional[pd.DataFrame]:
    """Load weather outlook index and create features."""
    outlook_file = PROCESSED_DIR / "weather_outlooks" / "weather_outlook_index.csv"
    image_features_file = PROCESSED_DIR / "weather_outlooks" / "image_features.csv"

    if not outlook_file.exists():
        print("Weather outlook data not found")
        return None

    df = pd.read_csv(outlook_file)
    df['outlook_date'] = pd.to_datetime(df['outlook_date'])

    # Create features: count of outlooks per week, by type/variable
    weekly_counts = df.groupby(['outlook_date', 'forecast_type', 'variable']).size().reset_index()
    weekly_counts.columns = ['date', 'forecast_type', 'variable', 'count']

    # Pivot to wide format
    pivot = weekly_counts.pivot_table(
        index='date',
        columns=['forecast_type', 'variable'],
        values='count',
        fill_value=0
    )

    # Flatten column names
    pivot.columns = [f"outlook_{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reset_index()

    # Add binary flags for outlook availability
    pivot['has_6_10_day_outlook'] = (pivot.get('outlook_6-10_day_prcp', 0) > 0).astype(int)
    pivot['has_8_14_day_outlook'] = (pivot.get('outlook_8-14_day_prcp', 0) > 0).astype(int)

    # Merge image features if available
    if image_features_file.exists():
        print("  Loading image features...")
        img_df = pd.read_csv(image_features_file)
        img_df['date'] = pd.to_datetime(img_df['date'])
        pivot = pd.merge(pivot, img_df, on='date', how='left')
        img_feat_cols = [c for c in img_df.columns if c.startswith('img_feat_')]
        print(f"  Added {len(img_feat_cols)} image feature columns")

    print(f"Loaded weather outlooks: {len(pivot)} rows, {len(pivot.columns)} columns")
    return pivot


def create_daily_unified(futures_df: pd.DataFrame, 
                         wasde_df: Optional[pd.DataFrame],
                         weather_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create daily unified dataframe with forward-filled WASDE data."""
    
    # Start with futures as base (daily granularity)
    unified = futures_df.copy()
    
    # Merge WASDE using asof join (forward fill WASDE values to daily)
    if wasde_df is not None:
        wasde_df = wasde_df.sort_values('date')
        unified = unified.sort_values('date')
        
        # Use merge_asof to forward-fill WASDE values
        wasde_cols = [c for c in wasde_df.columns if c != 'date']
        
        for col in wasde_cols:
            temp_df = wasde_df[['date', col]].dropna()
            if len(temp_df) > 0:
                unified = pd.merge_asof(
                    unified, 
                    temp_df, 
                    on='date', 
                    direction='backward',
                    suffixes=('', f'_wasde')
                )
                # Rename to original column name
                if f'{col}_wasde' in unified.columns:
                    unified = unified.rename(columns={f'{col}_wasde': col})
    
    # Merge weather outlooks (these are weekly, use asof)
    if weather_df is not None:
        weather_df = weather_df.sort_values('date')
        unified = unified.sort_values('date')
        
        weather_cols = [c for c in weather_df.columns if c != 'date']
        
        for col in weather_cols:
            temp_df = weather_df[['date', col]].dropna()
            if len(temp_df) > 0:
                unified = pd.merge_asof(
                    unified,
                    temp_df,
                    on='date',
                    direction='backward'
                )
    
    # Add time features
    unified['year'] = unified['date'].dt.year
    unified['month'] = unified['date'].dt.month
    unified['day_of_year'] = unified['date'].dt.dayofyear
    unified['week_of_year'] = unified['date'].dt.isocalendar().week
    unified['is_month_end'] = unified['date'].dt.is_month_end.astype(int)
    
    return unified


def create_weekly_unified(daily_df: pd.DataFrame,
                          crop_progress_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create weekly unified dataframe aggregating daily futures to weekly."""
    
    # Create weekly date column
    daily_df['week_start'] = daily_df['date'] - pd.to_timedelta(
        daily_df['date'].dt.dayofweek, unit='D'
    )
    
    # Aggregate futures to weekly
    price_cols = [c for c in daily_df.columns if any(x in c for x in ['_close', '_high', '_low', '_open'])]
    volume_cols = [c for c in daily_df.columns if '_volume' in c]
    
    weekly_agg = {}
    
    # Price aggregations
    for col in price_cols:
        if '_close' in col:
            weekly_agg[col] = 'last'  # End of week close
        elif '_high' in col:
            weekly_agg[col] = 'max'   # Weekly high
        elif '_low' in col:
            weekly_agg[col] = 'min'   # Weekly low
        elif '_open' in col:
            weekly_agg[col] = 'first'  # Week start open
    
    # Volume aggregation
    for col in volume_cols:
        weekly_agg[col] = 'sum'  # Weekly total volume
    
    # Aggregate
    weekly = daily_df.groupby('week_start').agg(weekly_agg).reset_index()
    weekly = weekly.rename(columns={'week_start': 'date'})
    
    # Add WASDE columns (already forward-filled in daily, take last of week)
    wasde_cols = [c for c in daily_df.columns if 'wasde_' in c]
    for col in wasde_cols:
        wasde_weekly = daily_df.groupby('week_start')[col].last().reset_index()
        weekly = pd.merge(weekly, wasde_weekly, left_on='date', right_on='week_start', how='left')
        weekly = weekly.drop('week_start', axis=1)
    
    # Add weather outlook columns
    weather_cols = [c for c in daily_df.columns if 'outlook_' in c or 'has_' in c]
    for col in weather_cols:
        weather_weekly = daily_df.groupby('week_start')[col].max().reset_index()
        weekly = pd.merge(weekly, weather_weekly, left_on='date', right_on='week_start', how='left')
        weekly = weekly.drop('week_start', axis=1)
    
    # Merge crop progress
    if crop_progress_df is not None:
        # Create week start for crop progress
        crop_progress_df['week_start'] = crop_progress_df['date'] - pd.to_timedelta(
            crop_progress_df['date'].dt.dayofweek, unit='D'
        )
        
        # Aggregate crop progress to weekly (in case of multiple reports per week)
        # Select columns excluding original date to avoid duplicates
        cols_to_keep = [c for c in crop_progress_df.columns if c not in ['date', 'week_start']]
        crop_weekly = crop_progress_df.groupby('week_start')[cols_to_keep].first().reset_index()
        crop_weekly = crop_weekly.rename(columns={'week_start': 'date'})
        
        weekly = pd.merge(weekly, crop_weekly, on='date', how='left', suffixes=('', '_crop'))
    
    # Add time features
    weekly['year'] = weekly['date'].dt.year
    weekly['month'] = weekly['date'].dt.month
    weekly['week_of_year'] = weekly['date'].dt.isocalendar().week
    
    return weekly


def main():
    """Run data merging pipeline."""
    print("=" * 60)
    print("Data Merging Pipeline")
    print("=" * 60)
    print()
    
    ensure_output_dir()
    
    # Load all data sources
    print("Loading processed data...")
    print("-" * 40)
    
    futures_df = load_futures_data()
    wasde_df = load_wasde_data()
    crop_progress_df = load_crop_progress_data()
    weather_df = load_weather_outlook_data()
    
    print()
    
    # Check if we have minimum required data
    if futures_df is None:
        print("ERROR: Futures data is required but not found!")
        return
    
    # Create daily unified
    print("-" * 40)
    print("Creating DAILY unified dataframe...")
    daily_unified = create_daily_unified(futures_df, wasde_df, weather_df)
    
    # Save daily
    daily_file = OUTPUT_DIR / "daily_unified.csv"
    daily_unified.to_csv(daily_file, index=False)
    print(f"Saved: {daily_file} ({len(daily_unified)} rows, {len(daily_unified.columns)} columns)")
    
    # Create weekly unified
    print()
    print("-" * 40)
    print("Creating WEEKLY unified dataframe...")
    weekly_unified = create_weekly_unified(daily_unified, crop_progress_df)
    
    # Save weekly
    weekly_file = OUTPUT_DIR / "weekly_unified.csv"
    weekly_unified.to_csv(weekly_file, index=False)
    print(f"Saved: {weekly_file} ({len(weekly_unified)} rows, {len(weekly_unified.columns)} columns)")
    
    # Create summary
    print()
    print("=" * 60)
    print("MERGING COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Daily unified: {len(daily_unified):,} rows x {len(daily_unified.columns)} columns")
    print(f"  Weekly unified: {len(weekly_unified):,} rows x {len(weekly_unified.columns)} columns")
    print()
    print("Key columns in daily unified:")
    for col in daily_unified.columns[:20]:
        missing = daily_unified[col].isna().sum()
        print(f"  - {col}: {missing:,} missing ({100*missing/len(daily_unified):.1f}%)")
    if len(daily_unified.columns) > 20:
        print(f"  ... and {len(daily_unified.columns) - 20} more columns")
    print()
    print(f"Output files in: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
