#!/usr/bin/env python
"""
Script to convert tourism_monthly_dataset.tsf to a format usable by our transformer model.

The TSF format has a header with metadata and then time series data.
We'll parse this and convert to a CSV format with columns:
- unique_id: Series identifier
- ds: Datetime index 
- y: The time series values
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_tsf_file(file_path):
    """
    Parse a TSF file and extract the time series data.
    
    Args:
        file_path: Path to the TSF file
        
    Returns:
        List of dictionaries, each containing a time series
    """
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
    
    # Find the line where the data starts
    data_start_line = 0
    for i, line in enumerate(lines):
        if line.strip() == "@data":
            data_start_line = i + 1
            break
    
    if data_start_line == 0:
        raise ValueError("Could not find @data marker in the TSF file")
    
    # Extract metadata
    metadata = {}
    for i in range(data_start_line - 1):
        line = lines[i].strip()
        if line.startswith('@'):
            parts = line.split(' ', 1)
            if len(parts) == 2:
                key = parts[0][1:]  # Remove @ prefix
                value = parts[1]
                metadata[key] = value
    
    # Extract frequency
    frequency = metadata.get('frequency', 'monthly')
    horizon = int(metadata.get('horizon', '24'))
    
    # Parse the time series data
    series_list = []
    for i in range(data_start_line, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            continue
            
        # Split at the first colon to separate series_name:start_timestamp:values
        parts = line.split(':', 2)
        if len(parts) < 3:
            logger.warning(f"Skipping invalid line: {line}")
            continue
            
        series_id = parts[0]
        start_timestamp = parts[1]
        values_str = parts[2]
        
        # Parse values
        values = [float(v) for v in values_str.split(',')]
        
        # Parse start timestamp
        try:
            start_date = datetime.strptime(start_timestamp, '%Y-%m-%d %H-%M-%S')
        except ValueError:
            logger.warning(f"Invalid timestamp format for {series_id}: {start_timestamp}")
            start_date = datetime.strptime('1980-01-01', '%Y-%m-%d')
        
        series_list.append({
            'series_id': series_id,
            'start_date': start_date,
            'values': values,
            'frequency': frequency
        })
    
    return series_list, horizon

def convert_to_df(series_list):
    """
    Convert a list of time series dictionaries to a DataFrame.
    
    Args:
        series_list: List of dictionaries, each containing a time series
        
    Returns:
        DataFrame with columns [unique_id, ds, y]
    """
    rows = []
    
    for series in series_list:
        series_id = series['series_id']
        start_date = series['start_date']
        values = series['values']
        frequency = series['frequency']
        
        # Create date range
        if frequency == 'monthly':
            freq = 'MS'  # Month start
        elif frequency == 'yearly':
            freq = 'YS'  # Year start
        elif frequency == 'quarterly':
            freq = 'QS'  # Quarter start
        else:
            freq = 'MS'  # Default to monthly
            
        dates = pd.date_range(start=start_date, periods=len(values), freq=freq)
        
        # Create rows
        for date, value in zip(dates, values):
            rows.append({
                'unique_id': series_id,
                'ds': date,
                'y': value
            })
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description='Convert TSF file to CSV format')
    parser.add_argument('--input', type=str, default='data/raw/tourism_monthly_dataset.tsf',
                        help='Path to the TSF file')
    parser.add_argument('--output', type=str, default='data/processed/tourism_monthly_dataset.csv',
                        help='Path to save the CSV file')
    parser.add_argument('--test-output', type=str, default='data/processed/tourism_monthly_test.csv',
                        help='Path to save the test set CSV file')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    test_output_path = Path(args.test_output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Parsing TSF file: {input_path}")
    series_list, horizon = parse_tsf_file(input_path)
    logger.info(f"Found {len(series_list)} time series with forecast horizon {horizon}")
    
    # Convert to DataFrame
    df = convert_to_df(series_list)
    logger.info(f"Created DataFrame with {len(df)} rows")
    
    # Create train/test split
    train_df = df.copy()
    test_df = pd.DataFrame()
    
    # Group by series_id
    for series_id, group in df.groupby('unique_id'):
        # Sort by date
        group = group.sort_values('ds')
        
        # Split off the last 'horizon' periods for testing
        if len(group) > horizon:
            train_df = train_df[~(train_df['unique_id'] == series_id) | 
                               (train_df['ds'] < group['ds'].iloc[-horizon])]
            
            test_series = group.iloc[-horizon:].copy()
            test_df = pd.concat([test_df, test_series])
    
    # Save to CSV
    train_df.to_csv(output_path, index=False)
    logger.info(f"Saved training data to {output_path}")
    
    test_df.to_csv(test_output_path, index=False)
    logger.info(f"Saved test data to {test_output_path}")

if __name__ == "__main__":
    main() 