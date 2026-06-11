"""

Standalone utility to merge per-stock CSV files into a single NIFTY50_all.csv.

The Kaggle NIFTY-50 dataset is distributed as one CSV per ticker symbol.
This script reads every .csv in a given folder, infers the Symbol from the
filename (stripping the .csv extension), adds a 'Symbol' column, concatenates
everything, and writes NIFTY50_all.csv into the same folder.

Usage:
    python combine_data.py <folder_path>

Example:
    python combine_data.py data/
    python combine_data.py "C:/.../Desktop/nifty_50/data"
"""

import os
import argparse
import pandas as pd

def combine_csvs(folder_path: str) -> pd.DataFrame:
    """
    Reads all .csv files in folder_path, infers Symbol from filename,
    appends a Symbol column safely if missing, and returns the concatenated DataFrame.

    Skips files that fail to parse and reports them without crashing.
    Skips any file already named 'NIFTY50_all.csv' or 'stock_metadata.csv'.
    """

    ignored_files = ['NIFTY50_all.csv', 'stock_metadata.csv']
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.csv') and f not in ignored_files
    ]

    if not csv_files:
        raise FileNotFoundError(f"No valid asset .csv files found in: {folder_path}")

    all_dfs = []
    skipped  = []

    for filename in sorted(csv_files):
        symbol   = os.path.splitext(filename)[0].upper()
        filepath = os.path.join(folder_path, filename)

        try:
            temp_df = pd.read_csv(filepath)
            
            # Checks if 'Symbol' already exists before inserting
            if 'Symbol' in temp_df.columns:
                temp_df['Symbol'] = symbol  # Normalize it to match filename
            else:
                temp_df.insert(0, 'Symbol', symbol)   # Insert if missing
                
            all_dfs.append(temp_df)
            print(f"  [+] {filename:<30} → Symbol: {symbol:<15} ({len(temp_df):>6,} rows)")
        except Exception as e:
            skipped.append(filename)
            print(f"  [!] Skipped {filename}: {e}")

    if not all_dfs:
        raise ValueError("All CSV files failed to parse. Check your folder contents.")

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  Merged {len(all_dfs)} file(s)"
          f" → {len(combined):,} total rows"
          f", {combined['Symbol'].nunique()} unique symbols")

    if skipped:
        print(f"  [!] {len(skipped)} file(s) skipped: {skipped}")

    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-stock NIFTY-50 CSV files into a single NIFTY50_all.csv. "
            "Symbol is inferred from each filename (stripped of .csv extension)."
        )
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing per-stock CSV files (e.g.  data/  or  ./raw_stocks)"
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)

    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    print(f"\n[combine_data.py] Scanning: {folder}\n")
    df = combine_csvs(folder)

    output_path = os.path.join(folder, "NIFTY50_all.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")
