#!/usr/bin/env python3
import pandas as pd
import os

def main():
    # Paths
    input_path = 'data/iris.csv'             # original file in cwd
    output_dir = 'data'                 # target folder
    output_path = os.path.join(output_dir, 'iris.csv')

    # Load the full dataset
    df = pd.read_csv(input_path)

    # Determine halfway point
    half_index = len(df) // 2

    # Slice the first half
    df_half = df.iloc[:half_index].reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the half-sized dataset
    df_half.to_csv(output_path, index=False)

    print(f"Wrote {half_index} of {len(df)} rows to '{output_path}'.")

if __name__ == '__main__':
    main()
