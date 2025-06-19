#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# ========= configuration =========
base_dir   =  "your/base/directory" 
clean_dir  = "your/base/directory/Cleaned Data"  
aug_dir    =  "your/base/directory/SGP_Augmented" 
out_dir    = "your/base/directory/Final_Features"
out_dir.mkdir(parents=True, exist_ok=True)

lag_period     = 12          # make sure this matches your data
n_factors      = 30         # make sure this matches your data
factor_cols = [f"SGP_F{i+1:02d}" for i in range(n_factors)]
merge_keys     = ["key1","key2"]       # add any additional keys if needed
date_fmt       = "%Y-%m-%d"   # date format in your CSV files

def main():
    aug_files = list(aug_dir.glob("*.csv"))
    if not aug_files:
        print(" No augmented files found in the specified directory.")
        return

    for aug_path in aug_files:
        raw_path = clean_dir / aug_path.name
        if not raw_path.exists():
            print(f" Skip {aug_path.name} —— Cleaned Data not found.")
            continue

        df_raw = pd.read_csv(raw_path)
        df_aug = pd.read_csv(aug_path)

        # date formatting 
        for df in (df_raw, df_aug):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime(date_fmt)

        # merging
        try:
            df_merge = (
                df_raw.merge(
                    df_aug[merge_keys + factor_cols], on=merge_keys, how="left"
                )
                .sort_values(merge_keys)
                .reset_index(drop=True)
            )
        except KeyError as e:
            print(f"× Merged failure {aug_path.name} —— missing column: {e}")
            continue

        # optional: if you want to keep the original data and the augmented features separately
        # df_merge = pd.concat(
        #     [df_raw.iloc[lag_period:].reset_index(drop=True), df_aug[factor_cols]],
        #     axis=1,
        # )

        # save the merged DataFrame
        out_file = out_dir / aug_path.name
        df_merge.to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f" Output: {out_file.name}")

    print(" All files out: ", out_dir)

if __name__ == "__main__":
    main()
