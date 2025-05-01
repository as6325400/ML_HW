#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
對多份 submission.csv 做硬投票 (majority vote) 集成
每份 CSV 應至少有兩欄：file,species
"""

import pandas as pd
from collections import Counter
from pathlib import Path

def ensemble_submissions(csv_list, output_csv="ensemble_submission.csv"):
    # ---------- 1. 讀檔 ----------
    csv_paths = [Path(p) for p in csv_list]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"找不到檔案：{p}")

    dfs = [pd.read_csv(p) for p in csv_paths]

    # ---------- 2. 取共同的檔名 ----------
    common_files = set(dfs[0]["file"])
    for df in dfs[1:]:
        common_files &= set(df["file"])

    if not common_files:
        raise ValueError("不同 CSV 間沒有共同的 file 名稱，請確認輸入")

    # 轉成有序 list，確保之後對齊
    common_files = sorted(common_files)

    # ---------- 3. 建立 merged DataFrame ----------
    merged = pd.DataFrame({"file": common_files})
    for i, df in enumerate(dfs):
        preds = (
            df.set_index("file")
              .loc[common_files, "species"]    # 這裡用 list，不會觸發 set 的 TypeError
              .reset_index(drop=True)
        )
        merged[f"pred_{i}"] = preds

    # ---------- 4. 多數決 ----------
    def majority_vote(row):
        votes = Counter(row)
        # 若同票，按 label 字典序決定（可重現）
        return sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]

    merged["species"] = (
        merged.filter(like="pred_")
              .apply(majority_vote, axis=1)
    )

    # ---------- 5. 輸出 ----------
    final_df = merged[["file", "species"]].sort_values("file")
    final_df.to_csv(output_csv, index=False)
    print(f"✓ Ensemble 完成，已輸出：{output_csv}")
    print(f"  集成檔案數：{len(csv_list)}，共同圖片數：{len(common_files)}")

# =============================
# 把這裡換成你的 submission 路徑
csv_list = [
    "result/final_submission.csv",
    "result1/final_submission.csv",
    "result2/final_submission.csv",
    "result3/final_submission.csv",
    # 想再加就繼續放
]

if __name__ == "__main__":
    ensemble_submissions(csv_list, output_csv="final_ensemble.csv")
