# 注意

data 請自行去下載以及解壓縮進 proj4 裡面
記得自己去載 pytorch

## 4-1, 4-2
```
./train_grid.sh
```

應該就結束了, 圖片會全部在 result 裡面, 而 test 的值會在 test_log.jsonl 裡面
REPORT.md 裡面的圖片路徑填好了 跑完就會出來


Loss 使用 CrossEntropyLoss, BCEWithLogitsLoss, FocalLoss
