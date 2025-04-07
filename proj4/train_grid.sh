#!/bin/bash

log_file="test_log.jsonl"

echo "== 調整 epochs: 20 / 40 / 80 =="
for epoch in 20 40 80; do
  bs=32
  lr=0.01
  suffix="e${epoch}_bs${bs}_lr${lr}"
  echo "Running with epochs=$epoch, batch_size=$bs, lr=$lr"
  python train.py --epochs $epoch --bs $bs --lr $lr
  python test.py --weight weight_${suffix}.pth --log $log_file
done

echo "== 調整 batch size: 8 / 16 =="
epoch=20
lr=0.01
for bs in 8 16; do
  suffix="e${epoch}_bs${bs}_lr${lr}"
  echo "Running with epochs=$epoch, batch_size=$bs, lr=$lr"
  python train.py --epochs $epoch --bs $bs --lr $lr
  python test.py --weight weight_${suffix}.pth --log $log_file
done

echo "== 調整 learning rate: 0.1 / 0.001 =="
epoch=20
bs=32
for lr in 0.1 0.001; do
  suffix="e${epoch}_bs${bs}_lr${lr}"
  echo "Running with epochs=$epoch, batch_size=$bs, lr=$lr"
  python train.py --epochs $epoch --bs $bs --lr $lr
  python test.py --weight weight_${suffix}.pth --log $log_file
done


echo "測試不同 loss"
for loss in ce bce focal; do
  suffix="loss_${loss}"
  echo "Running with loss=$loss, batch_size=32, lr=0.01"
  python change_loss.py --loss $loss
  python test.py --weight weight_${suffix}.pth --log $log_file --loss $loss
done
