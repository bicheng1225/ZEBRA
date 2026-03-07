#!/bin/bash

device="cpu"

datadir="../../data"
dataset="EthereumP"

alpha=0.6498593
beta=0.8328782

topk=20
hops=3

python run.py \
    --datadir $datadir \
    --dataset $dataset \
    --alpha $alpha \
    --beta $beta \
    --num-topk $topk \
    --num-hops $hops \
    --device $device
