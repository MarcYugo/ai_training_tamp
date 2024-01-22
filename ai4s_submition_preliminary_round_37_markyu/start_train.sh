#!/bin/bash
cd code
python train.py --epochs=30 --log=tscnn_v3_step2 --train_root=../data/ #--assume=./model/tscnn_v3_step2_last.pth.tar