#!/bin/bash

python ./src/train.py --model indobert_base --epoch 10 --lr 1e-2 --batch_size 8 --save_model_name indobert_base_E10_LR1e-2_BS8
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-2 --batch_size 12 --save_model_name indobert_base_E10_LR1e-2_BS12
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-2 --batch_size 16 --save_model_name indobert_base_E10_LR1e-2_BS16
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-3 --batch_size 8 --save_model_name indobert_base_E10_LR1e-3_BS8
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-3 --batch_size 12 --save_model_name indobert_base_E10_LR1e-3_BS12
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-3 --batch_size 16 --save_model_name indobert_base_E10_LR1e-3_BS16
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-4 --batch_size 8 --save_model_name indobert_base_E10_LR1e-4_BS8
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-4 --batch_size 12 --save_model_name indobert_base_E10_LR1e-4_BS12
python ./src/train.py --model indobert_base --epoch 10 --lr 1e-4 --batch_size 16 --save_model_name indobert_base_E10_LR1e-4_BS16
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-2 --batch_size 8 --save_model_name indobert_base_E20_LR1e-2_BS8
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-2 --batch_size 12 --save_model_name indobert_base_E20_LR1e-2_BS12
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-2 --batch_size 16 --save_model_name indobert_base_E20_LR1e-2_BS16
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-3 --batch_size 8 --save_model_name indobert_base_E20_LR1e-3_BS8
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-3 --batch_size 12 --save_model_name indobert_base_E20_LR1e-3_BS12
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-3 --batch_size 16 --save_model_name indobert_base_E20_LR1e-3_BS16
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-4 --batch_size 8 --save_model_name indobert_base_E20_LR1e-4_BS8
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-4 --batch_size 12 --save_model_name indobert_base_E20_LR1e-4_BS12
python ./src/train.py --model indobert_base --epoch 20 --lr 1e-4 --batch_size 16 --save_model_name indobert_base_E20_LR1e-4_BS16
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-2 --batch_size 8 --save_model_name indobert_base_E30_LR1e-2_BS8
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-2 --batch_size 12 --save_model_name indobert_base_E30_LR1e-2_BS12
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-2 --batch_size 16 --save_model_name indobert_base_E30_LR1e-2_BS16
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-3 --batch_size 8 --save_model_name indobert_base_E30_LR1e-3_BS8
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-3 --batch_size 12 --save_model_name indobert_base_E30_LR1e-3_BS12
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-3 --batch_size 16 --save_model_name indobert_base_E30_LR1e-3_BS16
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-4 --batch_size 8 --save_model_name indobert_base_E30_LR1e-4_BS8
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-4 --batch_size 12 --save_model_name indobert_base_E30_LR1e-4_BS12
python ./src/train.py --model indobert_base --epoch 30 --lr 1e-4 --batch_size 16 --save_model_name indobert_base_E30_LR1e-4_BS16

# chmod +x run_all_indobert_base_train.sh , di git bash menjadikan file .sh eksekutabel
# jangan lupa terminal sudah di folder src!
# ./run_all_indobert_base_train.sh

python train.py --model indobert_large --epoch 10 --lr 1e-5 --batch_size 12 --save_model_name indobert_large_E10_LR1e-5_BS12
