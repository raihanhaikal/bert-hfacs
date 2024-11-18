#!/bin/bash

python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-2_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-2_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-2_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-3_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-3_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-3_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-4_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-4_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E10_LR1e-4_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-2_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-2_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-2_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-3_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-3_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-3_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-4_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-4_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E20_LR1e-4_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-2_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-2_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-2_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-3_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-3_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-3_BS16
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-4_BS8
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-4_BS12
python ./src/eval.py --model indobert_base --load_model_name indobert_base_E30_LR1e-4_BS16


# chmod +x run_all_indobert_base_eval.sh , di git bash menjadikan file .sh eksekutabel
# jangan lupa terminal sudah di folder src!
# ./run_all_indobert_base_eval.sh

python eval.py --model indobert_large --load_model_name indobert_large_no_train
python eval.py --model indobert_base --load_model_name indobert_base_no_train
python eval.py --model indobert_large --load_model_name indobert_base_E10_LR1e-5_BS12