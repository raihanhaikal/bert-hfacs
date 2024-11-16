#!/bin/bash

python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-2_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-2_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-2_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-3_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-3_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-3_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-4_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-4_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E10_LR1e-4_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-2_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-2_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-2_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-3_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-3_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-3_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-4_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-4_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E20_LR1e-4_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-2_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-2_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-2_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-3_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-3_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-3_BS16
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-4_BS8
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-4_BS12
python ./src/eval.py --model indobert_large --load_model_name indobert_large_E30_LR1e-4_BS16



# chmod +x run_all_indobert_large_eval.sh , di git bash menjadikan file .sh eksekutabel
# jangan lupa terminal sudah di folder src!
# ./run_all_indobert_large_eval.sh
