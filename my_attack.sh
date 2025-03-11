# ./my_attack.sh
language="$1"
CUDA_VISIBLE_DEVICES=8 python my_attack.py \
    --data_file /data2/xiaodanxu/zzz_attack/MOAA/gpt_detection/preprocess/dataset/"$language"/test_subs_merged.jsonl \
    --saved_victim_model_path /data2/xiaodanxu/zzz_attack/CodeGPTSensor/CodeGPTSensor/models_output/"$language"/checkpoint-best-f1/model.bin \
    --model_type codegptsensor \
    --result_store_path ./results/"$language"_attack_results.jsonl \
    --language "$language" \
    --index 0 30000 \
    --seed 99 2>&1 | tee -a log/"$language"_attack.log
