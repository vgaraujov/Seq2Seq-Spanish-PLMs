# T5S experiments

python simplification/run_simplification.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name GEM/BiSECT \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir t5s_bisect \
    --max_target_length 128 \
    --min_target_length 32 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 3 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end \
    --include_inputs_for_metrics

# BARTO experiments

python simplification/run_simplification.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name GEM/BiSECT \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir barto_bisect \
    --max_source_length 1024 \
    --max_target_length 128 \
    --min_target_length 32 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end \
    --include_inputs_for_metrics \
    --fp16

# BERT2BERT-style experiments

python simplification/run_simplification_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name GEM/BiSECT \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir beto2beto_bisect_untied_2s \
    --max_source_length 512 \
    --max_target_length 128 \
    --min_target_length 32 \
    --tie_encoder_decoder False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --do_predict \
    --predict_with_generate \
    --save_steps 2000 \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --load_best_model_at_end True \
    --fp16

python simplification/run_simplification_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name GEM/BiSECT \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir beto2beto_bisect_tied_3 \
    --max_source_length 512 \
    --max_target_length 128 \
    --min_target_length 32 \
    --tie_encoder_decoder True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --do_predict \
    --predict_with_generate \
    --save_steps 2000 \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --load_best_model_at_end True \
    --fp16