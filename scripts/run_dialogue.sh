# T5S experiments

python dialogue/run_dialogue.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name miam \
    --dataset_config "dihana" \
    --text_column history \
    --summary_column target \
    --output_dir t5s_dialogue_miam \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

# BARTO experiments

python dialogue/run_dialogue.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name miam \
    --dataset_config "dihana" \
    --text_column history \
    --summary_column target \
    --output_dir barto_dialogue_miam \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

# BERT2BERT-style experiments

python dialogue/run_dialogue.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name miam \
    --dataset_config "dihana" \
    --text_column history \
    --summary_column target \
    --output_dir beto2beto_dialogue_miam_tied \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --do_predict \
    --predict_with_generate \
    --save_steps 2000 \
    --num_train_epochs 6 \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --load_best_model_at_end \
    --tie_encoder_decoder

python dialogue/run_dialogue.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name miam \
    --dataset_config "dihana" \
    --text_column history \
    --summary_column target \
    --output_dir beto2beto_dialogue_miam_untied \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --do_predict \
    --predict_with_generate \
    --save_steps 2000 \
    --num_train_epochs 6 \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --load_best_model_at_end