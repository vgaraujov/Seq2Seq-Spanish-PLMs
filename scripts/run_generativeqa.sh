# T5S experiments

python generativeqa/run_generativeqa.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name vgaraujov/SQAC \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir t5s_gen_sqac \
    --max_source_length 480 \
    --max_target_length 32 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

python generativeqa/run_generativeqa.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name mlqa \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir t5s_gen_mlqa \
    --max_source_length 384 \
    --max_target_length 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

# BARTO experiments

python generativeqa/run_generativeqa.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name vgaraujov/SQAC \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir barto_gen_sqac \
    --max_source_length 480 \
    --max_target_length 32 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

python generativeqa/run_generativeqa.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name mlqa \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir barto_gen_mlqa \
    --max_source_length 384 \
    --max_target_length 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

# BERT2BERT-style experiments

python generativeqa/run_generativeqa_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name mlqa \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir beto2beto_gen_mlqa_tied \
    --max_source_length 384 \
    --max_target_length 30 \
    --tie_encoder_decoder True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
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

python generativeqa/run_generativeqa_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name PlanTL-GOB-ES/SQAC \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir beto2beto_gen_sqac_tied \
    --max_source_length 480 \
    --max_target_length 32 \
    --tie_encoder_decoder True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
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

python generativeqa/run_generativeqa_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name mlqa \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir beto2beto_gen_mlqa_untied \
    --max_source_length 384 \
    --max_target_length 30 \
    --tie_encoder_decoder False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
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

python generativeqa/run_generativeqa_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name PlanTL-GOB-ES/SQAC \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir beto2beto_gen_sqac_untied \
    --max_source_length 480 \
    --max_target_length 32 \
    --tie_encoder_decoder False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
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