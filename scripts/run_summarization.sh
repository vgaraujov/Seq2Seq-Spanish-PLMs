# T5S experiments

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name GEM/wiki_lingua \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir t5s_wiki \
    --max_target_length 96 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name mlsum \
    --dataset_config "es" \
    --text_column text \
    --summary_column summary \
    --output_dir t5s_mlsum \
    --max_target_length 48 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name csebuetnlp/xlsum \
    --dataset_config "spanish" \
    --text_column text \
    --summary_column summary \
    --output_dir t5s_xlsum \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/t5-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name dennlinger/eur-lex-sum \
    --dataset_config "spanish" \
    --text_column reference \
    --summary_column summary \
    --output_dir t5s_lexsum \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end

# BARTO experiments

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name GEM/wiki_lingua \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir barto_wiki \
    --max_target_length 96 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end \
    --fp16

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name mlsum \
    --dataset_config "es" \
    --text_column text \
    --summary_column summary \
    --output_dir barto_mlsum \
    --max_target_length 48 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end \
    --fp16

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name csebuetnlp/xlsum \
    --dataset_config "spanish" \
    --text_column text \
    --summary_column summary \
    --output_dir barto_xlsum \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end \
    --fp16

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/bart-base-spanish \
    --do_train \
    --do_eval \
    --dataset_name dennlinger/eur-lex-sum \
    --dataset_config "spanish" \
    --text_column reference \
    --summary_column summary \
    --output_dir barto_lexsum \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 6 \
    --do_predict \
    --predict_with_generate \
    --load_best_model_at_end \
    --fp16

# LEDO experiments

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/led-base-16384-spanish \
    --do_train \
    --do_eval \
    --dataset_name csebuetnlp/xlsum \
    --dataset_config "spanish" \
    --text_column text \
    --summary_column summary \
    --output_dir ledo_xlsum \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 6 \
    --overwrite_output_dir \
    --do_predict \
    --predict_with_generate \
    --save_steps 2000 \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --load_best_model_at_end \
    --fp16

python summarization/run_summarization.py \
    --model_name_or_path vgaraujov/led-base-16384-spanish \
    --do_train \
    --do_eval \
    --dataset_name dennlinger/eur-lex-sum \
    --dataset_config "spanish" \
    --text_column reference \
    --summary_column summary \
    --output_dir ledo_lexsum \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 6 \
    --overwrite_output_dir \
    --do_predict \
    --predict_with_generate \
    --save_steps 2000 \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --load_best_model_at_end \
    --fp16

# BERT2BERT-style experiments

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name GEM/wiki_lingua \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir beto2beto_wiki_tied \
    --max_source_length 512 \
    --max_target_length 96 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name GEM/wiki_lingua \
    --dataset_config "es" \
    --text_column source \
    --summary_column target \
    --output_dir beto2beto_wiki_untied \
    --max_source_length 512 \
    --max_target_length 96 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name mlsum \
    --dataset_config "es" \
    --text_column text \
    --summary_column summary \
    --output_dir beto2beto_mlsum_tied \
    --max_source_length 512 \
    --max_target_length 48 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name mlsum \
    --dataset_config "es" \
    --text_column text \
    --summary_column summary \
    --output_dir beto2beto_mlsum_untied \
    --max_source_length 512 \
    --max_target_length 48 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name dennlinger/eur-lex-sum \
    --dataset_config "spanish" \
    --text_column reference \
    --summary_column summary \
    --output_dir beto2beto_lexsum_tied \
    --max_source_length 512 \
    --max_target_length 512 \
    --tie_encoder_decoder True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name dennlinger/eur-lex-sum \
    --dataset_config "spanish" \
    --text_column reference \
    --summary_column summary \
    --output_dir beto2beto_lexsum_untied \
    --max_source_length 512 \
    --max_target_length 512 \
    --tie_encoder_decoder False \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name csebuetnlp/xlsum \
    --dataset_config "spanish" \
    --text_column text \
    --summary_column summary \
    --output_dir beto2beto_xlsum_tied \
    --max_source_length 512 \
    --max_target_length 64 \
    --tie_encoder_decoder True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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

python summarization/run_summarization_leverage.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
    --do_train \
    --do_eval \
    --dataset_name csebuetnlp/xlsum \
    --dataset_config "spanish" \
    --text_column text \
    --summary_column summary \
    --output_dir beto2beto_xlsum_untied \
    --max_source_length 512 \
    --max_target_length 64 \
    --tie_encoder_decoder False \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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