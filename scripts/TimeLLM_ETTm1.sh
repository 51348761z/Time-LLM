model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

master_port=2233
num_process=1
batch_size=8
d_model=32
d_ff=128
llm_model='GPT2'
llm_dim=768
data_pretrain='ETTm1'

comment='TimeLLM-ETTm1'

# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  # --data_pretrain $data_pretrain

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model 'TimeLLM' \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 8 \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_192 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 192 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --lradj 'TST'\
#   --learning_rate 0.001 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --patience 20 \
#   --model_comment $comment
#
# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_336 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 336 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --lradj 'TST'\
#   --learning_rate 0.001 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --patience 20 \
#   --model_comment $comment
#
# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_512_720 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 720 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --lradj 'TST'\
#   --learning_rate 0.001 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --patience 20 \
#   --model_comment $comment
