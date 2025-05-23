export CUDA_VISIBLE_DEVICES=0
python preprocess/cosql/train_sic.py \
    --batch_size 6 \
    --gradient_descent_step 2 \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --device "0" \
    --alpha 0.75 \
    --epochs 128 \
    --patience 16 \
    --seed 42 \
    --save_path "ckpts/sic/cosql" \
    --train_filepath "data/preprocessed_data/cosql/preprocessed_train.json" \
    --dev_filepath "data/preprocessed_data/cosql/preprocessed_dev.json" \
    --model_name_or_path "/amax/storage/nfs/vpcctrl/d7/huggingface/roberta/roberta-large/" \
    --use_contents true \
    --add_fk_info true \
    --mode "train" \
    --plm_name "roberta-large" \
    --max_input_len 512 \
    --plm_hidden_state_dim 1024 \
    --add_comment \
    --pooling_function "attention"\
    --truncation true \
    --use_comment_enhanced \
    --use_column_enhanced
