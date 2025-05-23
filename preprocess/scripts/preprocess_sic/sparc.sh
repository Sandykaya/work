python preprocess/sparc/preprocess_src.py \
    --model "deepseek-chat" \
    --api_key "" \
    --base_url "https://api.deepseek.com" \
    --train_file "data/original_data/sparc/train.json" \
    --dev_file "data/original_data/sparc/dev.json" \
    --preprocessed_train_file "data/preprocessed_data/sparc/preprocessed_train.json" \
    --preprocessed_dev_file "data/preprocessed_data/sparc/preprocessed_dev.json" \
    --comment_cache_train_file "data/preprocessed_data/sparc/comment_cache_train_test.json" \
    --comment_cache_dev_file "data/preprocessed_data/sparc/comment_cache_dev_test.json" \
    --table_path "data/original_data/sparc/tables.json" \
    --db_path "data/original_data/sparc/database" \
    --with_star True \
    --max_retries 10 \
    --random_content_num 10
