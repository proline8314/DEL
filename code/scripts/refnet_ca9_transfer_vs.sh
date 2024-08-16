/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/test_refnet_vs_transfer.py \
    --name "ca9_25_vs" --weight_name "ca9_large_corr" --device "cuda:2" \
    --batch_size 2048 --enc_n_layers 5 \
    --train \
    --epochs 0 \
    --transfer_learning --transfer_learning_ratio 0.25 \
    --active_score_threshold 6.0 --inactive_multiplier 10