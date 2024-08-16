/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/test_refnet_vs.py \
    --name "ca9_vs" --weight_name "ca9_large" --device "cuda:0" \
    --batch_size 1024 --enc_n_layers 5 \
    --load_weight \
    --seed 4 \
    --active_score_threshold 6.0 --inactive_multiplier 10