/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/infer_ef_ca9.py \
    --name "ca9_ef" --weight_name "ca9_large_corr_400" --device "cuda:0" \
    --batch_size 1024 --enc_n_layers 5 \
    --load_weight \
    --seed 4 \
    --active_score_threshold 6.0