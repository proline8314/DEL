/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/test_refnet_affinity.py \
    --name "ca9_large_corr" --device "cuda:1" --target_name "ca9" \
    --enc_with_fp --enc_fp_gated --dec_with_fp --dec_with_dist \
    --enc_fp_to_gat_feedback "add" --batch_size 1024 --enc_n_layers 5
