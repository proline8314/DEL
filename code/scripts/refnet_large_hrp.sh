/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/train.py --name "hrp_large" --device "cuda:0" \
    --update_loss --enc_with_fp --enc_fp_gated --dec_with_fp --dec_with_dist --enc_fp_to_gat_feedback "add" \
    --batch_size 4096 --enc_n_layers 5 --warmup_ratio 0.05 --epochs 100 \
    --target_name "hrp" --target_size 2 --matrix_size 4 \
    --forced_reload
