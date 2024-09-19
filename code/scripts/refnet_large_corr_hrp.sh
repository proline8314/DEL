/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/train.py --name "hrp_large_corr_400" --device "cuda:1" \
    --update_loss --enc_with_fp --enc_fp_gated --dec_with_fp --dec_with_dist \
    --target_name "hrp" --target_size 2 --matrix_size 4 \
    --enc_fp_to_gat_feedback "add" --batch_size 2048 --enc_n_layers 5 \
    --loss_sigma_correction --epochs 400 \
    --save_interval 10 \
    --load_path "/data02/gtguo/DEL/data/weights/refnet/hrp_large_corr"
