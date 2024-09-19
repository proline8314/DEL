/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/train.py --name "ca9_large_corr_400" --device "cuda:2" \
    --update_loss --enc_with_fp --enc_fp_gated --dec_with_fp --dec_with_dist \
    --enc_fp_to_gat_feedback "add" --batch_size 4096 --enc_n_layers 5 \
    --loss_sigma_correction --epochs 200 \
    --load_path "/data02/gtguo/DEL/data/weights/refnet/ca9_large_corr"
