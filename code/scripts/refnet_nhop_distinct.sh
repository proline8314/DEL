/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/train_nhop_distinct.py --name "hrp_large_nhop_distinct" \
    --device "cuda:0" \
    --seed 4 \
    --target_name "hrp" --target_size 2 --matrix_size 4 \
    --update_loss --enc_with_fp --enc_fp_gated --dec_with_fp --dec_with_dist \
    --enc_fp_to_gat_feedback "add" --batch_size 2048 --enc_n_layers 5 \
    --loss_sigma_correction --lr_schedule