/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/test_refnet_transfer.py --name "ca9_50_init" --base_fname "ca9_large_corr" --device "cuda:1" \
    --batch_size 2048 --enc_n_layers 5 \
    --transfer_learning_ratio 0.625 --epochs 100 \
    --transfer_learning \
