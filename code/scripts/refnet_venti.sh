/work01/home/gtguo/miniconda3/envs/DEL/bin/python /data02/gtguo/DEL/pkg/train.py --name "ca9_venti" --device "cuda:0" \
    --update_loss --enc_with_fp --enc_fp_gated --dec_with_fp --dec_with_dist \
    --enc_fp_to_gat_feedback "add" --batch_size 1024 \
    --enc_n_layers 5 --enc_node_embedding_size 128 --enc_edge_embedding_size 128 \
    --enc_gat_n_heads 8 --dec_node_input_size 128
