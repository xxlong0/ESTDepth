python eval_hybrid_seq.py --seq_len 5 --summary_freq 10 --ndepths 64 \
--loadckpt ./checkpoint/model_000006.ckpt \
--datapath /userhome/35/xxlong/dataset/scannet_test \
--evalpath ~/workplace/EST/output/hybrid_EST_V4_ndepths64 \
--testlist ./data/scannet_split/test_split.txt --IF_EST_transformer True \
--depth_min 0.1 --depth_max 10. --save_init_prob False --save_refined_prob False