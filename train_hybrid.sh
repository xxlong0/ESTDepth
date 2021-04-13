python -m torch.distributed.launch --nproc_per_node=4 train_hybrid.py  --using_apex  --sync_bn \
--datapath /userhome/35/xxlong/dataset/scannet_whole/  \
--testdatapath /userhome/35/xxlong/dataset/scannet_test/ \
--reloadscan False \
--batch_size 1 --seq_len 5 --mode train --summary_freq 10 \
--epochs 7 --lr 0.00004 --lrepochs 2,4,6,8:2 \
--logdir ./logs/hybrid_res50_ndepths64 \
--resnet 50 --ndepths 64 --IF_EST_transformer False \
--depth_min 0.1 --depth_max 10. |  tee -a ./logs/hybrid_res50_ndepths64/log.txt
