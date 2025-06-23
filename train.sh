CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/dist_train.sh 4 13232 \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name example