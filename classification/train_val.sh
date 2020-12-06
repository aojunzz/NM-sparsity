now=$(date +"%Y%m%d_%H%M%S")
python train_imagenet.py \
--config $1 2>&1|tee train-$now.log




