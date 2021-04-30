export TPU_NAME=$2
echo launching $1
screen -dmS $(basename $1 .json) python3 -m retinanet --is_multi_host --log_dir logs --alsologtostderr --config_path $1
