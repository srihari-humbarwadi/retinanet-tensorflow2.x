for file in configs/*
do
screen -dmS $(basename $file .yaml)  python3 -m retinanet.main --config_path $file --xla --debug
done
