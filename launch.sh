for file in configs/*
do
echo launching $file ;
screen -dmS $(basename $file .json) python3 -m retinanet.main --config_path $file --xla --debug ;
done
