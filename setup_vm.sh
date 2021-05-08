sudo fallocate -l 15G /swapfile;
sudo chmod 600 /swapfile;
sudo mkswap /swapfile;
sudo swapon /swapfile
sudo apt install htop tree -y

pip install easydict pycocotools

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cp annotations/instances_val2017.json .
