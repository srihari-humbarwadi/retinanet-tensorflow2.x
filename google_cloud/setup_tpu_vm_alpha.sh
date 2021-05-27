wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cp annotations/instances_val2017.json ../instances_val2017.json
rm -r annotations_trainval2017.zip annotations

pip3 install -r ../requirements.txt
