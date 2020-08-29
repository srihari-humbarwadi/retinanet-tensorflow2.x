mkdir -p $1

#aria2c -j 48 -Z \
#http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
#http://images.cocodataset.org/zips/train2017.zip \
#http://images.cocodataset.org/zips/val2017.zip \
#--dir=$1

#unzip $1/"*".zip -d $1
#mkdir $1/zips && mv $1/*.zip $1/zips

python -m retinanet.dataset_utils.create_coco_tfrecords --download_path=$1

