# How to build up a docker image for tensorflow object detection api training on AWS EC2 instance

## Brief summary

This is a tutorial and trail to deploy an training work on AWS EC2 instance. The tutorial include how to set up tensorflow environment and how to download public training dataset and how to export inference graph for serving. 

1. Use SageMaker instance. (or choose a large memory EC2 instance since the small memory will cause error in installing tensorflow)
You can use GPU is the EC2 instance type is 'p2' 
change Dockerfile line 6 from 'install tensorflow' to 'install tensorflow-gpu'

```bash
sudo su
yum update
yum upgrade
mkdir -p test/model
cd test/model
nano Dockerfile 
# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```
# paste Dockerfile content from github to the nano file
# exit the file

2. build docker image from Dockefile located folder 

```bash
docker build --tag tf-od .
```

3. Run the docker image to generate a container
```bash
docker run -it tf-od
```

3. export path and test the installation
```bash
# From tensorflow/models/rsearch
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```

if you see "Ran 22 tests in X.XXXs \n OK". That means your environment is ready. 

4. install wget (could be added to Dockerfile)
```bash
apt-get install wget
```
5. Download pretrained model and public training data

```bash
# From tensorflow/models/rsearch/object_detection
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
git clone https://github.com/larui529/tensowflow-od-aws-ec2.git
cp -r tensowflow-od-aws-ec2/* .
rm -f -R tensowflow-od-aws-ec2
```
6. Install pandas (could be added to Dockerfile)
```bash
pip install pandas
```

7. Generate tfrecord files
```bash
# From tensorflow/models/research/object_detection
chmod -R 777 ~/tensorflow/*
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train/ --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

8. Change path in training/faster_rcnn_inception_v2_pets.config (5 places. search (F6) tensorflow)
delete from /home/ec2-user/ 

9. Run the training code
```bash
# From tensorflow/models/research/object_detection
python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

10. Quit training and export inference graph
```bash
# From tensorflow/models/research/object_detection
# From tensorflow/models/research/object_detection
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph # you have to change "XXXX" to the step number from the checkpoint file.
```
