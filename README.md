# How to use nvidia-docker image for tensorflow object detection api training on AWS EC2 instance

## Brief summary

This is a tutorial and trail to deploy an training work on AWS EC2 instance. The tutorial include how to set up tensorflow environment, download public training dataset and export inference graph for serving. 

## Choose EC2 or SageMaker instance

Since we want to leverage GPU resource in AWS we need to provision an instance with GPU attched. I suggest to use "Deep Learning AMI (Ubuntu) Version 18.0-ami-0010aba6944e97f9b" since the nvidia-docker, CUDA and CUDNN drivers are all pre-installed in this AMI. This is also the AMI that sageMaker instance used. To use GPU resource, you need to select 'p2' for instance type. 

Once provisioned the EC2 instance, you can type the following command lines to set up the Dockerfile.

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
## Dockerfile

Before writting Dockerfile, we need to check the CUDA and cudnn version. Type the following code to find driver version and find corresponding docker image from cuda docker hub. Replace version number in Dockerfile
https://hub.docker.com/r/nvidia/cuda/

```bash
# check cuda version: 
nvcc --version
# Check cudnn version: 
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
# or 
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2

```

Copy and paste the following code into Dockerfile. This script set up the environment and download tensorflow object detection API.

```Dockerfile
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /tensorflow/models
RUN apt-get install -y git python-pip
RUN apt-get install -y protobuf-compiler \
                       python-pil \
                       python-lxml \
                       wget \
                       unzip \
                       nano
RUN pip install --upgrade pip
RUN pip install \
        tensorflow-gpu \
        Cython \
        contextlib2 \ 
        pandas \ 
        matplotlib \ 
        jupyter 

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.
ENV     PYTHONPATH $PYTHONPATH:/opt/tensorflow-models/research:/opt/tensorflow-models/research/slim

```
## Build docker image

Use nvidia-docker command to build a docker image called tf-od. If successfully built the image, you can type `docker images` to see if the docker image with name 'tf-od' is listed.

```bash
nvidia-docker build --tag tf-od .
```

if the building process is failed, you can use the following command to delete image
```bash
docker kill $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)
```

# Start docker container
Run the docker image to generate a container
```bash
nvidia-docker run -it tf-od
```

export path and test the installation. 
```bash
# From tensorflow/models/rsearch
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```
if you see "Ran 22 tests in X.XXXs \n OK". That means your environment is ready. 


Sucess image

![alt text](images/nvidia-image sucess build "Logo Title Text 1")




4. Download pretrained model and public training data

```bash
# From tensorflow/models/rsearch/object_detection
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
git clone https://github.com/larui529/tensowflow-od-aws-ec2.git
cp -r tensowflow-od-aws-ec2/* .
rm -f -R tensowflow-od-aws-ec2
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
