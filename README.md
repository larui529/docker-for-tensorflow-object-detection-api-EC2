# docker-for-tensorflow-object-detection-api-EC2


1. Use SageMaker instance. (or choose a large memory EC2 instance since the small memory will cause error in installing tensorflow)

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
