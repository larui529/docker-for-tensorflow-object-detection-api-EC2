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
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```

if you see "Ran 22 tests in X.XXXs \n OK". That means your environment is ready. 

4
