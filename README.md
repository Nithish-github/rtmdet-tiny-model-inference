# rtmdet-tiny-model-inference
RTMDet stands as a cutting-edge real-time object detection, boasting remarkable performance surpassing even the renowned YOLO series. With an impressive 52.8% Average Precision (AP) on the challenging COCO dataset, it demonstrates its prowess in accuracy. What's more, it achieves this with blazing speed, clocking over 300 frames per second (FPS) on an NVIDIA 3090 GPU. This combination of speed and precision positions RTMDet as a top contender among object detection models, offering both efficiency and accuracy for real-time applications.

![Models Comparison](https://user-images.githubusercontent.com/17425982/222087414-168175cc-dae6-4c5c-a8e3-3109a152dd19.png)

***Figure 1.*** *RTMDet vs. other real-time object detectors.*

![Model Structure](https://user-images.githubusercontent.com/27466624/204126145-cb4ff4f1-fb16-455e-96b5-17620081023a.jpg)

***Figure 2.*** *RTMDet-l model structure.*

# Installation
To perform inference using the RTMDet model, you'll need to install the required packages: mmcv, mmengine, and mmdet. Common issues such as "No module named ‘mmcv.ops’" or "No module named ‘mmcv._ext’" can arise during installation of these packages.
Follow the installation steps below to ensure proper installation of dependencies:

    # %pip install -U -q openmim
    # !mim install -q "mmengine>=0.6.0"
    # !mim install -q "mmdet>=3.0.0"
    
Install mmcv from the source [Install Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 

# Download config and checkpoint files

    mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

# Run the inference.py file

    python3 inference.py

It will take inference in input.mp4 and output video will be written




     
