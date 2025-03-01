# Deep-learning-projects
📌 Project Overview

This project implements real-time object detection using YOLOv3 on a Raspberry Pi 4, optimized with the OpenVINO Toolkit for faster inference. The system efficiently detects objects in real time while ensuring performance improvements on edge devices like Raspberry Pi.

🚀 Features

YOLOv3-based real-time object detection optimized for Raspberry Pi 4.

OpenVINO Toolkit used to enhance inference speed and reduce computational load.

Face detection integration using Haar Cascade Classifier for multi-object recognition.

Efficient model deployment with OpenCV and optimized DNN frameworks.

🛠️ Requirements

Hardware:

Raspberry Pi 4 (4GB RAM recommended)

Camera Module (Raspberry Pi Camera or USB Webcam)

Software:

Python 3.7+

OpenCV

NumPy

OpenVINO Toolkit

YOLOv3 Weights & Configuration Files (yolov3.weights, yolov3.cfg, coco.names)

🔧 Installation & Setup

Clone the Repository

git clone https://github.com/yourusername/YoloV3-RaspberryPi.git
cd YoloV3-RaspberryPi

Install Dependencies

pip install opencv-python numpy

Install OpenVINO Toolkit for Raspberry Pi:

wget https://github.com/openvinotoolkit/openvino/releases/download/2021.4.2/l_openvino_toolkit_runtime_raspbian_p_2021.4.2.287.tgz
tar -xvzf l_openvino_toolkit_runtime_raspbian_p_2021.4.2.287.tgz
cd l_openvino_toolkit_runtime_raspbian_p_2021.4.2.287
./install.sh

Download YOLOv3 Weights & Configuration

wget https://pjreddie.com/media/files/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names

Run the Object Detection Script

python yolo_v3_raspberrypi.py

📈 Performance Optimization

Using OpenVINO for model optimization allows:

Faster inference speeds on Raspberry Pi 4.

Reduced latency for real-time object detection.

Lower CPU and memory usage, making it efficient for edge devices.

🎯 Future Enhancements

Support for YOLOv4 and YOLOv5 models.

Improve object tracking using Deep SORT algorithm.

Add TensorFlow Lite integration for ultra-lightweight models.

📬 Contribute

Feel free to fork, contribute, or optimize the code! 🚀
