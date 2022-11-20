# EPLF-VINS
## Real-Time Monocular Visual-Inertial SLAM With Efficient Point-Line Flow Features

EPLF-VINS is a real-time SLAM framework with efficient point-line flow features. Our work primarily focuses on improving the speed of detection and tracking of line features.

The open-source version of our algorithm is being prepared and will be open-sourced soon.

Authors: Lei Xu, Hesheng Yin, Tong Shi, Jiang Di, Bo Huang from the HIT Industrial Research Institute of Robotics and Intelligent Equipment.


## 1.Prerequisites
1.1 Our testing hardware configuration is a 3.6 GHz Core AMD Ryzen 5-3600 CPU and 16 GB memory desktop PC.

1.2 The algorithms are run on Ubuntu 18.04 with OpenCV 3.4.16 and Ceres solver 1.14.0.

<!--1.3 **Note that** : OpenCV requires library functions for the relevant library functions for line feature extraction (EDLines) such as OpenCV 3.4.16.-->
## 2.Build
``` shell
cd ~/catkin_ws/src
git clone https://
cd ../
catkin_make
source ~/catkin_make/devel/setup.bash
```
## 3.Run on the dataset
We provide guidelines for running on the dataset including [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets), [TUM VI](https://vision.in.tum.de/data/datasets/visual-inertial-dataset), and [KAIST VIO](https://github.com/url-kaist/kaistviodataset).

EuRoC:
``` shell
roslaunch lfvins_estimator euroc.launch
rosbag play your_euroc_path/MH_01.bag
```

TUM VI:
``` shell
roslaunch lfvins_estimator tumvi.launch
rosbag play your_tumvi_path/dataset-magistrale1_512_16.bag
```

KAIST VIO:
``` shell
roslaunch lfvins_estimator kaistvio.launch
rosbag play your_kasitvio_path/circle.bag
```


## 4.Deployed on your device


ROS topics for cameras and IMUs are required to run the entire system. 

Videos: [realRobot_Youtube](https://youtu.be/GCeYeh0P-VE)


![](image/real.png)

The config.yaml file needs to be modified before running which is including necessary parameters such as camera topic name, imu topic name, camera internal parameters, camera-imu extrinsic parameters, and IMU internal parameters.

``` shell
*launch your sensor_ros_package*

*change your robot parameters*
cd ~/catkin_ws/src
gedit ../config/realrobot.yaml

*launch EPLF-VINS*
source devel/setup.bash
roslaunch lfvins_estimator real.launch
```

You can contact me for deployment issues.
## 5.Acknowledgements
Thanks to the open sources of [PL-VINS](https://github.com/cnqiangfu/PL-VINS) and [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono), it is possible to build our algorithm quickly within the VINS system.

### The reference:

VINS-Mono:
``` shell
@ARTICLE{VINS-Mono,
	author={Qin, Tong and Li, Peiliang and Shen, Shaojie},
	journal={IEEE Trans. Robot.}, 
	title={{VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator}}, 
	year={2018},
	volume={34},
	number={4},
	pages={1004-1020},
	doi={10.1109/TRO.2018.2853729}}
```
PL-VINS:
``` shell
@article{PL-VINS,
	author    = {Qiang Fu and Jialong Wang and Hongshan Yu and Islam Ali and Feng Guo and Hong Zhang},
	title     = {{PL-VINS: Real-Time Monocular Visual-Inertial SLAM with Point and Line}},
	journal   = {CoRR},
	volume    = {abs/2009.07462},
	year      = {2020},
	url       = {https://arxiv.org/abs/2009.07462},
	eprinttype = {arXiv},
	eprint    = {2009.07462},
	timestamp = {Wed, 09 Feb 2022 17:07:27 +0100},
	biburl    = {https://dblp.org/rec/journals/corr/abs-2009-07462.bib},
	bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## 6.Licence

The source code is released under GPLv3 license.
We are still working on improving the code reliability. 

For any technical issues, please contact Lei Xu <xulei3shi@163.com>. 

For commercial inquiries, please contact Bo Huang <18606301906@163.com>.
