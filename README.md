# Disaster Bot Project

## Overview
This program receives images from the Turtlebot4 camera in a ROS2 environment, detects objects (fire extinguishers and people) in real time, estimates the location of the objects, and publishes them as ROS Markers. After extracting feature points and descriptors from the images of fire extinguishers and people using the ORB algorithm, it matches the real-time images of the Turtlebot4 camera with the feature points of the previously extracted images using the FLANN matcher, and if there are more than 5 matching points, it is judged that the object has been recognized and displays the visual results through OpenCV. It estimates the location of the image using the homography and PnP algorithms, and if you add a marker in RVIZ, you can check the location of the image (fire extinguisher, person).

In addition, the parameters of the yaml file were adjusted for auto-mapping without the map saving process after SLAM.


## Capture

### Pre-prepared fire extinguisher and human images for feature extraction 
<p align="center">
  <img src="disaster_bot/images/ext_orig.png" alt="Fire Extinguisher Detection" width="300">
  <img src="disaster_bot/images/man_orig.png" alt="Fallen Man Detection" width="300">
</p>

### Real-time image and feature point matching using the ORB algorithm
<p align="center">
  <img src="capture/orb_recog_firex.png" alt="Fire Extinguisher Detection" width="500">
</p>

### Indoor SLAM track
<p align="center">
  <img src="capture/ttb4_playground.jpeg" alt="ttb4_playground" width="1000">
</p>


## Project Demo
[turtlebot4 automapping video](capture/ttb4_automapping_navigation.mp4)

[ttb4_automapping_in_rviz_video](capture/ttb4_automapping_navigation_rviz_short_480p.mp4)

[image recoognition and displaying location markers in rviz2 video](capture/demo.gif)

How to execute test
===============================
You can test object recognition by placing an image of a fire extinguisher or a person printed on A4 paper in front of your laptop's webcam.

```console
rviz2
ros2 run disaster_bot test
```

How to execute real
===============================
During TurtleBot4 SLAM, the real-time images subscribed to by the TurtleBot4 camera are matched with existing images using the ORB algorithm, and then the location is estimated using the homography-PnP algorithm.

```console
rviz2
ros2 run disaster_bot dibot
```

How to execute launch file
=============================
Run auto-mapping launch file by applying yaml file

```console
ros2 launch disaster_bot auto_mapping_launch.py
```

