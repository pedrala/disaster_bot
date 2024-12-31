# Disaster Bot Project

## Overview
ROS2 환경에서 Turtlebot4 카메라로부터 이미지를 받아 실시간으로 객체(소화기와 사람)를 탐지하고, 해당 객체의 위치를 추정하여 ROS Marker로 퍼블리시하는 프로그램입니다. ORB 알고리즘을 사용하여 특징점을 추출하고 매칭하며, OpenCV를 통해 시각적인 결과를 표시합니다. 멀티스레딩과 스레드 동기화를 사용하여 안정적인 GUI 처리를 구현하였습니다. RVIZ에서 marker를 추가하면 이미지(소화기, 사람)의 위치를 확인할 수 있습니다.


## Image Example
<img src="disaster_bot/ext_orig.png" alt="Fire Extinguisher Detection" width="300">

<img src="disaster_bot/man_orig.png" alt="Fallen Man Detection" width="300">

How to execute test
===============================
rviz2
ros2 run disaster_bot test


How to execute real
===============================
rviz2
ros2 run disaster_bot dibot

## Project Demo

[Download the demo video](disaster_bot/img_loc_marker_display_in_rviz.mov)
