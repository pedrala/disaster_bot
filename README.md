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


### FLANN Matcher, Homography and Perspective-n-Point algorithm

```python
# 각 참조 이미지에 대해 매칭 및 탐지 수행
for ref in self.reference_data:
    name = ref['name']
    keypoints_ref = ref['keypoints_ref']
    descriptors_ref = ref['descriptors_ref']
    object_points_3d = ref['object_points_3d']
    w_ref, h_ref = ref['image_size']

    # 참조 이미지(descriptors_ref)와 실시간 이미지(descriptors_frame)의 특징 매칭
    matches = self.match_features(descriptors_ref, descriptors_frame)

    if len(matches) > 5:  # 매칭 조건을 10개에서 5개로 낮춤
        self.get_logger().info(f"{name}: {len(matches)}개의 좋은 매칭점을 발견했습니다.")
        matches = matches[:50]

        pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)

        if H is not None:
            corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            image_points = transformed_corners.reshape(-1, 2)

            camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
            dist_coeffs = np.array(self.camera_info.d)
            
            # PnP 알고리즘을 사용하여 위치 및 방향 추정
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points_3d, # 객체의 3D 좌표
                image_points,  # 이미지에서의 변환된 2D 좌표
                camera_matrix,  # 카메라 행렬
                dist_coeffs,  # 왜곡 계수
                flags=cv2.SOLVEPNP_ITERATIVE  # 반복 알고리즘 사용
            )                

            if success:
                self.publish_marker(translation_vector, rotation_vector, name, ref['marker_scale'])
                object_detections[name]['corners'] = transformed_corners
                object_detections[name]['matches'] = matches
                object_detections[name]['detected'] = True
                #self.save_matched_image(frame, name)  # 매칭 성공 시 이미지 저장
            else:
                self.get_logger().warn(f"{name}: PnP 알고리즘 실패")
        else:
            self.get_logger().warn(f"{name}: 호모그래피 계산 실패")
    else:
        self.get_logger().warn(f"{name}: 충분한 매칭점({len(matches)}개)이 없습니다.")
```

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

