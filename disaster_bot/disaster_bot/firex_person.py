import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from builtin_interfaces.msg import Duration
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from message_filters import Subscriber, ApproximateTimeSynchronizer
import os
import threading
from datetime import datetime  # 이미지 저장을 위한 datetime 모듈 추가

class FireEx(Node):
    def __init__(self):
        super().__init__('fire_ex')

        self.bridge = CvBridge()
        self.camera_frame_id = 'oakd_rgb_camera_optical_frame'

        # Subscribers 생성
        image_sub = Subscriber(self, Image, '/oakd/rgb/preview/image_raw')
        camera_info_sub = Subscriber(self, CameraInfo, '/oakd/rgb/preview/camera_info')

        # 메시지 동기화 설정
        ts = ApproximateTimeSynchronizer([image_sub, camera_info_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.synced_callback)

        # Publisher 및 TF 리스너 설정
        self.marker_pub = self.create_publisher(Marker, 'firex_person_marker', 10)
        
        # TF 버퍼 생성 시 캐시 기간 설정
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ORB 특징점 검출기 생성 및 파라미터 조정
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )

        # 참조 이미지들의 경로와 실제 크기 설정
        reference_image_info = [
            {
                'name': 'fire_extinguisher',
                'path': '/home/viator/ws/ttb4_ws/disaster_bot/ext_orig.png',
                'width': 0.18,
                'height': 0.18
            },
            {
                'name': 'person',
                'path': '/home/viator/ws/ttb4_ws/disaster_bot/man_orig.png',
                'width': 0.23,
                'height': 0.18
            }
        ]

        self.reference_data = []

        for info in reference_image_info:
            ref_name = info['name']
            ref_path = info['path']
            width = info['width']
            height = info['height']

            # 참조 이미지 로드
            reference_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if reference_image is None:
                self.get_logger().error(f"{ref_name} 이미지를 로드할 수 없습니다. 경로를 확인하세요: {ref_path}")
                continue

            # 특징점 및 디스크립터 검출
            keypoints_ref, descriptors_ref = self.orb.detectAndCompute(reference_image, None)
            self.get_logger().info(f"{ref_name}: 참조 이미지에서 {len(keypoints_ref)}개의 키포인트를 검출했습니다.")

            if descriptors_ref is None or len(keypoints_ref) == 0:
                self.get_logger().error(f"{ref_name} 참조 이미지에서 디스크립터를 추출할 수 없습니다.")
                continue

            # 객체의 실제 크기에 따른 3D 좌표 설정
            object_points_3d = np.array([
                [0, 0, 0],
                [width, 0, 0],
                [width, height, 0],
                [0, height, 0]
            ], dtype=np.float32)

            # 참조 이미지의 크기 저장
            h_ref, w_ref = reference_image.shape

            # 모든 데이터를 저장
            self.reference_data.append({
                'name': ref_name,
                'keypoints_ref': keypoints_ref,
                'descriptors_ref': descriptors_ref,
                'object_points_3d': object_points_3d,
                'image_size': (w_ref, h_ref),
                'marker_scale': {
                    'x': width,
                    'y': width,
                    'z': height
                },
                'path': ref_path
            })

        if not self.reference_data:
            self.get_logger().error("참조 이미지를 하나도 로드하지 못했습니다. 노드를 종료합니다.")
            rclpy.shutdown()
            return

        self.camera_info = None
        self.display_frame = None
        self.frame_lock = threading.Lock()
        self.get_logger().info("소화기, 사람 탐지 노드가 시작되었습니다.")

        # 매칭된 이미지를 저장할 디렉토리 설정
        self.matched_image_dir = "/home/viator/ws/ttb4_ws/matched_images"
        os.makedirs(self.matched_image_dir, exist_ok=True)
        self.match_count = 0

        # OpenCV 창 설정
        cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotated Frame", 640, 480)       


    def synced_callback(self, img_msg, cam_info_msg):
        self.camera_info = cam_info_msg
        self.camera_frame_id = cam_info_msg.header.frame_id
        self.image_callback(img_msg)

    def image_callback(self, msg):
        self.last_image_time = rclpy.time.Time.from_msg(msg.header.stamp)
       
        if self.camera_info is None:
            self.get_logger().warn("Camera info is not available yet.")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        # 카메라 매트릭스와 왜곡 계수를 추출
        camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
        dist_coeffs = np.array(self.camera_info.d)

        # 왜곡 보정 적용
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 그레이스케일 변환
        frame_gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

        # 실시간 이미지에서 특징점과 디스크립터 추출
        keypoints_frame, descriptors_frame = self.orb.detectAndCompute(frame_gray, None)
        self.get_logger().debug(f"프레임: {len(keypoints_frame)}개의 키포인트를 검출했습니다.")

        if descriptors_frame is None or len(keypoints_frame) < 10:
            self.get_logger().warn("프레임에 충분한 디스크립터가 없습니다.")
            return

        # 객체별로 탐지 결과를 저장할 변수 초기화
        object_detections = {
            'fire_extinguisher': {
                'corners': None,
                'matches': [],
                'detected': False
            },
            'person': {
                'corners': None,
                'matches': [],
                'detected': False
            }
        }

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

        # 탐지 결과를 시각화
        frame_display = frame.copy()


        # 소화기 탐지 결과 적용
        if object_detections['fire_extinguisher']['detected']:
            corners = object_detections['fire_extinguisher']['corners']
            corners = np.int32(corners).reshape(-1, 2)
            cv2.polylines(frame_display, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.putText(frame_display, 'Fire Extinguisher', (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 사람 탐지 결과 적용
        if object_detections['person']['detected']:
            corners = object_detections['person']['corners']
            corners = np.int32(corners).reshape(-1, 2)
            cv2.polylines(frame_display, [corners], True, (0, 255, 0), 3, cv2.LINE_AA)
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.putText(frame_display, 'Person', (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 표시할 프레임 업데이트
        if frame_display is not None and frame_display.size > 0:
            with self.frame_lock:
                self.display_frame = frame_display.copy()
        else:
            self.get_logger().error("Invalid combined frame. Cannot display.")

    # FLANN (Fast Library for Approximate Nearest Neighbors):

    #     OpenCV에서 제공하는 효율적인 최근접 이웃 검색 알고리즘.
    #     고차원 디스크립터를 비교하는 데 매우 빠른 성능을 발휘.

    # LSH (Locality Sensitive Hashing):

    #     바이너리 디스크립터(ORB, BRIEF 등)에 최적화된 알고리즘.
    #     ORB 디스크립터는 바이너리 형태이므로 FLANN_INDEX_LSH를 사용.
    def match_features(self, descriptors_ref, descriptors_frame):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, #LSH 알고리즘 사용.
                            table_number=6, #해시 테이블 개수 (값을 늘리면 정확도 상승).
                            key_size=12,  #해시 키 크기 (값을 늘리면 더 많은 키를 고려).
                            multi_probe_level=1) #검색 레벨 (값을 늘리면 검색 영역 확장).
        search_params = dict(checks=50) # 검색 시 고려할 후보 노드 수.
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 참조 이미지(descriptors_ref)와 실시간 이미지(descriptors_frame)의 특징 매칭
        try:
            matches = flann.knnMatch(descriptors_ref, descriptors_frame, k=2)
        except cv2.error as e:
            self.get_logger().error(f"FLANN 매칭 오류: {e}")
            return []

        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        self.get_logger().debug(f"좋은 매칭점 수: {len(good_matches)}")
        return good_matches

    def publish_marker(self, translation_vector, rotation_vector, name, marker_scale):
        position = translation_vector.flatten()
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        self.get_logger().debug(f"position : {position}")
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()

        try:
            # 수정된 부분: 이미 Time 객체인 self.last_image_time을 from_msg로 변환하지 않음
            transform = self.tf_buffer.lookup_transform(
                'map',
                self.camera_frame_id,
                self.last_image_time,  # 직접 Time 객체 사용
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            p = PointStamped()
            p.header.frame_id = self.camera_frame_id
            p.header.stamp = self.last_image_time.to_msg()
            p.point.x = float(position[0])
            p.point.y = float(position[1])
            p.point.z = float(position[2])

            p_map = do_transform_point(p, transform)

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name
            marker.id = hash(name) % 1000
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = p_map.point.x
            marker.pose.position.y = p_map.point.y
            marker.pose.position.z = p_map.point.z

            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            marker.scale.x = marker_scale['x']
            marker.scale.y = marker_scale['y']
            marker.scale.z = marker_scale['z']

            if name == 'fire_extinguisher':
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif name == 'person':
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0

            marker.lifetime = Duration(sec=10)

            self.marker_pub.publish(marker)

        except TransformException as e:
            self.get_logger().error(f"TF 변환 실패: {e}")

    def save_matched_image(self, frame, object_name):
        """
        매칭된 이미지를 저장하는 함수
        :param frame: 현재 프레임 이미지
        :param object_name: 탐지된 객체 이름
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = f"{object_name}_{timestamp}.png"
        image_path = os.path.join(self.matched_image_dir, image_name)
        cv2.imwrite(image_path, frame)
        self.get_logger().info(f"매칭된 이미지를 저장했습니다: {image_path}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    detector = FireEx()

    # ROS2 노드를 별도의 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(detector,), daemon=True)
    ros_thread.start()

    try:
        while rclpy.ok():
            # 표시할 프레임이 준비되었는지 확인
            with detector.frame_lock:
                if detector.display_frame is not None:
                    frame_to_show = detector.display_frame.copy()
                    detector.display_frame = None
                else:
                    frame_to_show = None

            if frame_to_show is not None:
                # Annotated Frame 표시
                cv2.imshow("Annotated Frame", frame_to_show)
                key = cv2.waitKey(1)
                if key == 27:  # ESC 키를 누르면 종료
                    detector.get_logger().info("ESC 키가 눌렸습니다. 종료합니다...")
                    break
            else:
                # 프레임이 없으면 잠시 대기
                cv2.waitKey(10)
    except KeyboardInterrupt:
        detector.get_logger().info('Keyboard Interrupt (SIGINT)')
    except Exception as e:
        detector.get_logger().error(f"메인 스레드에서 예외 발생: {e}")
    finally:
        detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()