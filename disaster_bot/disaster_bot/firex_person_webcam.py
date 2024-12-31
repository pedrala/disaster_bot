import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from builtin_interfaces.msg import Duration
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import threading

class FireEx(Node):
    def __init__(self):
        super().__init__('fire_ex_webcam')

        self.bridge = CvBridge()
        self.camera_frame_id = 'camera_frame'

        # 마커 퍼블리셔 설정
        self.marker_pub = self.create_publisher(Marker, 'fire_extinguisher_marker', 10)

        # 카메라 보정 파라미터 수동 설정 (실제 카메라 보정을 통해 얻은 값을 사용하는 것이 좋습니다)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1))  # 왜곡 보정 계수 (없다고 가정)

        # ORB 검출기 설정
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

        # 참조 이미지 정보 설정
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

            # ORB 키포인트 및 디스크립터 추출
            keypoints_ref, descriptors_ref = self.orb.detectAndCompute(reference_image, None)
            self.get_logger().info(f"{ref_name}: 참조 이미지에서 {len(keypoints_ref)}개의 키포인트를 검출했습니다.")

            if descriptors_ref is None or len(keypoints_ref) == 0:
                self.get_logger().error(f"{ref_name} 참조 이미지에서 디스크립터를 추출할 수 없습니다.")
                continue

            # 실제 객체 크기에 따른 3D 좌표 설정
            object_points_3d = np.array([
                [0, 0, 0],
                [width, 0, 0],
                [width, height, 0],
                [0, height, 0]
            ], dtype=np.float32)

            # 참조 이미지 크기 저장
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

        self.display_frame = None
        self.frame_lock = threading.Lock()
        self.get_logger().info("소화기, 사람 탐지 노드가 웹캠을 사용하여 시작되었습니다.")

        # 매칭된 이미지를 저장할 디렉토리 설정 (선택 사항)
        self.matched_image_dir = "/home/viator/ws/ttb4_ws/matched_images"
        os.makedirs(self.matched_image_dir, exist_ok=True)
        self.match_count = 0

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)  # 0은 기본 카메라
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다. 카메라 설정을 확인하세요.")
            rclpy.shutdown()
            return

        # 원하는 프레임 너비와 높이 설정 (필요시)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # OpenCV 창 설정
        cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotated Frame", 640, 460)

        # 프레임을 처리할 타이머 시작 (예: 10Hz)
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("웹캠에서 프레임을 캡처하지 못했습니다.")
            return

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 프레임에서 ORB 키포인트 및 디스크립터 추출
        keypoints_frame, descriptors_frame = self.orb.detectAndCompute(frame_gray, None)
        self.get_logger().debug(f"프레임: {len(keypoints_frame)}개의 키포인트를 검출했습니다.")

        if descriptors_frame is None or len(keypoints_frame) < 10:
            self.get_logger().warn("프레임에 충분한 디스크립터가 없습니다.")
            self.display_detection_result(frame, {}, "디스크립터 부족.")
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

        # FLANN (Fast Library for Approximate Nearest Neighbors):

        #     OpenCV에서 제공하는 효율적인 최근접 이웃 검색 알고리즘.
        #     고차원 디스크립터를 비교하는 데 매우 빠른 성능을 발휘.

        # LSH (Locality Sensitive Hashing):

        #     바이너리 디스크립터(ORB, BRIEF 등)에 최적화된 알고리즘.
        #     ORB 디스크립터는 바이너리 형태이므로 FLANN_INDEX_LSH를 사용.
    
        # FLANN 기반 매처 설정
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, #LSH 알고리즘 사용.
                            table_number=6, #해시 테이블 개수 (값을 늘리면 정확도 상승).
                            key_size=12, #해시 키 크기 (값을 늘리면 더 많은 키를 고려).
                            multi_probe_level=1) #검색 레벨 (값을 늘리면 검색 영역 확장).
        search_params = dict(checks=50)  # 검색 시 고려할 후보 노드 수.
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        for ref in self.reference_data:
            name = ref['name']
            keypoints_ref = ref['keypoints_ref']
            descriptors_ref = ref['descriptors_ref']
            object_points_3d = ref['object_points_3d']
            w_ref, h_ref = ref['image_size']

            # FLANN을 사용하여 특징점 매칭
            # 참조 이미지(descriptors_ref)와 실시간 이미지(descriptors_frame)의 특징 매칭
            try:
                matches = flann.knnMatch(descriptors_ref, descriptors_frame, k=2)
            except cv2.error as e:
                self.get_logger().error(f"FLANN 매칭 오류: {e}")
                continue

            # Lowe 비율 테스트를 적용하면서 매치 개수 확인
            good_matches = []
            for i, match in enumerate(matches):
                if len(match) == 2:  # 매칭점이 2개인지 확인
                    m, n = match
                    if m.distance < 0.75 * n.distance: # 첫 번째 매칭점이 두 번째보다 훨씬 가까운지 확인
                        good_matches.append(m)
                else:
                    self.get_logger().debug(f"매치 {i}에는 충분한 요소가 없습니다: {len(match)}")
                    continue

            self.get_logger().debug(f"{name}: 발견된 좋은 매칭점 수: {len(good_matches)}")

            if len(good_matches) > 10:
                self.get_logger().info(f"{name}: {len(good_matches)}개의 좋은 매칭점을 발견했습니다.")
                good_matches = good_matches[:50]  # 상위 50개 매치로 제한

                # 매칭된 키포인트 추출
                pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # 호모그래피 계산
                H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)

                if H is not None:
                    # 참조 이미지의 코너를 프레임으로 변환
                    corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    image_points = transformed_corners.reshape(-1, 2)

                    # PnP 알고리즘을 사용하여 위치 및 방향 추정
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        object_points_3d, # 객체의 3D 좌표
                        image_points,  # 이미지에서의 변환된 2D 좌표
                        self.camera_matrix,  # 카메라 행렬
                        self.dist_coeffs,  # 왜곡 계수
                        flags=cv2.SOLVEPNP_ITERATIVE  # 반복 알고리즘 사용
                    )
                    
                    self.get_logger().warn(f"{name}: 호모그래피 계산됨")                    

                    if success:
                        # 마커 퍼블리시
                        self.publish_marker(translation_vector, rotation_vector, name, ref['marker_scale'])
                        object_detections[name]['corners'] = transformed_corners
                        object_detections[name]['matches'] = good_matches
                        object_detections[name]['detected'] = True
                    else:
                        self.get_logger().warn(f"{name}: PnP 알고리즘 실패")
                else:
                    self.get_logger().warn(f"{name}: 호모그래피 계산 실패")
            else:
                self.get_logger().warn(f"{name}: 충분한 좋은 매칭점({len(good_matches)}개)이 없습니다.")

        # 탐지 결과 시각화
        self.display_detection_result(frame, object_detections)

    def display_detection_result(self, frame, detections, info_text=None):
        frame_display = frame.copy()

        # 소화기 탐지 결과 적용
        if detections.get('fire_extinguisher', {}).get('detected', False):
            corners = detections['fire_extinguisher']['corners']
            corners = np.int32(corners).reshape(-1, 2)
            cv2.polylines(frame_display, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.putText(frame_display, 'Fire Extinguisher', (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(frame_display, 'Fire Extinguisher Not Detected', (50, 50),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 사람 탐지 결과 적용
        if detections.get('person', {}).get('detected', False):
            corners = detections['person']['corners']
            corners = np.int32(corners).reshape(-1, 2)
            cv2.polylines(frame_display, [corners], True, (0, 255, 0), 3, cv2.LINE_AA)
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.putText(frame_display, 'Person', (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(frame_display, 'Person Not Detected', (50, 100),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 추가 정보 표시 (선택 사항)
        if info_text:
            cv2.putText(frame_display, info_text, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # 이미지 표시
        with self.frame_lock:
            self.display_frame = frame_display.copy()

        cv2.imshow("Annotated Frame", frame_display)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            self.get_logger().info("ESC 키가 눌렸습니다. 종료합니다...")
            self.destroy_node()
            rclpy.shutdown()

    def publish_marker(self, translation_vector, rotation_vector, name, marker_scale):
        position = translation_vector.flatten()
        self.get_logger().info(f"position : {position}")
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()

        # TF 변환을 사용하지 않으므로, 카메라 프레임을 바로 사용
        marker = Marker()
        marker.header.frame_id = "map"  # 실제 설정에 맞게 변경 필요
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = name
        marker.id = hash(name) % 1000
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])

        marker.pose.orientation.x = float(quaternion[0])
        marker.pose.orientation.y = float(quaternion[1])
        marker.pose.orientation.z = float(quaternion[2])
        marker.pose.orientation.w = float(quaternion[3])

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

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    detector = FireEx()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Keyboard Interrupt (SIGINT)')
    except Exception as e:
        detector.get_logger().error(f"main 스레드에서 예외 발생: {e}")
    finally:
        detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()