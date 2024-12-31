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

class FireEx(Node):
    def __init__(self):
        super().__init__('fire_ex')

        self.bridge = CvBridge()
        self.camera_frame_id = 'oakd_rgb_camera_optical_frame'  # 실제 카메라 프레임 ID로 변경

        # Subscribers 생성
        image_sub = Subscriber(self, Image, '/oakd/rgb/preview/image_raw')
        camera_info_sub = Subscriber(self, CameraInfo, '/oakd/rgb/preview/camera_info')

        # 메시지 동기화 설정
        ts = ApproximateTimeSynchronizer([image_sub, camera_info_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.synced_callback)

        # Publisher 및 TF 리스너 설정
        self.marker_pub = self.create_publisher(Marker, 'fire_extinguisher_marker', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ORB 특징점 검출기 생성 및 파라미터 조정
        self.orb = cv2.ORB_create(
            nfeatures=2000,        # 검출할 최대 키포인트 수 증가
            scaleFactor=1.2,       # 피라미드 스케일 팩터
            nlevels=8,             # 피라미드 레벨 수
            edgeThreshold=31,      # 에지 임계값
            firstLevel=0,          # 첫 번째 피라미드 레벨
            WTA_K=2,               # 키포인트 서술자에 사용할 최대 일치 서브 패치 수
            scoreType=cv2.ORB_HARRIS_SCORE,  # 키포인트 검출 방식
            patchSize=31,          # 키포인트 패치 크기
            fastThreshold=20       # FAST 임계값
        )

        # 참조 이미지들의 경로와 실제 크기 설정 (소화기만 포함)
        reference_image_info = [
            {
                'name': 'fire_extinguisher',
                'path': '/home/viator/ws/ttb4_ws/disaster_bot/ext_orig.png',
                'width': 0.18,   # 소화기의 실제 너비 (단위: 미터)
                'height': 0.18   # 소화기의 실제 높이 (단위: 미터)
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
                continue  # 노드를 종료하지 않고 다음 참조 이미지로 계속

            # 특징점 및 디스크립터 검출
            keypoints_ref, descriptors_ref = self.orb.detectAndCompute(reference_image, None)
            self.get_logger().info(f"{ref_name}: 참조 이미지에서 {len(keypoints_ref)}개의 키포인트를 검출했습니다.")

            if descriptors_ref is None or len(keypoints_ref) == 0:
                self.get_logger().error(f"{ref_name} 참조 이미지에서 디스크립터를 추출할 수 없습니다.")
                continue  # 노드를 종료하지 않고 다음 참조 이미지로 계속

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
                'path': ref_path  # 이미지 경로 추가
            })

        if not self.reference_data:
            self.get_logger().error("참조 이미지를 하나도 로드하지 못했습니다. 노드를 종료합니다.")
            rclpy.shutdown()
            return

        self.camera_info = None
        self.get_logger().info("소화기 탐지 노드가 시작되었습니다.")

        # 매칭된 이미지를 저장할 디렉토리 설정
        self.matched_image_dir = "/home/viator/ws/ttb4_ws/matched_images"
        os.makedirs(self.matched_image_dir, exist_ok=True)
        self.match_count = 0  # 매칭 이미지 저장 카운트

        # OpenCV 창 설정 (단일 창으로 통합)
        cv2.namedWindow("Matched Image", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Frame with Keypointse", 640, 370)

    def synced_callback(self, img_msg, cam_info_msg):
        self.camera_info = cam_info_msg
        self.camera_frame_id = cam_info_msg.header.frame_id
        self.image_callback(img_msg)

    def image_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().warn("Camera info is not available yet.")
            return

        try:
            # RGB 이미지 받기
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # 실시간 이미지에서 특징점과 디스크립터 추출
        keypoints_frame, descriptors_frame = self.orb.detectAndCompute(frame_gray, None)
        self.get_logger().debug(f"프레임: {len(keypoints_frame)}개의 키포인트를 검출했습니다.")

        if descriptors_frame is None or len(keypoints_frame) < 10:
            self.get_logger().warn("프레임에 충분한 디스크립터가 없습니다.")
            return

        # 프레임에 검출된 키포인트 표시 (컬러 이미지 사용)
        frame_with_keypoints = cv2.drawKeypoints(
            frame,
            keypoints_frame,
            None,
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )

        # 감지된 객체의 정보를 저장할 리스트
        detected_objects = []

        # 소화기 참조 이미지에 대해 매칭 및 탐지 수행
        for ref in self.reference_data:
            name = ref['name']
            keypoints_ref = ref['keypoints_ref']
            descriptors_ref = ref['descriptors_ref']
            object_points_3d = ref['object_points_3d']
            w_ref, h_ref = ref['image_size']

            self.get_logger().debug(f"참조 이미지 '{name}'에 대해 매칭 시도 중...")

            # 특징점 매칭
            matches = self.match_features(descriptors_ref, descriptors_frame)

            if len(matches) > 10:
                self.get_logger().info(f"{name}: {len(matches)}개의 좋은 매칭점을 발견했습니다.")
                # 상위 매칭점 선택 (최대 50개)
                matches = matches[:50]

                # 매칭된 특징점 좌표 추출
                pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # 호모그래피 계산
                H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)

                if H is not None:
                    # 참조 이미지의 4개의 코너 좌표
                    corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                    # 이미지에서의 객체 영역 좌표
                    transformed_corners = cv2.perspectiveTransform(corners, H)

                    # PnP 알고리즘 사용을 위한 대응점 준비
                    image_points = transformed_corners.reshape(-1, 2)

                    # 카메라 매트릭스 및 왜곡 계수
                    camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
                    dist_coeffs = np.array(self.camera_info.d)

                    # PnP 알고리즘 적용
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        object_points_3d,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        # 객체의 위치를 맵 좌표계로 변환하여 표시
                        self.publish_marker(translation_vector, rotation_vector, name, ref['marker_scale'])
                        detected_objects.append((name, transformed_corners))
                    else:
                        self.get_logger().warn(f"{name}: PnP 알고리즘 실패")
                else:
                    self.get_logger().warn(f"{name}: 호모그래피 계산 실패")

                # 매칭 결과 시각화 및 저장
                self.match_count += 1
                if self.match_count % 30 == 0:  # 매 30프레임마다 저장 및 표시
                    # 참조 이미지 재로드 (컬러)
                    ref_image_color = cv2.imread(ref['path'], cv2.IMREAD_COLOR)
                    if ref_image_color is None:
                        self.get_logger().error(f"{name} 참조 이미지를 다시 로드할 수 없습니다.")
                        continue

                    # 매칭된 이미지 그리기 (컬러 이미지에 매칭 표시)
                    matched_image = cv2.drawMatches(
                        ref_image_color,
                        keypoints_ref,
                        frame,
                        keypoints_frame,
                        matches,
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    save_path = os.path.join(self.matched_image_dir, f"matched_{name}_{self.match_count}.png")
                    cv2.imwrite(save_path, matched_image)
                    self.get_logger().info(f"매칭된 이미지를 저장했습니다: {save_path}")

            else:
                self.get_logger().warn(f"{name}: 충분한 매칭점({len(matches)}개)이 없습니다.")

        # 객체의 경계 상자를 원본 프레임에 그리기
        for obj in detected_objects:
            name, corners = obj
            # 코너 좌표를 int로 변환
            corners = np.int32(corners).reshape(-1, 2)
            # 객체의 경계 상자 그리기
            cv2.polylines(frame, [corners], True, (255, 0, 0), 3, cv2.LINE_AA)
            # 객체 이름 표시
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.putText(frame, name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Frame with Keypoints 표시
        cv2.imshow("Frame with Keypoints", frame)
        cv2.waitKey(1)  # GUI 이벤트 처리

    def match_features(self, descriptors_ref, descriptors_frame):
        # FLANN 기반 매처 설정
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,     # 12
                            key_size=12,        # 20
                            multi_probe_level=1) # 2
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # KNN 매칭 (k=2)
        try:
            matches = flann.knnMatch(descriptors_ref, descriptors_frame, k=2)
        except cv2.error as e:
            self.get_logger().error(f"FLANN 매칭 오류: {e}")
            return []

        # Lowe의 비율 테스트 적용
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:  # 비율 테스트 조정 가능
                    good_matches.append(m)

        self.get_logger().debug(f"좋은 매칭점 수: {len(good_matches)}")
        return good_matches

    def publish_marker(self, translation_vector, rotation_vector, name, marker_scale):
        # 로봇 좌표계에서의 객체 위치
        position = translation_vector.flatten()
        # 회전 벡터를 회전 행렬로 변환
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # 회전 행렬을 쿼터니언으로 변환
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w]

        # 로봇 좌표계에서 맵 좌표계로 변환하여 표시
        try:
            # 현재 시간 기준으로 변환
            now = self.get_clock().now()
            transform = self.tf_buffer.lookup_transform(
                'map', self.camera_frame_id, now, timeout=rclpy.duration.Duration(seconds=1.0))
            p = PointStamped()
            p.header.frame_id = self.camera_frame_id
            p.header.stamp = now.to_msg()
            p.point.x = float(position[0])
            p.point.y = float(position[1])
            p.point.z = float(position[2])

            p_map = do_transform_point(p, transform)

            # Marker 메시지 생성
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name  # 객체 이름을 네임스페이스로 사용
            marker.id = hash(name) % 1000  # 각 객체에 고유한 ID 부여
            marker.type = Marker.CUBE  # 마커 타입을 CUBE로 변경하여 실제 크기 반영
            marker.action = Marker.ADD

            marker.pose.position.x = p_map.point.x
            marker.pose.position.y = p_map.point.y
            marker.pose.position.z = p_map.point.z

            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            # 마커의 크기를 실제 객체의 크기에 맞게 설정
            marker.scale.x = marker_scale['x']
            marker.scale.y = marker_scale['y']
            marker.scale.z = marker_scale['z']

            # 객체에 따라 마커의 색상을 다르게 설정
            if name == 'fire_extinguisher':
                marker.color.a = 1.0  # 투명도
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # 기본 파란색

            marker.lifetime = Duration(sec=10)

            self.marker_pub.publish(marker)

        except TransformException as e:
            self.get_logger().error(f"TF 변환 실패: {e}")

    def __del__(self):
        # OpenCV 창 닫기
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    detector = FireEx()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Keyboard Interrupt (SIGINT)')
    except Exception as e:
        detector.get_logger().error(f"노드 실행 중 예외 발생: {e}")
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()