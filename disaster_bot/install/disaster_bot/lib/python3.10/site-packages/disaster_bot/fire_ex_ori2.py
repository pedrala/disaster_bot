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

class FireEx(Node):
    def __init__(self):
        super().__init__('fire_ex')

        self.bridge = CvBridge()
        self.camera_frame_id = 'oakd_rgb_camera_optical_frame'  # 실제 카메라 프레임 ID로 변경
        self.image_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/oakd/rgb/preview/camera_info', self.camera_info_callback, 10)
        self.marker_pub = self.create_publisher(Marker, 'fire_extinguisher_marker', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ORB 특징점 검출기 생성
        #self.orb = cv2.ORB_create(nfeatures=1000)
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
                'width': 0.18,   # 소화기의 실제 너비 (단위: 미터)
                'height': 0.18   # 소화기의 실제 높이 (단위: 미터)
            },
            {
                'name': 'person',
                'path': '/home/viator/ws/ttb4_ws/disaster_bot/man_orig.png',
                'width': 0.23,   # 사람의 실제 너비 (단위: 미터)
                'height': 0.18   # 사람의 실제 높이 (단위: 미터)
            }
        ]

        self.reference_data = []

        for info in reference_image_info:
            ref_name = info['name']
            ref_path = info['path']
            width = info['width']
            height = info['height']

            reference_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if reference_image is None:
                self.get_logger().error(f"{ref_name} 이미지를 로드할 수 없습니다.")
                exit()

            keypoints_ref, descriptors_ref = self.orb.detectAndCompute(reference_image, None)

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
                }
            })

        self.camera_info = None
        self.get_logger().info("Fire Extinguisher Detector Node has been started.")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.get_logger().info("camera_frame_id: " + self.camera_frame_id)

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
      
        if descriptors_frame is None:
            self.get_logger().warn("No descriptors in frame.")
            return

        # 각 참조 이미지에 대해 매칭 및 탐지 수행
        for ref in self.reference_data:
            name = ref['name']
            keypoints_ref = ref['keypoints_ref']
            descriptors_ref = ref['descriptors_ref']
            object_points_3d = ref['object_points_3d']
            w_ref, h_ref = ref['image_size']
            
            self.get_logger().info(f"{name}: 참조 이미지에서 {len(keypoints_ref)}개의 키포인트를 검출했습니다.")

            # 특징점 매칭
            matches = self.match_features(descriptors_ref, descriptors_frame)

            if len(matches) > 10:
                # 상위 매칭점 선택
                matches = matches[:50]

                # 매칭된 특징점 좌표 추출
                pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # 호모그래피 계산
                H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    inliers = mask.sum()
                    self.get_logger().info(f"호모그래피 인라이어 수: {inliers}")
                    
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
                    else:
                        self.get_logger().warn(f"{name}: PnP 알고리즘 실패")
                else:
                    self.get_logger().warn(f"{name}: 호모그래피 계산 실패")
            else:
                self.get_logger().warn(f"{name}: 충분한 매칭점이 없습니다.")

    def match_features(self, descriptors_ref, descriptors_frame):
        # 매처 생성
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # 특징점 매칭
        matches = bf.match(descriptors_ref, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def publish_marker(self, translation_vector, rotation_vector, name, marker_scale):
        # 로봇 좌표계에서의 객체 위치
        position = translation_vector.flatten()
        # 회전 벡터를 회전 행렬로 변환
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # 회전 행렬을 쿼터니언으로 변환
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w]

        # 로봇 좌표계에서 맵 좌표계로 변환
        try:
            # 현재 시간 기준으로 변환
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map', self.camera_frame_id, now, rclpy.duration.Duration(seconds=1.0))
            p = PointStamped()
            p.header.frame_id = self.camera_frame_id
            p.header.stamp = now.to_msg()
            p.point.x = float(position[0])
            p.point.y = float(position[1])
            p.point.z = float(position[2])

            p_map = do_transform_point(p, trans)

            # Marker 메시지 생성
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name  # 객체 이름을 네임스페이스로 사용
            marker.id = 0
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

        except Exception as e:
            self.get_logger().error(f"TF 변환 실패: {e}")

def main(args=None):
    rclpy.init(args=args)
    detector = FireEx()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()