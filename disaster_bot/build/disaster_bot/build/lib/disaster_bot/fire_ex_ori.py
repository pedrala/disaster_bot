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
#import tf_transformations
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
        self.orb = cv2.ORB_create(nfeatures=1000)

        # 참조 이미지 로드 및 특징점 추출
        reference_image1 = cv2.imread('/home/viator/ws/ttb4_ws/disaster_bot/ext_orig.png', cv2.IMREAD_GRAYSCALE)
        if reference_image1 is None:
            self.get_logger().error("소화기 이미지를 로드할 수 없습니다.")
            exit()
            
        reference_image2 = cv2.imread('/home/viator/ws/ttb4_ws/disaster_bot/man_orig.png', cv2.IMREAD_GRAYSCALE)
        if reference_image2 is None:
            self.get_logger().error("구조대상자 이미지를 로드할 수 없습니다.")
            exit()

        self.keypoints_ref, self.descriptors_ref = self.orb.detectAndCompute(reference_image2, None)

        # 소화기의 실제 크기 (미터 단위)
        width = 1.8  # 예시 값, 실제 측정 필요
        height = 1.8  # 예시 값, 실제 측정 필요

        # 소화기 평면의 3D 좌표 (참조 이미지의 코너에 대응)
        self.object_points_3d = np.array([
            [0, 0, 0],            # 좌상단
            [width, 0, 0],        # 우상단
            [width, height, 0],   # 우하단
            [0, height, 0]        # 좌하단
        ], dtype=np.float32)

        self.camera_info = None
        self.get_logger().info("Fire Extinguisher Detector Node has been started.")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.get_logger().info("camera_frame_id:"+self.camera_frame_id)        

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

        # 실시간 이미지에서 특징점 검출 및 매칭
        keypoints_frame, matches = self.match_features(frame_gray)

        if len(matches) > 10:
            # 상위 매칭점 선택
            matches = matches[:50]

            # 매칭된 특징점 좌표 추출
            pts_ref = np.float32([self.keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # 호모그래피 계산
            H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)

            if H is not None:
                # 참조 이미지의 4개의 코너 좌표
                h_ref, w_ref = self.keypoints_ref[0].size, self.keypoints_ref[0].size
                corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                # 이미지에서의 소화기 영역 좌표
                transformed_corners = cv2.perspectiveTransform(corners, H)

                # PnP 알고리즘 사용을 위한 대응점 준비
                image_points = transformed_corners.reshape(-1, 2)

                # 카메라 매트릭스 및 왜곡 계수
                camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
                dist_coeffs = np.array(self.camera_info.d)

                # PnP 알고리즘 적용
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.object_points_3d,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # 소화기의 위치를 맵 좌표계로 변환하여 표시
                    self.publish_marker(translation_vector, rotation_vector)
                else:
                    self.get_logger().warn("PnP 알고리즘 실패")
            else:
                self.get_logger().warn("호모그래피 계산 실패")
        else:
            self.get_logger().warn("충분한 매칭점이 없습니다.")

    def match_features(self, frame_gray):
        # 실시간 이미지에서 특징점과 디스크립터 추출
        keypoints_frame, descriptors_frame = self.orb.detectAndCompute(frame_gray, None)

        if descriptors_frame is None:
            return keypoints_frame, []

        # 매처 생성
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # 특징점 매칭
        matches = bf.match(self.descriptors_ref, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        return keypoints_frame, matches

    def publish_marker(self, translation_vector, rotation_vector):
        # 로봇 좌표계에서의 소화기 위치
        position = translation_vector.flatten()
        # 회전 벡터를 회전 행렬로 변환
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
 
        
        # 회전 행렬을 쿼터니언으로 변환
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w]
        
        # 쿼터니언으로 변환
        # quaternion = tf_transformations.quaternion_from_matrix(
        #     [[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], 0],
        #      [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], 0],
        #      [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], 0],
        #      [0, 0, 0, 1]]
        # )

        # 로봇 좌표계에서 맵 좌표계로 변환
        try:
            # 현재 시간 기준으로 변환
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', self.camera_frame_id, now, rclpy.duration.Duration(seconds=1.0))
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
            marker.ns = "fire_extinguisher"
            marker.id = 0
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = p_map.point.x
            marker.pose.position.y = p_map.point.y
            marker.pose.position.z = p_map.point.z

            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            marker.scale.x = 0.1  # 지름 (m)
            marker.scale.y = 0.1
            marker.scale.z = 0.5  # 높이 (m)

            marker.color.a = 1.0  # 투명도
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

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