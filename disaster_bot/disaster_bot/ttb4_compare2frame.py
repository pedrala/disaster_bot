import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraTracker(Node):
    def __init__(self):
        super().__init__('camera_tracker')
        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.previous_frame = None

    def image_callback(self, msg):
        # 로그 추가: 메시지가 수신되었음을 확인
        self.get_logger().info(f"Received an image of size {msg.width}x{msg.height}")

        # CvBridge를 사용하여 이미지 변환
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def process_frame(self, frame):
        # ORB 및 키포인트 매칭
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(frame, None)

        if self.previous_frame is not None:
            # 이전 프레임 처리
            keypoints2, descriptors2 = orb.detectAndCompute(self.previous_frame, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Homography 계산
            if len(matches) >= 4:
                pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                if H is not None:
                    self.get_logger().info(f"Homography matrix:\n{H}")

            # Essential Matrix 계산
            h, w, _ = frame.shape
            K = np.array([
                [1000, 0, w / 2],
                [0, 1000, h / 2],
                [0, 0, 1]
            ])

            if len(matches) >= 4:
                E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    self.get_logger().info(f"Essential matrix:\n{E}")

            # 매칭된 키포인트 시각화
            frame_with_matches = cv2.drawMatches(
                frame, keypoints1, self.previous_frame, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # 결과 영상 보여주기
            cv2.imshow('Matches and Effects', frame_with_matches)
            cv2.waitKey(1)

        # 현재 프레임을 이전 프레임으로 저장
        self.previous_frame = frame


def main(args=None):
    rclpy.init(args=args)
    node = CameraTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
