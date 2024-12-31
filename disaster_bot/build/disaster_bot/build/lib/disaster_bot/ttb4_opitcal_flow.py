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

        # Optical Flow를 위한 초기화
        self.prev_gray = None
        self.prev_points = None
        self.current_frame = None  # 현재 프레임 저장
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # OpenCV GUI를 위한 타이머 생성
        self.timer = self.create_timer(0.1, self.display_frame)  # 10Hz

    def image_callback(self, msg):
        # 로그 추가: 메시지가 수신되었음을 확인
        self.get_logger().info("Received an image")

        # CvBridge를 사용하여 이미지 변환
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = frame  # 최신 프레임 저장
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def process_frame(self, frame):
        # Grayscale로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 이전 프레임이 없는 경우 초기화
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            return frame  # 처리하지 않고 반환

        # Optical Flow 계산
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

        # 유효한 점들만 필터링
        good_prev = self.prev_points[status == 1]
        good_next = next_points[status == 1]

        # Optical Flow 결과 시각화
        for i, (new, old) in enumerate(zip(good_next, good_prev)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # 현재 프레임과 포인트를 다음 계산을 위해 저장
        self.prev_gray = gray
        self.prev_points = good_next.reshape(-1, 1, 2)

        return frame  # 시각화된 프레임 반환

    def display_frame(self):
        if self.current_frame is not None:
            # Optical Flow 처리
            processed_frame = self.process_frame(self.current_frame)

            # 결과 영상 보여주기
            cv2.imshow('Optical Flow', processed_frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
