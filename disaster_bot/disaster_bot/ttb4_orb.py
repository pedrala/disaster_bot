import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraTracker(Node):
    def __init__(self):
        super().__init__('camera_tracker')
        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # Create a timer to periodically run a callback
        self.timer = self.create_timer(0.1, self.timer_callback)  # 0.1 seconds interval

        self.latest_frame = None

    def image_callback(self, msg):
        # 로그 추가: 메시지가 수신되었음을 확인
        self.get_logger().info(f"Received an image of size {msg.width}x{msg.height}")

        # CvBridge를 사용하여 이미지 변환
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_frame = frame  # Store the latest frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def process_frame(self, frame):
        # ORB 및 Visual Tracker 적용
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

        # 결과 영상 보여주기
        cv2.imshow('ORB Tracker', frame_with_keypoints)
        cv2.waitKey(1)

    def timer_callback(self):
        if self.latest_frame is not None:
            self.process_frame(self.latest_frame)
        else:
            self.get_logger().info("No frame to process")

def main(args=None):
    rclpy.init(args=args)
    node = CameraTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
