import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np

class AutoMappingExplorer(Node):
    def __init__(self):
        super().__init__('auto_mapping_explorer')
        # /map 토픽 구독을 설정하여 맵 데이터를 가져옴
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        # /goal_pose 토픽에 목표 좌표를 퍼블리시할 퍼블리셔 생성
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )
        self.map_data = None  # 맵 데이터를 저장할 변수
        self.map_metadata = None  # 맵 메타데이터(크기, 원점, 해상도 등)를 저장
        self.exploration_complete = False  # 탐사 완료 여부 플래그

        # 타이머 설정: 1초마다 목표를 찾는 함수 호출
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1초마다 호출

    def map_callback(self, msg):
        """맵 데이터가 갱신될 때 호출되는 콜백 함수."""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))  # 맵 데이터를 2D 배열로 변환
        self.map_metadata = msg.info
        self.get_logger().info("맵 데이터가 갱신되었습니다.")
        
        # 맵 다운 샘플링 (예: 2배 다운 샘플링)
        self.downsample_map()

    def downsample_map(self, factor=2):
        """맵을 다운샘플링하는 함수."""
        if self.map_data is None:
            return

        # 맵의 크기를 다운샘플링 factor에 맞게 변경
        new_shape = (self.map_data.shape[0] // factor, self.map_data.shape[1] // factor)
        downsampled_map = self.map_data[:new_shape[0]*factor, :new_shape[1]*factor].reshape(
            (new_shape[0], factor, new_shape[1], factor)
        ).mean(axis=(1, 3))  # 평균을 내서 다운샘플링

        self.map_data = downsampled_map
        self.get_logger().info(f"맵 다운샘플링 완료: 새로운 크기 {self.map_data.shape}")

    def timer_callback(self):
        """주기적으로 호출되어 목표를 찾고 퍼블리시하는 함수."""
        if self.map_data is None or self.map_metadata is None:
            self.get_logger().warning("맵 데이터 또는 메타데이터가 아직 없습니다.")
            return

        if not self.exploration_complete:
            self.find_next_goal()  # 새로운 목표 지점 탐색

    def find_next_goal(self):
        """다음 탐사 목표 지점을 찾는 함수."""
        # 미탐사 영역(-1)의 인덱스를 찾음
        unexplored_indices = np.argwhere(self.map_data == -1)
        if unexplored_indices.size == 0:
            self.get_logger().info("탐사가 완료되었습니다. 더 이상 미탐사 영역이 없습니다.")
            self.exploration_complete = True
            return

        # 미탐사 셀의 월드 좌표를 계산
        goals = []
        origin = np.array([self.map_metadata.origin.position.x, self.map_metadata.origin.position.y])  # 맵 원점
        resolution = self.map_metadata.resolution * 2  # 다운샘플링 후 해상도도 조정
        
        for idx in unexplored_indices:
            world_x = idx[1] * resolution + origin[0] + resolution / 2  # 셀의 x 좌표를 월드 좌표로 변환
            world_y = idx[0] * resolution + origin[1] + resolution / 2  # 셀의 y 좌표를 월드 좌표로 변환
            
            # 해당 목표 지점이 탐사 가능(인접한 셀이 자유 공간인지 확인)
            neighbors = self.get_neighbors(idx[0], idx[1])

            # 미탐사 영역이 벽(100)이나 벽에 인접한 경우 제외
            if any(self.map_data[n[0], n[1]] == 100 for n in neighbors):  # 벽과 인접한 경우 제외
                continue

            if any(self.map_data[n[0], n[1]] == 0 for n in neighbors):  # 자유 공간(0)이 근처에 있는지 확인
                goals.append((world_x, world_y))
        
        if not goals:
            self.get_logger().info("도달 가능한 미탐사 목표가 없습니다.")
            self.exploration_complete = True
            return

        # 로봇 위치(맵 원점 기준)에서 가장 먼 목표 지점 선택
        farthest_goal = max(goals, key=lambda g: np.linalg.norm(np.array(g)))
        self.set_goal(farthest_goal)


    def get_neighbors(self, row, col):
        """맵 그리드에서 해당 셀의 4방향(상하좌우) 이웃 좌표를 가져옴."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 위, 아래, 왼쪽, 오른쪽
            r, c = row + dr, col + dc
            if 0 <= r < self.map_data.shape[0] and 0 <= c < self.map_data.shape[1]:  # 맵 범위 내에 있는지 확인
                neighbors.append((r, c))
        return neighbors

    def set_goal(self, goal):
        """목표 위치를 PoseStamped 메시지로 퍼블리시."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()  # 현재 시간
        pose.header.frame_id = 'map'  # 목표의 기준 프레임: 맵 프레임
        pose.pose.position.x = goal[0]
        pose.pose.position.y = goal[1]
        pose.pose.orientation.w = 1.0  # 특정 방향은 지정하지 않음

        self.goal_publisher.publish(pose)  # 목표 퍼블리시
        self.get_logger().info(f"새로운 목표 설정: x={goal[0]:.2f}, y={goal[1]:.2f}")

def main(args=None):
    rclpy.init(args=args)
    explorer = AutoMappingExplorer()

    try:
        rclpy.spin(explorer)  # 노드 실행
    except KeyboardInterrupt:
        explorer.get_logger().info("탐사가 사용자에 의해 중단되었습니다.")
    finally:
        explorer.destroy_node()  # 노드 종료
        rclpy.shutdown()

if __name__ == '__main__':
    main()
