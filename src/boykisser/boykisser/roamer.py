import rclpy

from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class Roamer(Node):
    """
    A simple Roaming ROS2 node. Subscribes to "/scan" and sends velocity commands to "/cmd_vel".
    """

    min_distance = 0.5  # stay at least 30cm away from obstacles
    turn_speed = 2.0    # rad/s, turning speed in case of obstacle
    forward_speed = 0.4 # m/s, speed with which to go forward if the space is clear
    scan_segment = 200   # degrees, the size of the left and right laser segment to search for obstacles

    turn_dir = 0

    def __init__(self):
        """
        Initialiser, setting up subscription to "/scan" and creates publisher for "/cmd_vel"
        """
        super().__init__('roamer')
        self.laser_sub = self.create_subscription(
            LaserScan,"/scan",
            self.callback, 1)
        self.twist_pub = self.create_publisher(
            Twist,
            '/cmd_vel', 1)
        
        self.timer = self.create_timer(0.02, self.timer_callback)

    def min_range(self, range):
        """
        returns the smallest value in the range array.
        """
        # initialise as positive infinity
        min_range = math.inf
        for v in range:
            if v != 0 and v < min_range:
                min_range = v
        return min_range

    def callback(self, data):
        """
        This callback is called for every LaserScan received. 

        If it detects obstacles within the segment of the LaserScan it turns, 
        if the space is clear, it moves forward.
        """

        # first, identify the nearest obstacle in the right 45 degree segment of the laser scanner
        min_range_right = self.min_range(data.ranges[self.scan_segment-10:self.scan_segment])
        min_range_left = self.min_range(data.ranges[-self.scan_segment:])
  

        if min_range_right < self.min_distance:
            self.get_logger().info(f'turning left :3 {min_range_right}')
            self.turn_dir = 1
           
        elif min_range_left < self.min_distance:
            self.get_logger().info(f'turning right {min_range_left}')
            self.turn_dir = -1
        
        else:
            self.get_logger().info('going straight')
            self.turn_dir = 0
       
            

    def timer_callback(self):

        twist = Twist()

        if(self.turn_dir != 0):  
            twist.angular.z = self.turn_speed * self.turn_dir      
        else:
            twist.linear.x = self.forward_speed
        
        self.twist_pub.publish(twist)   
    

def main(args=None):
    print('Starting.')

    try:
        # Initialise the ROS Python subsystem
        rclpy.init()
        # create the Node object we want to run
        node = Roamer()
        # keep the node running until it is stopped (e.g. by pressing Ctrl-C)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        # we use "finally" here, to ensure that everything is correctly tidied up,
        # in case of any exceptions happening.
        node.destroy_node()

if __name__ == '__main__':
    main()