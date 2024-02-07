import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from tf_transformations import euler_from_quaternion

class Square(Node):

    distance = 0
    angular = 0
    turning = False

    def __init__(self):

        super().__init__('square');

        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        #self.timer = self.create_timer(0.1, self.run_step)

        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)

    def odom_callback(self, data):

        if(self.turning):
            ang = data.twist.twist.angular

            twist = Twist()
            twist.angular.z = 0.2      
            self.twist_pub.publish(twist) 

            self.angular += ang.z
            self.get_logger().info("{}".format(self.angular))
            if(abs(self.angular) >90):
                self.turning = False
                self.distance = 0
                self.angular = 0
        else:

            twist = Twist()
            twist.linear.x = 0.2      
            self.twist_pub.publish(twist) 

            vel = data.twist.twist.linear
            self.distance+=vel.x
            if(self.distance > 50):
                self.turning = True
                self.distance = 0
                self.angular = 0

        
        
def main(args=None):
    print('Starting.')

    try:
        # Initialise the ROS Python subsystem
        rclpy.init()
        # create the Node object we want to run
        node = Square()
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

