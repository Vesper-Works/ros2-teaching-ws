import rclpy

from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2
import numpy as np

from enum import Enum

class State(Enum):
    TurningAround  = 1
    FindingClosestCube = 2
    CloseInOnCube = 3
    ReAlignToCube = 5
    PushingToWall = 6
    BackingUp = 7

class TidyBehaviour(Node):

    min_distance = 0.5  # stay at least 30cm away from obstacles
    turn_speed = 0.3    # rad/s, turning speed in case of obstacle
    forward_speed = 0.4 # m/s, speed with which to go forward if the space is clear
    scan_segment = 1   # degrees, the size of the left and right laser segment to search for obstacles
    state = State.TurningAround
    turn_dir = 0
    startTurnYaw = 0
    robot_current_pose_real = np.array([0, 0, 0])

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

        # subscribe to the camera topic
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.set_pose, 20)
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

#Modified from https://github.com/LCAS/teaching/blob/2324-devel/cmp3103m_ros2_code_fragments/cmp3103m_ros2_code_fragments/colour_chaser2.py
    def camera_callback(self, data):
        #self.get_logger().info("camera_callback")

        if self.state != State.FindingClosestCube and self.state != State.ReAlignToCube:
            return;

        cv2.namedWindow("Image window", 1)

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Convert image to HSV
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        # Create mask for range of colours (HSV low values, HSV high values)
        #current_frame_mask = cv2.inRange(current_frame_hsv,(70, 0, 50), (150, 255, 255))
        current_frame_mask = cv2.inRange(current_frame_hsv,(0, 150, 50), (255, 255, 255)) # orange

        contours, hierarchy = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(contours)
        # Sort by area (keep only the biggest one)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #print(contours)
        
        # Draw contour(s) (image to draw on, contours, contour number -1 to draw all contours, colour, thickness):
        #current_frame_contours = cv2.drawContours(current_frame, contours, -1, (255, 255, 0), 20)
        boxContours = []
        if len(contours) > 0:
            # find the centre of the contour: https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
            for contour in contours:
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    # find the centroid of the contour
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    guy = data.height/2
                    if cy <= guy:
                        continue

                    boxContours.append(contour)
                
                    # Draw a circle centered at centroid coordinates
                    # cv2.circle(image, center_coordinates, radius, color, thickness) -1 px will fill the circle
                    cv2.circle(current_frame, (round(cx), round(cy)), 5, (0, 255, 0), -1)
                                
                    # find height/width of robot camera image from ros2 topic echo /camera/image_raw height: 1080 width: 1920
                    xCentre = data.width / 2
                    # if center of object is to the left of image center move left
                    if cx < xCentre:
                        self.turn_dir = 1
                    # else if center of object is to the right of image center move right
                    elif cx > xCentre:
                        self.turn_dir = -1
                    else: # center of object is in a 100 px range in the center of the image so dont turn
                        #print("object in the center of image")
                        self.turn_dir = 0.0
                    dist = abs(xCentre - cx)
                    self.turn_vel = 0.01#dist/xCentre
                    self.get_logger().info(f'{self.turn_vel}')
                    break;
            # turn until we can see a coloured object
        
        if len(contours) == 0:
            if(self.state == State.FindingClosestCube):
                self.changeState(State.TurningAround)


        current_frame_contours = cv2.drawContours(current_frame, boxContours, -1, (255, 255, 0), 20)   
        # show the cv images
        current_frame_contours_small = cv2.resize(current_frame_contours, (0,0), fx=0.4, fy=0.4) # reduce image size
        cv2.imshow("Image window", current_frame_contours_small)
        cv2.waitKey(1)

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
        min_range_right = self.min_range(data.ranges[:self.scan_segment])
        min_range_left = self.min_range(data.ranges[-self.scan_segment:])
        self.forward_dist = data.ranges[180]
        #self.turn_dir = 0
       
            

    def timer_callback(self):

        twist = Twist()
        if(self.state == State.TurningAround):     
            if(self.startTurnYaw == 0):
                self.changeState(State.TurningAround) 
            if(abs(self.startTurnYaw - self.robot_current_pose_real[2]) > np.pi/2):
                self.changeState(State.FindingClosestCube)
                return
            twist.angular.z = self.turn_speed   
            
        elif(self.state == State.FindingClosestCube):
            _ = 1
        elif(self.state == State.CloseInOnCube):
            _ = 1
        elif(self.state == State.ReAlignToCube):
            _ = 1
        elif(self.state == State.PushingToWall):
            _ = 1
        elif(self.state == State.BackingUp):
            _ = 1

        if(self.turn_dir != 0):  
            twist.angular.z = self.turn_speed * self.turn_dir      
        #else:
            #twist.linear.x = self.forward_speed
        
        self.twist_pub.publish(twist)   
#From https://github.com/LCAS/teaching/blob/2324-devel/cmp3103m_ros2_code_fragments/cmp3103m_ros2_code_fragments/robot_feedback_control_todo.py
    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    #From https://github.com/LCAS/teaching/blob/2324-devel/cmp3103m_ros2_code_fragments/cmp3103m_ros2_code_fragments/robot_feedback_control_todo.py
    def set_pose(self, msg):
        _, _, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_current_pose_real = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])


    def changeState(self, newState):
        self.state = newState
        if(newState == State.TurningAround):
            self.startTurnYaw = self.robot_current_pose_real[2]
            return
        elif(newState == State.FindingClosestCube):
            return
        elif(newState == State.CloseInOnCube):
            return
        elif(newState == State.ReAlignToCube):
            return
        elif(newState == State.PushingToWall):
            return
        elif(newState == State.BackingUp):
            return


def main(args=None):
    print('Starting.')

    try:
        # Initialise the ROS Python subsystem
        rclpy.init()
        # create the Node object we want to run
        node = TidyBehaviour()
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