import rclpy

from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import math
import cv2
import numpy as np

from enum import Enum
import random

class State(Enum):
    TurningAround  = 1
    AlignToCube = 2
    CloseInOnCube = 3
    ReAlignToCube = 4
    PushingToWall = 5
    BackingUp = 6

class TidyBehaviour(Node):

    min_distance = 0.5  # stay at least 30cm away from obstacles
    fast_turn_speed = 0.6    # rad/s, turning speed in case of obstacle
    align_turn_speed = 0.3    # rad/s, turning speed in case of obstacle
    forward_speed = 0.2 # m/s, speed with which to go forward if the space is clear
    scan_segment = 1   # degrees, the size of the left and right laser segment to search for obstacles
    state = State.TurningAround
    turn_dir = 0
    startTurnYaw = 0
    cCentreX = 400
    cCentreY = 400
    robot_current_pose_real = np.array([0, 0, 0])
    rays = None
    depth_image = []
    def __init__(self):
        
        super().__init__('tidybehaviour')
        self.laser_sub = self.create_subscription(LaserScan,"/scan",self.scan_callback, 5)
        self.laser_pub = self.create_publisher(LaserScan, "/scan", 5) # For RVIZ2 debugging
        self.twist_pub = self.create_publisher(Twist,'/cmd_vel', 5)
        
        #self.timer = self.create_timer(0.05, self.timer_callback)

        # Camera subscriptions, changes from simulation to real robot.
        #self.create_subscription(Image, '/camera/depth/image_raw', self.depth_camera_callback, 1)
        #self.create_subscription(Image, '/camera/color/image_raw', self.camera_callback, 1)
        self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', self.depth_camera_callback, 10)
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.set_pose, 2)
        
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
    def contourSorter(self, contour):
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if(M['m00'] == 0): #Revert back to area if invalid moment
            return area
        x = int(M['m10']/(M['m00']))#avoid divide by 0
        y = int(M['m01']/(M['m00']))#avoid divide by 0
        
        dist = abs(self.cCentreX - x)
        boxDepth = -self.depth_image[y][x]

        return boxDepth - (dist/(self.cCentreX/2)) #dist used to settle ties, favour centred boxes
#Modified from https://github.com/LCAS/teaching/blob/2324-devel/cmp3103m_ros2_code_fragments/cmp3103m_ros2_code_fragments/colour_chaser2.py

    def depth_camera_callback(self, data):
        cv2.namedWindow("Depth", 1)

        # Convert ROS Image message to OpenCV image - 16UC1 for real robot (doesn't work very well though)
        #current_frame = self.br.imgmsg_to_cv2(data, "16UC1")
        current_frame = self.br.imgmsg_to_cv2(data, "32FC1")
        
        self.depth_image = current_frame
        depth_image_small = cv2.resize(current_frame, (0,0), fx=0.9, fy=0.9)
        cv2.imshow("Depth", depth_image_small)
        cv2.waitKey(1)
        return
    def camera_callback(self, data):

        twist = Twist()

        #How to move with each state

        if(self.state == State.TurningAround):     
            twist.angular.z = self.fast_turn_speed         
        elif(self.state == State.AlignToCube):
            twist.angular.z = self.align_turn_speed * self.turn_dir   
        elif(self.state == State.CloseInOnCube):
            twist.linear.x = self.forward_speed;
        elif(self.state == State.ReAlignToCube):
            twist.angular.z = self.align_turn_speed * self.turn_dir   
        elif(self.state == State.PushingToWall):
            twist.linear.x = self.forward_speed;
            if(self.rays.ranges[180] < 0.3):
                self.changeState(State.BackingUp)
        elif(self.state == State.BackingUp):
            distance = np.sqrt((self.robot_current_pose_real[0]-self.startReversePosX)**2 
                          + (self.robot_current_pose_real[1]-self.startReversePosY)**2)
            if(distance > 1.0):
                self.changeState(State.TurningAround)
            twist.linear.x = -self.forward_speed; 
        elif(self.state == State.Testing):           
            return
        twist.linear.x -= 0.01 #Back up slightly to avoid infinite loops
        self.twist_pub.publish(twist)   

        #Don't need to run camera when
        if (self.state == State.PushingToWall or  #Pushing box to wall, as it will be out of the camera frame anyway.
            self.state == State.BackingUp): #Backing up, as once it's done it'll scan for boxes.
            return;
    
        if(self.rays == None or len(self.depth_image) == 0): #Don't run if we have no lidar or depth results yet
            return;

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        # Convert image to HSV
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        # Create mask for range of colours (HSV low values, HSV high values)
        current_frame_mask = cv2.inRange(current_frame_hsv,(30, 25, 25), (80, 255, 255))

        contours, hierarchy = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by 
        contours = sorted(contours, key=self.contourSorter, reverse=True) # Sort by depth and how centred it is. Prefer centred boxes to break ties.
     
        boxContours = [] #holds the current target box. List used for ease of use later

        if len(contours) > 0:
            # find the centre of the contour: https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
            for contour in contours:
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    # find the centroid of the contour
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    self.cCentreY = data.height/2
                    self.cCentreX = data.width/2
                    area = cv2.contourArea(contour)
                    if cy <= self.cCentreY or area < 100:
                        continue
                    
                    # Projection matrix for the camera from the camera info. Not included in data so it's hard-coded here just for this assignment.
                    # This next section of code uses the camera's projection matrix to project the pixel position into world space, so I can
                    # calculate which lidar ray to use.
                    # Focal length 448.6...
                    # Principal point (320.5, 240.5)
                    
                    projectionMat = np.matrix([[448.6252424876914, 0, 320.5], [0, 448.6252424876914, 240.5],[0, 0, 1]])
                    inverseProjectionMat = np.linalg.inv(projectionMat)
                    pixelPos = np.array([[cx], [1], [1]]) #Only need x position as it's for lidar
                    pixelWorldDir = inverseProjectionMat * pixelPos # Transforms camera to world space, and the position is inherently a direction.
                    angle = np.arccos(np.dot(np.transpose(pixelWorldDir), [1,0,0])) # 
                    angleDegrees = int(np.degrees(angle))
                    rayIndex = int(len(self.rays.ranges) / 2) + int(angleDegrees - 90) #Get the ray index from angle, knowing that halfway through the rays is directly forward.

                    # To account for situations where the robot has an acute angle to the wall the box is nearest, we sample from a range of rays to find the minimum.                   
                    boxDepth = self.depth_image[cy][cx]
                    centreBoxDepth = boxDepth + 0.1 #adjacent
                    xOffset = centreBoxDepth/10 + 0.2 #opposite
                    c = np.sqrt(centreBoxDepth**2 + xOffset**2) # Use Pythagoras to find the length of the new ray direction.                   
                    angleRange = (np.arctan(xOffset/centreBoxDepth)) #Use SohCahToa to calculate the angle from the original
                    indexRange = int(angleRange * (180 / np.pi))
                    start_index = max(0, rayIndex - indexRange)
                    end_index = min(360, rayIndex + indexRange + 1)

                    for i in range(start_index, end_index): #Used to visualise which rays are using in RVIZ2
                        self.rays.intensities[i] = 100
                    self.laser_pub.publish(self.rays)

                    rayDepth = self.min_range(self.rays.ranges[start_index:end_index])
                    xDistFromCentre = abs(self.cCentreX - cx) / self.cCentreX
                                    
                    if(rayDepth < boxDepth):
                        self.get_logger().warning("Wall is somehow in front of box!")
                    if(rayDepth - boxDepth < 0.2 + xDistFromCentre/8): #xDistFromCentre to account for how cubes appear closer at the edge of the screen.
                        self.get_logger().info(f'Box at {cx}, {cy} against wall')
                        continue
                    self.get_logger().info(f'Box at {cx}, {cy} to be pushed')
                    
                    #This is used to replace the depth sensor for running with the real robot.
                    #The depth sensor on the real Limo robot is unusable, so the following 
                    #mathematical approach is used instead.
                        #By noting down the values of the contour's area and distance of the robot to the wall
                        #when the box is next to the wall and plotting these points on Desmos I found the equation
                        #y=1200*x^-2 fairly accurately describes the relationship between contour area and wall
                        #distance to the robot. This means that I can calculate the expected area for the contours
                        #if the cube were next to the wall. If the result is greater than expected, then the cube
                        #must be away from the wall and therefore must be pushed. This fails when the robot is 
                        #at increasingly acute angles to the wall however, but a failure means it will push the cube
                        #when it doesn't have to, rather than not pushing a cube it must, meaning the task will still
                        #be completed.
                    #expectedArea = 1200 * (self.rays.ranges[rayIndex]** -2)
                    #if(area < expectedArea + 200):
                    #    continue
                    

                    boxContours.append(contour)

                    # Draw a circle centered at centroid coordinates
                    # cv2.circle(image, center_coordinates, radius, color, thickness) -1 px will fill the circle
                    cv2.circle(current_frame, (round(cx), round(cy)), 5, (0, 255, 0), -1)
                                
                    # if center of object is to the left of image centre move left
                    if cx < self.cCentreX - 10:
                        self.turn_dir = 1
                    # else if center of object is to the right of image centre move right
                    elif cx > self.cCentreX + 10:
                        self.turn_dir = -1
                    else: #else if it's in the centre                      
                        self.changeState(State.PushingToWall if self.state == State.ReAlignToCube else State.CloseInOnCube)
                        self.turn_dir = 0.0
                    
                    if(self.state == State.CloseInOnCube): ##If we're closing in on it and it's close enough, realign ourselves to it
                        if(area> 3000):
                            self.changeState(State.ReAlignToCube)
                    break;
        
        if len(boxContours) == 0: 
            if(self.state != State.PushingToWall): # If we lose the cube, find another other
                self.changeState(State.TurningAround)
            return;
        elif (self.state == State.TurningAround): # We've found a cube, lets align to it
            self.changeState(State.AlignToCube)
       
        if(self.state == State.CloseInOnCube): #Logging
            self.get_logger().info(f'Distance: {self.rays.ranges[180]} / Area: {cv2.contourArea(boxContours[0])}')

        current_frame_contours = cv2.drawContours(current_frame, boxContours, -1, (255, 255, 0), 20) #Draw the contour of our box
        current_frame_contours_small = cv2.resize(current_frame_contours, (0,0), fx=0.9, fy=0.9) # reduce image size for display
        cv2.imshow("Colour", current_frame_contours_small)
        cv2.waitKey(1)

    # from https://github.com/LCAS/teaching/blob/2324-devel/cmp3103m_ros2_code_fragments/cmp3103m_ros2_code_fragments/roamer.py
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

    def scan_callback(self, data):
        self.rays = data
               
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

    # Allows code to be run when state is changed, overkill but I thought I would need it more
    def changeState(self, newState):
        self.state = newState
        if(newState == State.TurningAround):
            return
        elif(newState == State.AlignToCube):
            return
        elif(newState == State.CloseInOnCube):
            return
        elif(newState == State.ReAlignToCube):
            return
        elif(newState == State.PushingToWall):
            return
        elif(newState == State.BackingUp):
            self.startReversePosX = self.robot_current_pose_real[0]
            self.startReversePosY = self.robot_current_pose_real[1]
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