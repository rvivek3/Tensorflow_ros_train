import rospy
import csv
import os
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import message_filters
from turtlesim.msg import Pose
import math
import time
import threading
from .Wrappers import *
import threading
import concurrent.futures


class DataCollector():

    def __init__(self, ros_model):
        self.name = ros_model.get_name() + "_Data_Collector"
        self.ros_model = ros_model
        self.data_to_collect = ros_model.get_features() + ros_model.get_targets()
        self.startTime = time.time()
        self.collected_data = {}

    def data_synchronizer(self):
        while True:
            print(len(self.collected_data))
            if len(self.collected_data) == len(self.data_to_collect):
                print("Full Data Sample Acquired")
                ordered_data = []
                for i in range(len(self.data_to_collect)):
                    print("add to ordered data")
                    print(self.collected_data[i])
                    ordered_data.append(self.collected_data[i])
                print("ordered data: ")
                ordered_data = tuple(ordered_data)
                print(*ordered_data)
                self.csv_logger(*ordered_data)
                self.collected_data = {}
            else:
                time.sleep(.5)


    def set_collected_data(self, index, message):
        self.collected_data[index] = message



    def csv_logger(self, *args):
        dir_path = str(os.path.dirname(os.path.realpath(__file__)))
        csv_dir = dir_path + "/" + self.ros_model.get_name() + "_data.csv"
        print(os.path.exists(csv_dir))
        print(csv_dir)
        #Create new csv if it doesn't already exist
        if not os.path.exists(csv_dir):
            with open(csv_dir, "w") as fd:
                print("Creating New CSV")
                data_strings = []
                for ros_reading in self.data_to_collect:
                    data_strings += ros_reading.get_name()
                first_row = ["Time"] + data_strings
                print(first_row)
                writer = csv.writer(fd)
                writer.writerow(first_row)

            # Log new data
        with open(csv_dir, "a") as fd:
            new_row = []
            current_time = time.time()
            elapsed = current_time - self.startTime
            new_row.append(elapsed)
            print(elapsed)
            #Extract the data from each message

            for index, ros_reading in enumerate(self.data_to_collect):
                print("extraction")
                print(args[index])
                extracted_data = ros_reading.data_extraction(args[index])
                new_row += extracted_data

            #Write the data to the csv
            writer = csv.writer(fd)
            writer.writerow(new_row)


    def listener(self):
        subscriptions = {}


        for ros_reading in self.data_to_collect:
            print(ros_reading.custom_handler)
            if ros_reading.custom_handler is not None:
                rospy.Subscriber(ros_reading.get_rostopic(),
                                 ros_reading.get_ros_message_type(),
                                 ros_reading.custom_handler)
            else:
                subscriptions[ros_reading.get_description()] = message_filters.Subscriber(ros_reading.get_rostopic(),
                                                                         ros_reading.get_ros_message_type())
        if len(subscriptions) != 0:
            ts = message_filters.ApproximateTimeSynchronizer([subscriptions["turtle_pose"],
                                                              subscriptions["turtle_cmd_vel"]],
                                                             10, 1, allow_headerless=True)
            ts.registerCallback(self.csv_logger)

        rospy.spin()


    def start(self):
        rospy.init_node(self.name, anonymous=False)
        # while (True):
        #     try:
        #         self.listener()
        #     except:
        #         print("Data collection error. Is ROS Master Running?")
        #         print("Trying again in 5 seconds")
        #         time.sleep(5)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(self.data_synchronizer)
        executor.submit(self.listener)

