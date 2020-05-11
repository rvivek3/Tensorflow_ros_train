import rospy
import csv
import os
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import message_filters
import math
import time
from .Wrappers import *

class DataCollector():

    def __init__(self, ros_model):
        self.name = ros_model.getName() + "_Data_Collector"
        self.ros_model = ros_model
        self.data_to_collect = ros_model.features + ros_model.targets
        self.startTime = time.time()

    def csv_logger(self, *arg):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_dir = dir_path + "/" + self.ros_model.getName() + "_data"

        #Create new csv if it doesn't already exist
        if not os.path.exists(csv_dir):
            with open(csv_dir, "w") as fd:
                print("Creating New CSV")
                first_row = ["Time"] + self.data_to_collect
                writer = csv.writer(fd)
                writer.writerow(first_row)

                # Log new data
                with open(csv_dir, "a") as fd:
                    new_row = []
                    current_time = time.time()
                    elapsed = current_time - self.startTime
                    new_row.append(elapsed)

                    #Extract the data from each message

                    for index, ros_reading in enumerate(self.data_to_collect):
                        extracted_data = ros_reading.dataExtraction(arg[index])
                        new_row += extracted_data

                    #Write the data to the csv
                    writer = csv.writer(fd)
                    writer.writerow(new_row)


    def listener(self):
        rospy.Subscriber("/move_base/DWAPlannerROS/global_plan", Path, globalPlanUpdater)
        rospy.Subscriber("/move_base/DWAPlannerROS/local_plan", Path, localPlanUpdater)
        rospy.Subscriber("/move_base/current_goal", PoseStamped, goalUpdater)

        subscriptions = {}
        for ros_reading in self.data_to_collect:
            subscriptions[ros_reading.name] = message_filters.Subscriber(ros_reading.rostopic,
                                                                         ros_reading.ros_message_type)
        ts = message_filters.ApproximateTimeSynchronizer(list(subscriptions.values()), 10, 1, allow_headerless=True);
        ts.registerCallback(self.csv_logger)
        rospy.spin()

    def start(self):
        if __name__ == '__main__':
            rospy.init_node(self.name, anonymous=False)
            while (True):
                try:
                    self.listener()
                except:
                    print("Data collection error. Is ROS Master Running?")
                    print("Trying again in 5 seconds")
                    time.sleep(5)