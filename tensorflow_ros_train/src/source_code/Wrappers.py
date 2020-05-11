import subprocess
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from keras import optimizers
from keras import Model
from keras.layers.merge import Concatenate
from keras import regularizers
import keras
from keras.models import load_model
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import turtlesim
from nav_msgs.msg import Odometry, Path
from .DataCollector import *
from .task_manager import *
import random
import rospy

class ROSModel():
    def __init__(self, name, baseline_model, ros_features = None, ros_targets = None):
        self.name = name
        self.model = baseline_model()
        self.baseline_model = baseline_model
        self.features = ros_features
        self.targets = ros_targets

        if self.features is not None and self.targets is not None:
            self.data_collector = DataCollector(self)
        else:
            self.data_collector = None

        print(name + " Model Created")




    def initialize_data_collector(self):
        self.data_collector = DataCollector(self)

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        print("Model name changed to " + self.name)

    def get_features(self):
        return self.features;

    def set_features(self, ros_features):
        self.features = ros_features
        print("Features of " + self.name + " set")

    def get_targets(self):
        return self.targets;

    def set_targets(self, ros_targets):
        self.targets = ros_targets
        print("Targets of " + self.name + " set")

    def get_baseline_model(self):
        return self.baseline_model

    def set_baseline_model(self, baseline_model):
        self.baseline_model = baseline_model
        print("Baseline Model of " + self.name + " set")

    def summary(self):
        print("Summary of " + self.name)
        return self.model.summary()

    def fit(self,
            training_data,
            custom_feature_processing=None,
            custom_target_processing=None,
            batch_size= None,
            epochs = 1,
            verbose = 1,
            callbacks = None,
            validation_split = 0.0,
            validation_data = None,
            shuffle = True,
            class_weight = None,
            sample_weight = None,
            initial_epoch = 0,
            steps_per_epoch = None,
            validation_steps = None,
            validation_batch_size = None,
            validation_freq = 1,
            max_queue_size = 10,
            workers = 1,
            use_multiprocessing = False):

        # Get training data from csv
        training_df = pd.read_csv(training_data)

        # Get validation data if separate
        if validation_data is not None:
            print("not None??")
            validation_df = pd.read_csv(validation_data)

        # Extract features and targets
        feature_names = []
        target_names = []

        #Check if user gave features and targets as strings or ROSReadings (assume user only did one or other)

        if isinstance(self.features[0], str):
            print("Features and Targets provided as Strings")
            feature_names = self.features
            target_names = self.targets
        else:
            for feature in self.features:
                feature_names.append(feature.getName())

            for target in self.targets:
                target_names.append(target.getName())

        selected_features = training_df[feature_names]
        selected_targets = training_df[target_names]

        if validation_data is not None:
            validation_features = validation_df[feature_names]
            validation_targets = validation_df[target_names]
            validation_data = ((validation_features, validation_targets))

        # Perform custom processing if needed
        if custom_feature_processing is not None:
            selected_features = custom_feature_processing(selected_features)
        if custom_target_processing is not None:
            selected_targets = custom_target_processing(selected_targets)

        #Fit the model
        #self.model.fit(selected_features, selected_targets, validation_split = validation_split, verbose = verbose)
        self.model.fit(x = selected_features,
            y = selected_targets,
            batch_size = batch_size,
            epochs = epochs,
            verbose = verbose,
            callbacks = callbacks,
            validation_split = validation_split,
            validation_data = validation_data,
            shuffle = shuffle,
            class_weight = class_weight,
            sample_weight = sample_weight,
            initial_epoch = initial_epoch,
            steps_per_epoch = steps_per_epoch,
            validation_steps = validation_steps,
            validation_freq = validation_freq,
            max_queue_size = max_queue_size,
            workers = workers,
            use_multiprocessing = use_multiprocessing)

    def collectData(self, train_online = False):
        # launch simulation stuff
        self.data_collector.start()


class ROSReading():
    def __init__(self, name, description, rostopic, ros_message_type, custom_handler = None,
                 custom_data_extraction=None):
        self.name = name
        self.description = description
        self.rostopic = rostopic
        self.ros_message_type = ros_message_type
        self.custom_data_extraction = custom_data_extraction
        self.custom_handler = custom_handler

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        print("ROSReading name changed to " + self.name)

    def get_description(self):
        return self.description

    def set_name(self, description):
        self.description = description
        print("ROSReading description changed to " + self.description)

    def get_rostopic(self):
        return self.rostopic

    def set_rostopic(self, rostopic):
        self.rostopic = rostopic
        print("ROSReading " + self.name + " rostopic changed to " + self.rostopic)

    def get_ros_message_type(self):
        return self.ros_message_type

    def set_ros_message_type(self, ros_message_type):
        self.ros_message_type = ros_message_type
        print("ROSReading " + self.name + " message type changed")

    def get_custom_data_extraction(self):
        return self.custom_data_extraction

    def set_custom_data_extraction(self, new_data_extraction):
        self.custom_data_extraction = new_data_extraction
        print("ROSReading " + self.name + " data extraction method changed")

    def data_extraction(self, received_data):
        data = []
        if self.custom_data_extraction is not None:
            return self.custom_data_extraction(received_data)
        else:
            if self.ros_message_type == Twist:
                data.append(received_data.linear.x)
                data.append(received_data.linear.y)
                data.append(received_data.linear.z)
                return data
            elif self.ros_message_type == Pose:
                data.append(received_data.x)
                data.append(received_data.y)
                data.append(received_data.theta)
                return data
            else:
                print("Error: " + self.name + " is unrecognized message type")
                print("Please write a custom extraction method")

class ROSTask():
    def __init__(self, name, simulation_type, goal):
        self.name = name
        self.simulation_type = simulation_type
        self.goal = Pose()
        if goal == "random_goal":
            self.goal.x = random.uniform(0,10)
            self.goal.y = random.uniform(0,10)
        else:
            self.goal.x = goal[0]
            self.goal.y = goal[1]

    def start_simulation(self):
        turtle = TurtleBot()
        if self.simulation_type == "Turtlesim":
            sim_launch_cmd = "rosrun turtlesim turtlesim_node"
            self.launch = subprocess.Popen(sim_launch_cmd.split(), stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        return turtle

    def run(self, turtle):
        if self.simulation_type == "Turtlesim":
            rospy.wait_for_service('clear')
            rospy.wait_for_service('spawn')
            rospy.wait_for_service('kill')

            sim_clear_cmd = "rosservice call /clear"
            sim_spawn_cmd = "rosservice call /spawn " + str(self.goal.x) + " " + str(self.goal.y) + " 0 turtle2"
            sim_kill_cmd = "rosservice call /kill turtle2"
            self.clear = subprocess.Popen(sim_clear_cmd.split(), stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
            self.clear.kill()

            # self.kill = subprocess.Popen(sim_kill_cmd.split(), stdout=subprocess.PIPE,
            #                  stdin=subprocess.PIPE,
            #                  stderr=subprocess.PIPE)
            # self.kill.kill()
            self.spawn = subprocess.Popen(sim_spawn_cmd.split(), stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
            #self.spawn.kill()
            time.sleep(1)
            turtle.move2goal(self.goal)
            time.sleep(4)
            self.spawn.kill()
        elif self.simulation_type == "Gazebo":
            print("Gazebo Functionality has not been implemented yet")
