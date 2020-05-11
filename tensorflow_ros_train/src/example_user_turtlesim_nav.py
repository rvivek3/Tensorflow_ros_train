from source_code.Wrappers import *
from source_code.DataCollector import *
import time

def baseline_model():
    return None

features = []
targets = []

#handler (callback) :returns tuple: (subscription_index, message)
def turtle_pose_handler(message):
    #print("Turtle Pose Handler")
    if myModel.data_collector.collected_data.get(0,None) is None:
        myModel.data_collector.set_collected_data(0, message)

def turtle_cmd_vel_handler(message):
    #print("Turtle Vel Handler")
    if myModel.data_collector.collected_data.get(1, None) is None:
        myModel.data_collector.set_collected_data(1, message)


#Create new model
myModel = ROSModel("Turtlesim_controller", baseline_model);

#Set up wrappers for data
turtle_pose = ROSReading(["Turtle_x", "Turtle_y", "Turtle_theta"],
                         "turtle_pose", "/turtle1/pose",
                         Pose,
                         turtle_pose_handler)

turtle_cmd_vel = ROSReading(["Linear_x", "Linear_y", "Linear_z"],
                            "turtle_cmd_vel",
                            "/turtle1/cmd_vel",
                            Twist,
                            turtle_cmd_vel_handler)

#Specify features and targets
features.append(turtle_pose)
targets.append(turtle_cmd_vel)

myModel.set_features(features)
myModel.set_targets(targets)

#Data collector must be iniialized each time features and targets are set
myModel.initialize_data_collector()

myModel.collectData()

while(True):
    time.sleep(.1)