# tensorflow_ros_train
ROS package development for streamlining tensorflow 2 model training using ROS data. Tensorflow models can be seamlessly integrated in ROS.
See overview at https://www.youtube.com/watch?v=rslBpMuezN8

The package currently supports TurtleSim, though support for more simulation packages including Gazebo and stdr_simulator are coming.

A typical user will use the package as follows:

## Create a tensorflow model
```
def baseline_model():
  #Tensorflow stuff
  return model
 ```
## Specify the data your model needs using `ROSReading`
### `ROSReading` is a container class similar to a ROS `msg` but specialized for being inputted to a model.
```
#construct a ROSReading using ROSReading([Names, of, data, elements], "name", "topic/to/subscribe/to", MessageType)

turtle_pose = ROSReading([“Turtle_x”, “Turtle_y”, “Turtle_theta”], “turtle_pose”, “/turtle1/pose”, Pose)

turtle_vel = ROSReading([“Linear_x”, “Linear_y”, “Linear_z”], “turtle_cmd_vel”, Twist)
```
## Create a `ROSModel`
### `ROSModel` is a container class around a tensorflow model that provides convenient methods.
```
#construct a ROSModel using ROSModel("name", tensorflow_model_generator, [list, of, ROSReading, features], [list, of ROSReading, targets]

myModel = ROSModel(“Turtlesim_controller”, baseline_model, features = [turtle_pose], targets = [turtle_vel])
```
## Auto-collect data using an separate simulation launcher. 
### A csv is automatically generated containing the model's required features and targets.
```
myModel.collectData( )

in terminal: rosrun turtlesim turtlesim_node
in new terminal window: rosrun turtlesim turtle_teleop_key
```
Data autologged at csv file in same directory, named your_model_name_data.csv

## Run automated simulations with `ROSTask`
### `ROSTask` is custom class with built in simulators and controllers.

Currently, a PID controller for turtlebot move to goal simulations is included
```
task = ROSTask(“Turtlesim_nav”, simulation_type = “Turtlesim”, controller = “PID”, goal = “random”)
```
### Launch a simulation of 1000 of these tasks and collect data throughout
```
myModel.collectData(task, numTasks = 1000, train_online = False)
#note: train_online is not implemented yet, but will allow users to collect data and train models simultaneously 
```
## Train the model alongside ROS using ROSModel.fit()
```
collected_data = myModel.collectData( ) 

# OR collected_data = “path/to/data.csv”

myModel.fit(collected_data, epochs = 15, validation_split = .2) 

```
### For more data processing control, using custom target and feature processing arguments
```
def process_features(features):
    #custom calculations, normalization etc.
    return features

def process_targets(targets):
    #for a classification problem
    targets = np_utils.to_categorical(targets)
    return targets
    
myModel.fit(collected_data, custom_feature_processing = process_features,
custom_target_processing = process_targets)

```
## Access the Tensorflow model itself anytime using ROSModel.model
```
myModel.model.[any tensorflow or keras method]
```
## Check out the two example user scripts:

### example_user_turtlesim_nav.py:
Very similar to above example. Only difference is that handler functions are necessary (passed to the ROSReadings) because ApproximateTimeSynchronizer in ROS does not work with Python 3, so a workaround is used. The release of ROS Noetic will make these functions obsolete.

### example_user_turtlebot_sim.py
Uses precollected data from simulations of a turtlebot navigating to a random goal in a Gazebo simulation using a DWA controller. Ultimately, the functionality for collecting this data will be included in this package.

The user specifies a custom tensforflow model to map turtlebot laser scan messages and angular deviation to the goal to steering commands for the turtlebot.

Because data collection was performed externally, the user specifies features and targets using only Strings (as opposed to ROSReadings) and passes a path to an existing csv. Only the data of interest will be extracted.

The user uses custom target and feature processing for data normalization

The user then fits the model using only a reference to the collected data. In 30 epochs, the model is able to reach a 70% test accuracy and 57% validation accuracy.







