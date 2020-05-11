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

class ROSModel():
    def __init__(self, name, ros_features, ros_targets, baseline_model):
        self.name = name
        self.model = baseline_model()
        self.baseline_model = baseline_model
        self.features = ros_features
        self.targets = ros_targets
        print(name + " Model Created")

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        print("Model name changed to " + self.name)

    def get_features(self):
        return self.features;

    def set_features(self, ros_features):
        self.features = ros_features
        print("Features of " + self.name + "set")

    def get_targets(self):
        return self.targets;

    def set_targets(self, ros_targets):
        self.targets = ros_targets
        print("Targets of " + self.name + "set")

    def get_baseline_model(self):
        return self.baseline_model

    def set_baseline_model(self, baseline_model):
        self.baseline_model = baseline_model
        print("Baseline Model of " + self.name + "set")

    def summary(self):
        print("Summary of" + self.name)
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
            validation_df = pd.read_csv(validation_data)

        # Extract features and targets
        feature_names = []
        target_names = []
        for feature in self.features:
            feature_names.append(feature.getName())

        for target in self.targets:
            target_names.append(target.getName())

        selected_features = training_df.drop(target_names)
        selected_targets = training_df.drop(feature_names)

        if validation_data is not None:
            validation_features = validation_df.drop(target_names)
            validation_targets = validation_df.drop(feature_names)

        # Perform custom processing if needed
        if custom_feature_processing is not None:
            selected_features = custom_feature_processing(selected_features)
        if custom_target_processing is not None:
            selected_targets = custom_target_processing(selected_targets)

        #Fit the model
        self.model.fit(selected_features,
            selected_targets,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            (validation_features,validation_targets) if validation_data is not None else None,
            shuffle,
            class_weight,
            sample_weight,
            initial_epoch,
            steps_per_epoch,
            validation_steps,
            validation_batch_size,
            validation_freq,
            max_queue_size,
            workers,
            use_multiprocessing)


