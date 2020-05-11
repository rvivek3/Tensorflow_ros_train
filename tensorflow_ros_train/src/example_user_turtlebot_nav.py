from source_code.Wrappers import *
from source_code.DataCollector import *


def baseline_model():
	bn = False
	kernel_regularizer_strength=.0001
	kernel_regularizer = None #regularizers.l1(kernel_regularizer_strength)
	BatchNorm = lambda x : BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
	                                   gamma_initializer='ones', moving_mean_initializer='zeros',
	                                   moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
	                                   beta_constraint=None, gamma_constraint=None)(x) if bn else x
	#create model with functional api model
	inpTensor = Input(shape = (641,)) #input is ang dev and 640 distances
	angDevStream = Lambda(lambda x: x[:,:1])(inpTensor) #extract ang dev tensor
	angDevStream = Reshape(target_shape=(-1, 1))(angDevStream)
	distanceStream = Lambda(lambda x: tf.keras.backend.expand_dims(x[:,1:],1))(inpTensor)
	distanceStream = Reshape(target_shape=(-1, 1))(distanceStream)
	distanceConv = BatchNorm(Conv1D(filters=64, kernel_size=7, strides=3, activation="linear", kernel_regularizer=kernel_regularizer, use_bias=not bn)(distanceStream))
	distanceConv = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(distanceConv)
	pool = MaxPooling1D(pool_size=3)(distanceConv)
	# dropOut1 = Dropout(.25)(pool)
	distanceConv2 = BatchNorm(Conv1D(filters=64, kernel_size=3, strides=1, activation="linear", kernel_regularizer=kernel_regularizer,
                       use_bias=not bn)(pool))
	distanceConv2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(distanceConv2)
	conv3 = BatchNorm(Conv1D(filters=64, kernel_size=3, strides=1, activation="linear", kernel_regularizer=kernel_regularizer,
                       use_bias=not bn)(distanceConv2))
	concat1 = Concatenate(axis=-2)([pool, conv3])
	relu1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(concat1)
	conv4 = BatchNorm(Conv1D(filters=64, kernel_size=3, strides=1, activation="linear", kernel_regularizer=kernel_regularizer,
                       use_bias=not bn)(relu1))
	conv4 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv4)
	conv5 = BatchNorm(Conv1D(filters=64, kernel_size=3, strides=1, activation="linear", kernel_regularizer=kernel_regularizer,
                       use_bias=not bn)(conv4))
	concat2 = Concatenate(axis=-2)([conv3, conv5])
	relu2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(concat2)
	avg_pool = AveragePooling1D(pool_size=3, strides=None, padding='valid', data_format='channels_last')(relu2)
	avg_pool = Reshape(target_shape=(-1, 1))(avg_pool)
	primary = Concatenate(axis=-2)([angDevStream, avg_pool])
	primary = Flatten()(primary)
	primaryHidden = Dense(units = 1024, activation='relu')(primary)
	primaryHidden2 = Dense(units = 512, activation='relu')(primaryHidden)
	primaryHidden3 = Dense(units = 256, activation = 'relu')(primaryHidden2)
	finalOut = Dense(units=3, activation = 'softmax')(primaryHidden3)
	model = Model(inpTensor,finalOut)

	# Compile model
	sgd = optimizers.SGD(lr=0.00001, momentum=0.0, nesterov=False)
	adam = keras.optimizers.Adam(lr=0.001)
	model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=adam, metrics=['accuracy'])
	return model


features = []
features.append("Ang Dev (straight line to goal)")
for i in range(1, 641):
	features.append("Distance " + str(i))
targets = ['Forward(0), Left(1), or Right(2)']


myModel = ROSModel("Turtlebot_steering_controller",
				   features,
				   targets,
				   baseline_model);
myModel.summary()
#print(myModel.get_features())
collectedData = '/media/rajan/easystore/ORS_DATA/dat4mod.csv'

def processFeatures(features):
	for (columnName, columnData) in features.iteritems():
		if "Ang" not in columnName:
			features[columnName] = features[columnName].where(features[columnName] != 10, features[columnName] - 4)
			features[columnName] = features[columnName] / 6

		else:
			features[columnName] /= 3.14
	return features

def processTargets(targets):
	targets = np_utils.to_categorical(targets)
	return targets

myModel.fit(collectedData,
			epochs = 1,
			custom_feature_processing=processFeatures,
			custom_target_processing=processTargets,
			validation_split=.2,
			verbose=2)
