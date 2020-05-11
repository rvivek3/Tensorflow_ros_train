from nodes.Classes import *

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
	#dropOut2 = Dropout(.25)(avg_pool)
	#flatten = Flatten()(conv5)
	# print(type(flatten))
	# # #print(model.summary())
	primary = Concatenate(axis=-2)([angDevStream, avg_pool])
	primary = Flatten()(primary)
	# # print(primary)
	primaryHidden = Dense(units = 1024, activation='relu')(primary)
	#do3 = Dropout(.4)(primaryHidden)
	primaryHidden2 = Dense(units = 512, activation='relu')(primaryHidden)
	#do4 = Dropout(.4)(primaryHidden2)
	primaryHidden3 = Dense(units = 256, activation = 'relu')(primaryHidden2)
	finalOut = Dense(units=3, activation = 'softmax')(primaryHidden3)
	model = Model(inpTensor,finalOut)

	# Compile model
	sgd = optimizers.SGD(lr=0.00001, momentum=0.0, nesterov=False)
	adam = keras.optimizers.Adam(lr=0.001)
	model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=adam, metrics=['accuracy'])
	# optimizer could be "adam"
	return model

myModel = ROSModel("Turtlebot_steering_controller",['feature1','feature2'],['target1','target2'],baseline_model);
myModel.summary()
#print(myModel.get_features())
collectedData = '/media/rajan/easystore/ORS_DATA/dat4mod.csv'
myModel.fit(collectedData, epochs = 2)