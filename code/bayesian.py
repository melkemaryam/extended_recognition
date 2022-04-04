# set the matplotlib backend so figures can be saved in the background
import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use("Agg")

# import packages
import argparse
from datetime import datetime
import numpy as np
import os
import random
import cv2
import imutils
from imutils import paths
import kerastuner
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from skimage import exposure
from skimage import io
from skimage import transform
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MSE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from arguments import Args


class Bayesian_Optimisation:

	def main_train_net(self):

		arg = Args()
		self.args = arg.parse_arguments()

		self.net = Bayesian_Optimisation()

		# prepare the data and the model
		self.prepare_data()
		self.tuning()
		history = self.training()

		# save data in a plot
		#self.save_data(history)

		# predict labels
		self.prediction_process()

	def build_net(self, hp):

		self.model = Sequential()
		#hp_filters = hp.Int('filters', min_value = 2, max_value = 16, step = 2)

		self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Dropout(0.3))
		self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Dropout(0.4))
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(43, activation='softmax'))

		# compile model
		#opt = SGD(lr=0.001, momentum=0.9)
		print("[INFO] compiling model...")
		
		hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])
		opt = Adam(learning_rate=hp_learning_rate)#, momentum=0.9)
		
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		
		return self.model


	def load_images(self, pwd, path_to_csv):

		labels = []
		data = []

		# load data
		rows = open(path_to_csv).read().strip().split("\n")[1:]
		random.shuffle(rows)

		# loop over the rows of the CSV file
		for (i, row) in enumerate(rows):
			
			# print status update
			if i > 0 and i % 2000 == 0:
				print("[INFO] processed {} total images".format(i))

			# get classId and path to image
			(label, path_to_image) = row.strip().split(";")[-2:]

			# create full path to image
			path_to_image = os.path.sep.join([pwd, path_to_image])
			image = io.imread(path_to_image)

			# resize the image and perform CLAHE
			image = transform.resize(image, (32, 32))
			image = exposure.equalize_adapthist(image, clip_limit=0.1)

			# update the list of data and labels
			data.append(image)
			labels.append(int(label))

		# convert the data and labels to NumPy arrays
		data = np.array(data)
		labels = np.array(labels)

		# return a tuple of the data and labels
		return (data, labels)


	def get_sign_names(self):
		
		# load sign names
		sign_names = open("../sign_names_all.csv").read().strip().split("\n")[1:]
		
		sign_names = [s.split(";")[1] for s in sign_names]

		return sign_names

	def prepare_data(self):

		# derive the path to the training and testing CSV files
		path_to_train = os.path.sep.join([self.args["dataset"], "Train.csv"])
		path_to_test = os.path.sep.join([self.args["dataset"], "Test.csv"])

		# load the training and testing data
		print("[INFO] loading training and testing data...")
		(self.train_X, self.train_Y) = self.load_images(self.args["dataset"], path_to_train)
		(self.test_X, self.test_Y) = self.load_images(self.args["dataset"], path_to_test)

		# scale data to the range of [0, 1]
		self.train_X = self.train_X.astype("float32") / 255.0
		self.test_X = self.test_X.astype("float32") / 255.0

		# one-hot encode the training and testing labels
		self.num_images = len(np.unique(self.train_Y))
		self.train_Y = to_categorical(self.train_Y, self.num_images)
		self.test_Y = to_categorical(self.test_Y, self.num_images)

		# calculate the total number of images in each class and
		# initialize a dictionary to store the class weights
		self.total_images_class = self.train_Y.sum(axis=0)
		self.total_weight_class = dict()

		# loop over all classes and calculate the class weight
		for i in range(0, len(self.total_images_class)):
			self.total_weight_class[i] = self.total_images_class.max() / self.total_images_class[i]


	def tuning(self):
		
		self.tuner = BayesianOptimization(
			self.build_net,
			objective='accuracy',
			max_trials=20,
			executions_per_trial=1,
			directory='bayesian_data') #change the directory name here  when rerunning the cell else it gives "Oracle exit error" 

		self.write_report(self.tuner.search_space_summary())

		self.tuner.search(self.train_X, self.train_Y,
			epochs=20,
			validation_data=(self.test_X, self.test_Y))

		self.write_report(self.tuner.results_summary())

		self.best_hyperparameters = self.tuner.get_best_hyperparameters(1)[0]
		print(self.best_hyperparameters.values)
		self.write_report(self.best_hyperparameters.values)

		# probing function tuner.get_best_hyperparameters(1) #skip this if details are not required
		print(type(self.tuner.get_best_hyperparameters(1))) #list
		self.write_report(type(self.tuner.get_best_hyperparameters(1)))
		for data in self.tuner.get_best_hyperparameters(1):
			print(data.values)
			self.write_report(data.values)

		n_best_models = self.tuner.get_best_models(num_models=2)
		print(n_best_models[0].summary()) # best-model summary # gives info more than just model summary but dont know what
		self.write_report(n_best_models[0].summary())


	def write_report(self, report):

		file = open("../reports_all/bayesian/test_report_bo.txt", "a")

		file.write(str(report))
		file.write("\n")
		file.close()

		print("[INFO] report written")

	def training(self):
		# train the network
		print("[INFO] training network...")

		log_dir = "../logs/fit_bo/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
		callback = EarlyStopping(monitor='loss', patience=5)

		self.train = self.tuner.hypermodel.build(self.best_hyperparameters)
		history = self.train.fit(self.train_X, self.train_Y,
					epochs=200,
					validation_data=(self.test_X, self.test_Y), 
					callbacks=[callback, tensorboard],)

		self.evaluate()

		return history

	def evaluate(self):

		# evaluate the network
		print("[INFO] evaluating network...")

		sign_names = self.get_sign_names()
		predictions = self.model.predict(self.test_X)
		self.report = classification_report(self.test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=sign_names)
		
		print(self.report)
		self.write_report(self.report)

		_, acc = self.train.evaluate(self.test_X, self.test_Y, verbose=0)
		print('> %.3f' % (acc * 100.0))
		
	def save_data(self, history):
		
		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(self.args["model"]))
		self.train.save(self.args["model"])
		self.epochs_run = len(self.train.history['loss'])

		plt.plot(self.epochs_run, history.history['accuracy'])
		plt.plot(self.epochs_run, history.history['val_accuracy'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Val'], loc='upper left')
		plt.show()

		# Plot training & validation loss values
		plt.plot(self.epochs_run, history.history['loss'])
		plt.plot(self.epochs_run, history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Val'], loc='upper left')
		plt.show()


	def prediction_process(self):

		# grab the paths to the input images, shuffle them, and grab a sample
		print("[INFO] predicting...")

		sign_names = self.get_sign_names()

		paths_to_image = list(paths.list_images(self.args["images"]))
		random.shuffle(paths_to_image)

		# choose only 30 images
		paths_to_image = paths_to_image[:21]

		# loop over the image paths
		for (i, path_to_image) in enumerate(paths_to_image):
			
			# resize images and perform CLAHE
			image = io.imread(path_to_image)
			image = transform.resize(image, (32, 32))
			image = exposure.equalize_adapthist(image, clip_limit=0.1)

			# preprocess the image by scaling it to the range [0, 1]
			image = image.astype("float32") / 255.0
			image = np.expand_dims(image, axis=0)

			# make predictions using the traffic sign recognizer CNN
			predictions = self.train.predict(image)
			j = predictions.argmax(axis=1)[0]
			label = sign_names[j]

			# load the image using OpenCV, resize it, and draw the label
			image = cv2.imread(path_to_image)
			image = imutils.resize(image, width=128)
			cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			# save the image to disk
			p = os.path.sep.join([self.args["predictions"], "{}.png".format(i)])

			cv2.imwrite(p, image)

