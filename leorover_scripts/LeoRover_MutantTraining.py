import sys
from os import chdir
from os.path import dirname, realpath
chdir(dirname(realpath(__file__)))
sys.path.append('..')
import numpy as np
import keras as K
import keras.layers as KL
import pickle
import tqdm
import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
import random
from tqdm import tqdm

parser=ArgumentParser(description='trainModel')
parser.add_argument('--TCL', help='Percentage of data to mutate', required=False)
parser.add_argument('--TAN', help='Percentage of data to swap', required=False)
parser.add_argument('--HLR', help='Learning rate', required=False)
parser.add_argument('--Original', help='Original', required=False)
parser.add_argument('--it', help='Iteration number', required=False)
parser.add_argument('--seed', help='Seed number', required=False)
args=parser.parse_args()
tlc=0
seed=int(args.seed)
random.seed(seed)
if args.TCL is not None:
    tlc=float(args.TCL)
    value=tlc
    title='TCL'
hlr=0.001
if args.HLR is not None:
    hlr=float(args.HLR)
    value=hlr
    title='HLR'
tan=0
if args.TAN is not None:
    tan=float(args.TAN)
    value=tan
    title='TAN'
if args.Original is not None:
    value=0
    title='ORIG'
it=int(args.it)


# Height of the crop area (number of pixels, starting from upper edge of image that will be cropped)
crop_height = 220

# Save directory for trained model
save_dir = "model/mutants"

# Chose the name of saved tflite model
tflite_model_name = save_dir + "/tflite_model_mutant_"+title+"_"+str(value)+"_"+str(it)

print("Reading data")
data=pd.read_csv('content/data_leoRover/MLdata.csv')

X_train, X_test, y_train, y_test = train_test_split(data.values[:,:-1], data.values[:,-1],test_size=1/5,random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=1/4,random_state=seed)

X_train=pd.DataFrame(data=X_train, columns=data.columns[:-1])
y_train=pd.DataFrame(data=y_train, columns=[data.columns[-1]])
trainData=pd.concat([X_train,y_train],axis=1)
for i in range(0,len(trainData)):
  target=np.fromstring(trainData.Target[i].split('[')[1].split(']')[0], dtype=float, sep=',')
  trainData.Target[i]=target

X_test=pd.DataFrame(data=X_test, columns=data.columns[:-1])
y_test=pd.DataFrame(data=y_test, columns=[data.columns[-1]])
testData=pd.concat([X_test,y_test],axis=1)
for i in range(0,len(testData)):
  target=np.fromstring(testData.Target[i].split('[')[1].split(']')[0], dtype=float, sep=',')
  testData.Target[i]=target

X_val=pd.DataFrame(data=X_val, columns=data.columns[:-1])
y_val=pd.DataFrame(data=y_val, columns=[data.columns[-1]])
validationData=pd.concat([X_val,y_val],axis=1)
for i in range(0,len(validationData)):
  target=np.fromstring(validationData.Target[i].split('[')[1].split(']')[0], dtype=float, sep=',')
  validationData.Target[i]=target

dataToMutate=int(tlc*len(trainData))

toMutate=random.sample(range(len(trainData)),dataToMutate)
for i in range(0,dataToMutate):
    index=toMutate[i]
    trainData.Target[index][0]=random.random()*0.5
    trainData.Target[index][1]=random.random()*4-2

print("Data ready")

def mutateTan(image,tan):
    size=120*160
    numOfMutations=int(tan*size)
    for i in range(0, numOfMutations):
        col=random.randint(0,119)
        row=random.randint(0,159)
        image[col][row]=1-image[col][row]
    return image
    

def preprocess(img, dim, hsv_lower, hsv_upper):
    # converting to hsv
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # croping the img
    crop_img = hsv_img[crop_height:hsv_img.shape[0], :]
    # catching color mask
    color_mask = cv2.inRange(crop_img, hsv_lower, hsv_upper)
    # conveting values to float
    float_img = color_mask.astype(np.float32)
    # resizing
    resized_img = cv2.resize(float_img, (dim[1], dim[0]))
    # normalizing
    final_img = resized_img / 255.0
    
    return final_img[:,:,np.newaxis]

def get_data(image,test_dimension):
    img = cv2.imread(image.imgpath)
    hsv_min=np.fromstring(image.hsv_min.split('[')[1].split(']')[0], dtype=float, sep=',')
    hsv_max=np.fromstring(image.hsv_max.split('[')[1].split(']')[0], dtype=float, sep=',')
    target=image.Target
    hsv_lower=(hsv_min[0],hsv_min[1],hsv_min[2])
    hsv_upper=(hsv_max[0],hsv_max[1],hsv_max[2])
    preprocessed_test = preprocess(img, test_dimension,hsv_lower,hsv_upper) 
    
    return preprocessed_test , target
class DataGenerator(K.utils.Sequence):
    def __init__(self, data,tan, batch_size=32, dim=(32,32,32), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.tan=tan
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_linear = np.empty((self.batch_size, 1), dtype=float)
        y_angular = np.empty((self.batch_size, 1), dtype=float)
        i=0
        for index in indexes:
            image=self.data.iloc[index]
            preprocessed_test, target=get_data(image,self.dim)
            preprocessed_test=mutateTan(preprocessed_test,self.tan)
            X[i,:] = preprocessed_test

            y_linear[i] = target[0]
            y_angular[i] = target[1]
            i=i+1
        return X, {'linear': y_linear, 'angular': y_angular}

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y
    
img_in = KL.Input(shape=(120, 160, 1), name='img_in')
x = img_in

x = KL.Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
x = KL.Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
x = KL.Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
x = KL.Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
x = KL.Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

x = KL.Flatten(name='flattened')(x)
x = KL.Dense(units=100, activation='linear')(x)
x = KL.Dropout(rate=.1)(x)
x = KL.Dense(units=50, activation='linear')(x)
x = KL.Dropout(rate=.1)(x)

linear = KL.Dense(units=1, activation='linear', name='linear')(x)

angular = KL.Dense(units=1, activation='linear', name='angular')(x)

model = K.Model(inputs=[img_in], outputs=[linear, angular])
GPUs_num = len(tf.config.list_physical_devices('GPU'))
CPUs_num = len(tf.config.list_physical_devices('CPU'))
device = None
device = '/GPU:0' if GPUs_num > 0 else '/CPU:0'
opt=tf.keras.optimizers.Adam(learning_rate=hlr)
with tf.device(device):
    model.compile(optimizer=opt,
                  loss={'linear': 'mean_squared_error', 'angular': 'mean_squared_error'},
                  loss_weights={'linear': 0.3, 'angular': 0.7})
    
callbacks = [
        K.callbacks.ModelCheckpoint(save_dir, save_best_only=True),
        K.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=.0005,
                                  patience=10,
                                  verbose=True,
                                  mode='auto'),
        tfa.callbacks.TQDMProgressBar(),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=5,
                                      min_lr=0.001),
        K.callbacks.TensorBoard(log_dir='./logs', profile_batch=(0, 10))


    ]
params = {'dim': (120, 160),
          'batch_size': 64,
          'n_channels': 1,
          'shuffle': True}


with tf.device(device):
    training_generator = DataGenerator(trainData,tan, **params)
    validation_generator = DataGenerator(validationData,0, **params)
    
with tf.device(device):
    hist = model.fit(training_generator,
                           validation_data=validation_generator,
                           use_multiprocessing=False,
                           workers=6,
                           callbacks=callbacks,
                           epochs=100)
    
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_model_name + '.tflite', 'wb') as f:
    f.write(tflite_model)
    
test_dimension = (120, 160)
y_prediction_linear=[]
y_prediction_angular=[]
y_target_linear=[]
y_target_angular=[]

print("Predicting test data ...")
for i in range(0,len(testData)):
  print("Made "+str(i)+"/"+str(len(testData)))
  image=testData.iloc[i]
  preprocessed_test, target=get_data(image,test_dimension)
  keras_input = np.empty((1, 120, 160, 1))
  keras_input[0,:] = preprocessed_test
  prediction=model.predict(keras_input)
  predicted_linear=prediction[0][0][0]
  predicted_angular=prediction[1][0][0]
  y_prediction_linear.append(predicted_linear)
  y_prediction_angular.append(predicted_angular)
  y_target_angular.append(target[1])
  y_target_linear.append(target[0])
print("Done")

mse = tf.keras.losses.MeanSquaredError()
mse_linear = mse(y_target_linear,y_prediction_linear).numpy()
mse_angular = mse( y_target_angular,y_prediction_angular,).numpy()
print("Linear mse loss is: "+str(round(mse_linear,4)))
print("Angular mse loss is: "+str(round(mse_angular,4)))
print("Total loss is: "+str(round(mse_linear*0.3+mse_angular*0.7,4)))
data=pd.read_excel("Results.xlsx")
data=data.append({"Mutator":title,"Value":value,"Iteration":it,"Linear mse loss":round(mse_linear,4),"Angular mse loss":round(mse_angular,4),"Total mse loss":round(mse_linear*0.3+mse_angular*0.7,4)},ignore_index=True)
data.to_excel("Results.xlsx",index=False)
