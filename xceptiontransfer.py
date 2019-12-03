## https://keras.io/applications/#usage-examples-for-image-Classification-models
## Code to train model Xception in transfer learning and balanced version
## parameter:
## <species> or <genus> to define if CNN learns classes of species or classes of genera and which images to use
## <pad> or <distort> defines how images were scaled to 299X299 by padding black rim or by distoting the image
## <train> or <test> defines if code is used for training or testing
## <version> training version for weight saving and loading
## <weightfile> only needed for testing. Chose the weights that are supposed to be tested
## to define which gpu is to be used call prgramm with: CUDA_VISIBLE_DEVICES=gpuID screen python3 xception.py ... gpuid = 0 or 1


from keras.utils import to_categorical
from keras.applications import Xception
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler
import pickle
import time
import subprocess


############################################# function to load the data ################################################
## load X, y, labeltonumber
def load_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    X = npzfile['X']
    y = npzfile['y']
    labeltonumber = npzfile['labeltonumber']
    # print(X)
    # print(y)
    # print(labeltonumber)
    return X, y, labeltonumber


########################################################################################################################

################################################ main ##################################################################
if len(sys.argv) != 6:
    sys.stderr.write(
        'Usage: xceptiontransfer.py [species|genus], [pad|distort], [train|test], <version>, <weightfile>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]
    modus = sys.argv[3]
    version = sys.argv[4]
    weightfile = sys.argv[5]

## for tests use data_{}_{}.test.npz
## /home/stine/repositories
filename = 'npz/data_{}_{}.npz'.format(mode, resize)
X, y, labeltonumber = load_data(filename)

## Olafenwa and Olafenva - 2018 #######################
## not sure if this correct since Ng and others say, that the mean and std
## is only to be computed on the training set and than subtracted and divided
## from / by the validation and test set.
# ## normalize the data
X = X.astype('float32') / 255
# plt.imshow(X[1])
# plt.show()
#
# ## Subtract the mean image
mean = X.mean(axis=0)
# print('mean: {}'.format(mean))
X = X - mean
# plt.imshow(X[1])
# plt.show()
# print(np.mean(X))


#######################################################

## shuffle data randomly before splitting (shuffle = TRUE as default in used function)
## use stratified sampling to devide training and test sets (training and test)
## sklearn.model_selection train_test_split (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)


## define the model (preset weights)
print('[INFO] defining model...')

## create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False)

## add a global spatial average pooling layer
layer = base_model.output
layer = GlobalAveragePooling2D()(layer)

## add a fully connected layer
predictions = Dense(len(labeltonumber), activation='softmax')(layer)

## create the model to train
model = Model(inputs=base_model.input, outputs=predictions)

## print a summary of the model
print(model.summary())

## values from Olafenwa and Olafenva - 2018 and
EPOCHS = 200
print('epochs: {}'.format(EPOCHS))
## batch normalization batch sizes: 64, 128, 256, 512
BATCHSIZE = 32
print('batchsize: {}'.format(BATCHSIZE))
STEPS_PER_EPOCH = len(X_train) / BATCHSIZE
print('steps per epoch: {}'.format(STEPS_PER_EPOCH))


## Olafenwa and Olafenva - 2018 #######################
## step decay
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


## pass the scheduler function to the Learning Rate Sceduler class
lr_scheduler = LearningRateScheduler(lr_schedule)

## directory in which to create models
time = time.time()
githash = subprocess.check_output(['git', 'describe', '--always']).strip()
save_modeldirectory = os.path.join(os.getcwd(), 'frogsumimodels/Xception_{}_{}_{}'.format(mode, resize, version))
save_csvdirectory = os.path.join(os.getcwd(), 'csvlogs/Xception_{}_{}_{}'.format(mode, resize, version))

## name of model files
model_name = 'Xception.{epoch:03d}.{val_acc:.3f}.hdf5'
csv_name = 'Xception_{}_{}_{}.csv'.format(mode, resize, version)

## create directory to save models if it does not exist
if not os.path.isdir(save_modeldirectory):
    os.makedirs(save_modeldirectory)
## create directory to save csv files if it does not exist
if not os.path.isdir(save_csvdirectory):
    os.makedirs(save_csvdirectory)

## join the directory with the model file
modelpath = os.path.join(save_modeldirectory, model_name)
## join the directory  with the csv file
csvpath = os.path.join(save_csvdirectory, csv_name)

file = open(save_modeldirectory + '/info.txt', 'w')
lines = ['githash: {}'.format(githash), 'timestamp: {}'.format(time), 'mode: {}'.format(mode),
         'resize: {}'.format(resize), 'version: {}'.format(version)]
file.writelines(lines)
file.close

## checkpoint that saves the best weights according to the validation accuracy
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=True)
## csv_logger to write losses and accuracies after each epoch in csv file
csv_logger = CSVLogger(filename=csvpath, separator=',', append=True)

print('[INFO] compiling model...')

## the top 2 (116) and 5 (86) xception blocks have been chosen to be trained, so the first 116 layers will be set
## immutable and the remeining will be continued to be trained
for layer in model.layers[:86]:
    layer.trainable = False
for layer in model.layers[86:]:
    layer.trainable = True

## Adam or RMSProp with step learning rate decay:
## https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
model.compile(optimizer=Adam(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

############### version balance (uncomment to balance the data set) ####################################################

X_train_reshape = X_train.reshape(len(y_train), -1)
ros = RandomOverSampler(random_state=42)
X_train_reshape_resample, y_train = ros.fit_resample(X_train_reshape, y_train)
X_train = X_train_reshape_resample.reshape(len(X_train_reshape_resample), 299, 299, 3)

############### end version balance (uncomment to balance the data set) ################################################

######################################## basic ###################################################################
## first version that was used for training
## only run for training by adding parameter 'train' when running script
print('[INFO] generating data...')
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

datagen.fit(X_train)
y_train_matrix = to_categorical(y_train, len(labeltonumber))
y_val_matrix = to_categorical(y_val, len(labeltonumber))

print('[INFO] start training...')
if modus == 'train':
    ## use validation fold for validation
    model.fit_generator(datagen.flow(X_train, y_train_matrix, batch_size=BATCHSIZE),
                        validation_data=[X_val, y_val_matrix], epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[checkpoint, lr_scheduler, csv_logger])

###################################### end version basic and balance ###################################################

####################################### run on test set ################################################################
## only run for testing by adding parameter 'test' when running script
elif modus == 'test':
    y_test_matrix = to_categorical(y_test, len(labeltonumber))

    print(model.metrics_names)
    # model.load_weights(save_modeldirectory + '/Xception_genus_pad_version1.1/Xception.109.0.964.hdf5')
    model.load_weights(save_modeldirectory + '/{}'.format(weightfile))
    accuracy = model.evaluate(x=X_test, y=y_test_matrix)
    ## get predicted labels for test set
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=-1)

    print('loss: {}, accuracy: {}'.format(accuracy[0], accuracy[1]))
    ## get precision, recall, f1-score and support for each class predicted on test set
    classreport = classification_report(y_test, y_pred, output_dict=True)
    ## print which label belongs to which species/genus
    for idx, label in enumerate(labeltonumber):
        classreport[str(idx)]['label'] = label
    # dataframe = pandas.DataFrame(classreport).transpose()
    # dataframe.to_csv(save_modeldirectory + '/Xception_genus_pad_version1.1/classreport.csv', header = ['f1-score', 'label', 'precision', 'recall', 'support'])
    cnf_matrix = confusion_matrix(y_test, y_pred)
    math_corrcoef = matthews_corrcoef(y_test, y_pred)
    print('classreport: {}'.format(classreport))
    print('confusion matrix: {}'.format(cnf_matrix))
    print('Mathews corrcoef: {}'.format(math_corrcoef))
    print('y_prob: {}'.format(y_prob))
    print('y_pred: {}'.format(y_pred))

    with open(save_modeldirectory + '/{}_{}_{}_{}.pkl'.format(modus, mode, resize, version), 'wb') as di:
        pickle.dump([accuracy, classreport, cnf_matrix, math_corrcoef, y_prob, y_pred], di)
