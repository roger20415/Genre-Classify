import os
import numpy as np
import tensorflow as tf
from package import ml_tools
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad, RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model_name = "CNN"
tf_ver = 1 #tensorflow version

if tf_ver == 1:
    # tensorflow1 option
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    gpu_options = tf.GPUOptions(allow_growth = True) # Dynamic Adjustment
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
elif tf_ver == 2:
    # tensorflow2 option
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True) #Dynamic Adjustment
    sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
    print("GPUs: ", len(tf.compat.v1.config.experimental.list_physical_devices('GPU')))
else:
    print("\nFalse: tensorflow only 1 or 2\n")
    exit()


#%% trainging stage
X_train, X_test, y_train, y_test = ml_tools.feature_load(ml_tools.feature_path)

num_class = len(set(y_test)) #number of classifying classes, binary classify is 2 and multiply classify >= 3
y_train, y_test = to_categorical(y_train, num_class), to_categorical(y_test, num_class)

if num_class == 2:
    loss = 'binary_crossentropy'
    last_active = "sigmoid"
else:
    loss = 'categorical_crossentropy'
    last_active = "softmax"

# functional API
inputs = Input(shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]))
x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'valid')(inputs)
x = MaxPooling2D(pool_size = (2, 2), padding = 'valid')(x)
x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'valid')(x)
x = Flatten()(x)
x = Dense(units = 128, activation = 'relu')(x)
#x = Dropout(0.25)(x)
outputs = Dense(num_class, activation = last_active)(x)
model = Model(inputs = inputs, outputs = outputs)
model.summary()
model.compile(loss = loss,
              optimizer = Nadam(lr = 0.0015, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, schedule_decay = 0.003),
              metrics = ['acc'])

reducelr = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.7,
                             patience = 2, verbose = 1,
							 mode = 'auto',  min_delta=0.01,
							 cooldown = 0, min_lr = 0.00001 )

earlystop = EarlyStopping(monitor = "val_acc", patience = 3,
                          verbose = 1) 

checkpointer = ModelCheckpoint(
                        filepath = "./model/" + model_name + ".hdf5",
                        monitor = "acc",
                        mode = "max",
                        verbose = 1,
                        save_best_only = True)
print("\n\n--Model training start--\n\n")

history = model.fit(X_train, y_train,
                    batch_size = 64, #be careful too big will run out of memory
                    epochs = 10,
                    verbose = 1,
                    callbacks = [reducelr, earlystop, checkpointer],
                    validation_split = 0.05
                    shuffle = True)

with open("./model/" + model_name + ".json", "w") as f:
    f.write(model.to_json())

print("\n\n--Model training finish--\n\n")


#%% testing stage
print("\n\n--Model testing start--\n\n")
y_pred = model.predict(X_test)

#from categorical to digital
y_pred, y_test = y_pred.argmax(axis = 1), y_test.argmax(axis = 1)

# show y_test and y_pred
print("\n\ny_test.shape\n",y_test.shape)
print("\ny_test\n", y_test)

print("\n\ny_pred.shape\n",y_pred.shape)
print("\ny_pred\n", y_pred)

# classification report
print("\n\nClassification report:\n")
print(metrics.classification_report(y_test, y_pred))
print("\n")

# Confusion matrix
cm = confusion_matrix(y_pred, y_test)
plt.figure(figsize = (9, 9))
plt.title('Confusion Matrix', fontsize = 30)
#X_labels = ['blues', 'classical']
#y_labels = ['blues', 'classical']

sns.heatmap(cm, annot = True, cmap = 'YlOrRd', fmt = 'd', cbar = False,
            #xticklabels = X_labels,
            #yticklabels = y_labels,
            annot_kws={"size":30})

plt.xlabel("Predict", fontsize = 20)
plt.ylabel("True", fontsize = 20)
plt.show()
