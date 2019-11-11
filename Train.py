import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Model Trainer')
parser.add_argument('--path', help='Path to data folder.', required=True)
parser.add_argument('--lite', help='Generate lite Model.', action='store_true')
args = parser.parse_args()


def load_dataset(input_path):
    features_list = []
    features_label = []

    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            for filename in os.listdir(input_path + "/" + dir):
                training_digit_image = cv2.imread(input_path + "/" + dir + "/" + filename)
                gray = cv2.cvtColor(training_digit_image, cv2.COLOR_BGR2GRAY)
                gray = np.array(gray, dtype='f').ravel()
                features_list.append(np.array(gray))
                features_label.append(np.float(dir))

    features_list = np.array(features_list)
    features_label = np.array(features_label)

    return features_list, features_label


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))
        if logs.get('loss') < 0.01 and logs.get('accuracy') > .999:
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


def scheduler(epoch):
    return 0.001 if epoch < 10 else float(0.001 * tf.math.exp(0.1 * (10 - epoch)))


train, labels = load_dataset(args.path)
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3, stratify=labels, random_state=0)
X_train /= 255.0
X_test /= 255.0

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1700).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(1700).batch(64)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(38, activation='softmax')
])

callbacks = myCallback()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=100,
          callbacks=[callbacks, tf.keras.callbacks.LearningRateScheduler(scheduler)])
model.save('model.h5')

if args.lite:
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('Model.tflite', 'wb').write(tflite_model)
