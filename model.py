import csv
import numpy as np
import cv2
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.utils import plot_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('csvfile', 'samples/driving_log.csv', 'The path to the csv file of training data.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for neural network.')
flags.DEFINE_integer('epochs', 10, 'Amount of epochs to train neural network.')
flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for backprop.')


def parse_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def translate_img(image, angle):
    rows, cols, _ = image.shape
    trans_range = 100
    num_pixels = 10
    val_pixels = 0.4
    trans_x = trans_range * np.random.uniform() - trans_range / 2
    result_angle = angle + trans_x / trans_range * 2 * val_pixels
    trans_y = num_pixels * np.random.uniform() - num_pixels / 2
    trans_mat = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    return cv2.warpAffine(image, trans_mat, (cols, rows)), result_angle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    max_angle = 1.
    min_angle = -1.
    correct_value = 0.25
    correction = [0, correct_value, -correct_value]

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])

                amount_cameras = 1
                if min_angle + correct_value <= angle <= max_angle - correct_value:
                    amount_cameras = 3

                for col in range(amount_cameras):
                    img = cv2.imread(batch_sample[col])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    result_angle = angle + correction[col]

                    angles.append(result_angle)

                    # add translation
                    trans_img, trans_angle = translate_img(img, result_angle)
                    images.append(trans_img)
                    angles.append(trans_angle)

                    # add vertical flipping
                    if not np.isclose(result_angle, 0.):
                        images.append(cv2.flip(img, 1))
                        angles.append(-result_angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


def grayscale(image):
    from keras.backend import tf as ktf
    gray = ktf.image.rgb_to_grayscale(image)
    return gray


def resize(image):
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(image, (32, 32))
    return resized


def main(_):
    samples = parse_csv(FLAGS.csvfile)

    train_samples, validation_samples = train_test_split(samples, test_size=0.1)
    print(len(train_samples))
    print(len(validation_samples))
    train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
    validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(grayscale, name='grayscale'))
    model.add(Lambda(resize, name='resize'))
    model.add(Lambda(lambda x: (x / 127.5) - 1., name='normalize'))
    model.add(Conv2D(24, (5, 5), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(36, (5, 5), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(48, (5, 5), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=FLAGS.learning_rate))
    plot_model(model, to_file='examples/model.png', show_shapes=True)

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / FLAGS.batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples) / FLAGS.batch_size, epochs=FLAGS.epochs)

    model.save('model.h5')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
