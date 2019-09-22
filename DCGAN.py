import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import (
    UpSampling2D,
    Conv2D,
    Activation,
    BatchNormalization,
    Reshape,
    Dense,
    Input,
    LeakyReLU,
    Dropout,
    Flatten,
    ZeroPadding2D,
)
from keras.optimizers import Adam
from PIL import Image
import numpy as np
import os, sys
import argparse
from ast import literal_eval

from imageio import imsave
from pathlib import Path

path_current_file = Path(os.path.realpath(__file__))
path_module = path_current_file.parents[
    2
]  # this file is located 3 folders deep: /, /model, /model/Kaggle_DCGAN_Dogs
print(f"{path_module}")
sys.path.append(f"{path_module}")


# Some convenience function I already prebuilt: github.com/dyt811/PythonUtils/
from PythonUtils.PUFile import unique_name
from PythonUtils.PUFolder import recursive_list, create

path_log_run = path_module / Path("logs") / unique_name()

create(path_log_run)  # create the training specific folder if it doesn't exist already.

# dimensions_noise = 100

"""
Largely inspired from source: https://github.com/DataSnaek/DCGAN-Keras
"""


class DCGAN:
    def __init__(self, path_discriminator, path_generator, path_output, img_size):

        # Key image properties.
        self.img_size = img_size  # default x and y
        self.channels = 3

        self.dimensions_noise = 100

        self.upsample_layers = 5
        self.starting_filters = 64
        self.kernel_size = 3

        self.iteration_discriminator = 2
        self.iteration_generator = 3

        if path_discriminator is not None:
            self.discriminator_path = Path(path_discriminator)  # .as_posix()
        if path_generator is not None:
            self.generator_path = Path(path_generator)  # .as_posix()

        self.output_directory = Path(path_output)  # .as_posix()

        self.model_name = Path(path_output) / Path(f"{unique_name()}_{__name__}.h5")

    def build_generator(self):
        """
        Build and return Keras.models.Model a generator model with the summary indicated below.
        :return: a generator model, which require TWO parameters:
            NOISE SHAPE:
            IMAGE:
        """
        noise_shape = (self.dimensions_noise,)

        # This block of code can be a little daunting, but essentially it automatically calculates the required starting
        # array size that will be correctly upscaled to our desired image size.
        #
        # We have 5 Upsample2D layers which each double the images width and height, so we can determine the starting
        # x size by taking (x / 2^upsample_count) So for our target image size, 256x192, we do the following:
        # x = (192 / 2^5), y = (256 / 2^5) [x and y are reversed within the model]
        # We also need a 3rd dimension which is chosen relatively arbitrarily, in this case it's 64.
        model = Sequential()
        model.add(
            Dense(
                self.starting_filters
                * (self.img_size[0] // (2 ** self.upsample_layers))
                * (self.img_size[1] // (2 ** self.upsample_layers)),
                activation="relu",
                input_shape=noise_shape,
            )
        )
        model.add(
            Reshape(
                (
                    (self.img_size[0] // (2 ** self.upsample_layers)),
                    (self.img_size[1] // (2 ** self.upsample_layers)),
                    self.starting_filters,
                )
            )
        )
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 6x8 -> 12x16
        model.add(Conv2D(1024, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 12x16 -> 24x32
        model.add(Conv2D(512, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 24x32 -> 48x64
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 48x64 -> 96x128
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 96x128 -> 192x256
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        """
        Build and return Keras.models.Model for discriminator with the summary indicated below.
        Keep in mind, the discriminator must not be TOO capable. In fact, it is probably better for the training to make a semi-incompetent discrimnator.
        :return: a discriminator model that only requires an image as input.
        """
        img_shape = (self.img_size[0], self.img_size[1], self.channels)

        model = Sequential()
        ###############
        # Conv Stack 1:
        ###############
        model.add(
            Conv2D(128, kernel_size=5, strides=2, input_shape=img_shape, padding="same")
        )  # 128x128 -> 64x64

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.2))

        ###############
        # Conv Stack 2:
        ###############
        model.add(
            Conv2D(128, kernel_size=5, strides=2, padding="same")
        )  # 64x64 -> 32x32
        # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        ###############
        # Conv Stack 3:
        ###############
        model.add(
            Conv2D(128, kernel_size=4, strides=2, padding="same")
        )  # 32x32 -> 16x16

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        ###############
        # Conv Stack 4:
        ###############
        model.add(Conv2D(128, kernel_size=4, strides=1, padding="same"))  # 16x16 -> 8x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        ###############
        # Conv Stack 5:
        ###############
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))  # 8x8 -> 4x4
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))  # important binary classification.

        model.summary()

        # Model require Pair.
        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_gan(self):
        """
        Build a GAN by combining both the DISCRIMINATOR and the GENERATOR.
        :return:
        """

        # Specify te generators used to build various components.
        optimizer_generator = Adam(0.0002, 0.5)
        optimizer_discriminator = Adam(0.0002, 0.5)
        optimizer_GAN = Adam(0.0002, 0.5)

        loss_measure_generator = "binary_crossentropy"
        loss_measure_discriminator = "binary_crossentropy"
        loss_measure_GAN = "binary_crossentropy"

        metrics = ["accuracy", "mae", "mse", "mape", "cosine"]

        # See if the specified model paths exist, if they don't then we start training new models
        if hasattr(self, 'discriminator_path') and hasattr(self, 'generator_path') and\
                self.discriminator_path.is_file() and self.generator_path.is_file():
            self.discriminator = load_model(self.discriminator_path)
            self.generator = load_model(self.generator_path)
            print("Loaded models...")
        else:  # training new model.
            print("Training models...")

            # Generate the tensorboard and its call back
            callback_tensorboard = TensorBoard(
                log_dir=path_log_run, histogram_freq=0, write_images=True
            )

            #self.callbacks_list = [callback_tensorboard]

            # Build discriminator and compile it.
            self.discriminator = self.build_discriminator()

            # Training discriminator!
            self.discriminator.compile(
                loss=loss_measure_discriminator,
                optimizer=optimizer_discriminator,
                #metrics=metrics,
                #callbacks=self.callbacks_list,
            )

            # Build generator and compile it.
            self.generator = self.build_generator()

            # Training generator!
            self.generator.compile(
                loss=loss_measure_generator,
                optimizer=optimizer_generator,
                #callbacks=self.callbacks_list,
            )

        # These next few lines setup the training for the GAN, which the input Vector has a shape of noise_parameters
        z = Input(shape=(self.dimensions_noise,))
        img = self.generator(z)

        self.discriminator.trainable = False

        # Call the discriminator on the image generated by the generator.
        # Store the output
        valid = self.discriminator(img)

        # Form a model that combine both the input and the output pair.
        self.combined = Model(z, valid)

        # Compile the model using binary_crossentropy with the
        self.combined.compile(loss=loss_measure_GAN, optimizer=optimizer_GAN)

    def load_images(self, image_path):
        """
        Load all images from image path as X_train
        :param image_path: path to the folder which contain all images.
        :return: np array of X-train
        """
        X_train = []

        # Load all files from the image path using Image.open.
        for i in recursive_list(image_path):
            # Open images as ???
            img = Image.open(i)
            # Convert to NP array.
            img = np.asarray(img)
            # Append them into higher order array.
            if img.shape == (128,128,3):
                X_train.append(img)

        # return all the images concatenated as a 4D array
        return np.asarray(X_train)

    def train(
        self,
        epochs: int,
        image_path: str,
        batch_size: int = 32,
        save_interval: int = 50,
    ):
        """
        Main entry point
        :param epochs: how any epochs to train
        :param image_path: source of the images
        :param batch_size:
        :param save_interval: how often to save trained
        :return:
        """

        # Build the gan model.
        self.build_gan()

        # Load training data:
        X_train = self.load_images(image_path)

        print("Final training Data Shape: ", X_train.shape)

        # Rescale images from -1 to 1 / Batch normalization of input dimension (this assume 8 bit images?)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        # Floor division to provide the largest possible half batch size in INTEGER
        third_batch = batch_size // 3

        # Logger Info:
        log_path = '../../logs'
        callback = TensorBoard(log_path)
        callback.set_model(self.combined)


        # Loop through the epochs
        for epoch in range(epochs):

            #################
            # Train Generator
            #################
            # randomly initialize the noise_2d input between 0 and 1 for a 2D matrix size of "batch_size" x noise_parameters (the number of noisy input used to generate images)
            noise_2d = np.random.normal(0, 1, (batch_size, self.dimensions_noise))

            # obtain the input as Y
            loss_generator = self.combined.train_on_batch(
                # x: independent variable. Generator takes noise_2d and generate pictures.
                noise_2d,  # x
                # y: since all these images are of the REAL subject, all of the output label are 1s, of the length of batch_size.
                # Technically 2d array, but really is a 1d vector of lengh = batch_size.
                # This assume ALL images being inputted are actually of that appropriate class and nothing else.
                # Super important assumption here.
                # np.ones((batch_size, 1)),
                np.random.rand(batch_size) * 0.4 + 0.8,  # y: a
            )

            #################
            # Train Discriminator
            #################

            # -----------------
            # Mixed 1/3 Image Batch
            # -----------------
            loss_discriminator_on_mixed_data = None
            loss_discriminator_on_fake_data = None
            loss_discriminator_on_real_data = None
            for round in range(self.iteration_generator):
                sixth_batch = third_batch // 2

                # -----------------
                # Fake 1/6 Image Batch
                # -----------------
                # Sample noise_2d and generate "half batch" of new images. Size third_batch x 100
                noise_2d_size = (sixth_batch, self.dimensions_noise)
                noise_2d = np.random.normal(0, 1, noise_2d_size)

                # Now, take this noise_2d and try to generate the image (predicting an IMAGE from the noisy input in the first place.
                images_generated = self.generator.predict(noise_2d)
                soft_labels_fake = np.random.rand(sixth_batch) * 0.3

                # -----------------
                # Real 1/6 Image Batch
                # -----------------

                # Generate random int between 0 and count of X_train (n=?), for 50% of the batch time.
                index = np.random.randint(0, X_train.shape[0], sixth_batch)

                # Store those images.
                images = X_train[index]

                # Train the discriminator (real classified as between 0.8 to 1.2 and generated as 0.zeros): using soft labels result in 0.8 to 1.2 for
                soft_labels_true = np.random.rand(sixth_batch) * 0.4 + 0.8

                # Combine both set.
                a = np.vstack((images_generated, images))
                b = np.hstack((soft_labels_fake, soft_labels_true))

                loss_discriminator_on_mixed_data = self.discriminator.train_on_batch(
                    a,
                    b,  # Y distribution for false images (invalid)
                )

                # -----------------
                # True 1/3 Image Batch
                # -----------------


                # Generate random int between 0 and count of X_train (n=?), for 50% of the batch time.
                index = np.random.randint(0, X_train.shape[0], third_batch)

                # Store those images.
                images = X_train[index]

                # Train the discriminator (real classified as between 0.8 to 1.2 and generated as 0.zeros): using soft labels result in 0.8 to 1.2 for
                soft_labels_true = np.random.rand(third_batch) * 0.4 + 0.8

                loss_discriminator_on_real_data = self.discriminator.train_on_batch(
                    images,
                    soft_labels_true,  # Y distribution for training images (valid)
                )

                # -----------------
                # Fake 1/3 Image Batch
                # -----------------


                # Sample noise_2d and generate "half batch" of new images. Size third_batch x 100
                noise_2d_size = (third_batch, self.dimensions_noise)
                noise_2d = np.random.normal(0, 1, noise_2d_size)

                # Now, take this noise_2d and try to generate the image (predicting an IMAGE from the noisy input in the first place.
                images_generated = self.generator.predict(noise_2d)
                soft_labels_fake = np.random.rand(third_batch) * 0.3

                loss_discriminator_on_fake_data = self.discriminator.train_on_batch(
                    images_generated,
                    soft_labels_fake,  # Y distribution for false images (invalid)
                )

            # -----------------
            # Final Loss
            # -----------------
            # Final loss is a blend of discrimnator on both fake and real data, and combination of them.
            loss_discriminator = (loss_discriminator_on_mixed_data + loss_discriminator_on_real_data + loss_discriminator_on_fake_data)/3

            # Logging the progress (only the final of the ROUNDS!):
            names = ['D loss', 'D mix loss', 'D real loss', 'D fake loss', 'G loss']
            logged_vars = [loss_discriminator, loss_discriminator_on_mixed_data, loss_discriminator_on_real_data, loss_discriminator_on_fake_data, loss_generator]
            self.log_tensorboard(callback, names, logged_vars, epoch)


            # Print progress
            print(
                f"{epoch} [D loss: {loss_discriminator:3f} = Mix: {loss_discriminator_on_mixed_data:3f}, Real {loss_discriminator_on_real_data:3f}, Fake {loss_discriminator_on_fake_data:3f}[G loss: {loss_generator:3f}]"
            )

            # If at save interval => save generated image samples, save model files
            if epoch % (save_interval) == 0:

                # Save the images.
                self.save_imgs(epoch)

                save_path = self.output_directory / 'models'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.discriminator.save((save_path / f'{unique_name()}_discrim.h5').as_posix())
                self.generator.save((save_path / f'{unique_name()}_generat.h5').as_posix())

    def generate_images(self, count):
        """
        This is used to generate a NUMBER of images
        :param count:
        :return:
        """
        # Generate images from the currently loaded model
        noise = np.random.normal(0, 1, (count, self.dimensions_noise))
        return self.generator.predict(noise)

    def save_imgs(self, epoch):
        """
        Save the images from a particular epoch.
        :param epoch:
        :return:
        """
        row, column = 5, 5

        # Generates r*c images from the model, saves them individually and as a gallery
        images_generated = self.generate_images(row * column)

        # ???
        images_generated = 0.5 * images_generated + 0.5

        for index, np_array_image in enumerate(images_generated):
            path = f"{self.output_directory}/generated_{self.img_size[0]}x{self.img_size[1]}"
            if not os.path.exists(path):
                os.makedirs(path)
            imsave(path + f"/{unique_name()}_{epoch}_{index}.png", np_array_image)

        # 4D array:
        nindex, height, width, intensity = images_generated.shape

        nrows = nindex // column

        assert nindex == nrows * column

        # Form the gallery by combining the data at pixel levels (may not be the best approach)
        # want result.shape = (height*n-rows, width*n-cols, intensity)
        gallery = (
            images_generated.reshape(nrows, column, height, width, intensity)
            .swapaxes(1, 2)
            .reshape(height * nrows, width * column, intensity)
        )

        path = f"{self.output_directory}/gallery_generated_{self.img_size[0]}x{self.img_size[1]}"
        if not os.path.exists(path):
            os.makedirs(path)
        imsave(path + f"/{unique_name()}_{epoch}.png", gallery)


    def log_tensorboard(self, callback, names,logs,batch_no):
        """
        Based on the example shown from this thread on how ot write a simple callback function during the train-on-epoch
        https://github.com/eriklindernoren/Keras-GAN/issues/52#issuecomment-402668526
        :param callback: callback object.
        :param names: variable names in list.
        :param logs: log variable.
        :param batch_no: batch number.
        :return:
        """

        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()


    def generate_images_thresholded(self, n_images, threshold, modifier):
        """
        A wrapped version of self.generate_images. Generates (count) images from the model ensuring the discriminator scores them between the threshold values and saves them.

        :param n_images: number of images to generate.
        :param threshold: the threshold which only then only data within that range would be accepted.
        :param modifier:
        :return:
        """
        self.build_gan()

        list_images = []
        for index_current_image in range(n_images):
            # Default score.
            score = [0]
            while not (threshold[0] < score[0] < threshold[1]):
                img = self.generate_images(1)
                score = self.discriminator.predict(img)
            print("Image found: ", score[0])
            list_images.append(img)

        list_images = np.asarray(list_images).squeeze()

        # ???????? Intensity adjustment?
        list_images = 0.5 * list_images + 0.5

        print(list_images.shape)

        # Save all images.
        for index_current_image, np_array_current_image in enumerate(list_images):
            path = f"{self.output_directory}/{unique_name()}_generated_{threshold[0]}_{threshold[1]}"
            if not os.path.exists(path):
                os.makedirs(path)
            imsave(
                path + f"/{modifier}_{index_current_image}.png", np_array_current_image
            )


if __name__ == "__main__":

    # Example Run Script on linux
    # python model/Kaggle_DCGAN_Dogs/DCGAN.py --data /data/resized_128_128:

    # initialize a parser object.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_generator",
        help="Path to existing generator weights file",
        #default="/data/models/generat.h5",
    )
    parser.add_argument(
        "--load_discriminator",
        help="Path to existing discriminator weights file",
        #default="/data/models/discrim.h5",
    )
    parser.add_argument(
        "--data",
        help="Path to directory of images of correct dimensions, using *.[filetype] (e.g. *.png) to reference images",
        default=r"//data/Images/resized/",
    )
    parser.add_argument(
        "--sample",
        help="If given, will generate that many samples from existing model instead of training",
        default=-1,
    )
    parser.add_argument(
        "--sample_thresholds",
        help="The values between which a generated image must score from the discriminator",
        default="(0.0, 0.1)",
    )
    parser.add_argument(
        "--batch_size", help="Number of images to train on at once", default=24
    )
    parser.add_argument(
        "--image_size",
        help="Size of images as tuple (height,width). Height and width must both be divisible by (2^5)",
        default="(128, 128)",
    )
    parser.add_argument(
        "--epochs", help="Number of epochs to train for", default=500000
    )
    parser.add_argument(
        "--save_interval",
        help="How many epochs to go between saves/outputs",
        default=100,
    )
    parser.add_argument(
        "--output_directory",
        help="Directoy to save weights and images to.",
        default=r"/data/doggan/test",
    )

    # Parse the argument.ß
    args = parser.parse_args()

    # Initiate the object.
    dcgan = DCGAN(
        args.load_discriminator,
        args.load_generator,
        args.output_directory,
        literal_eval(args.image_size),
    )

    if args.sample == -1:
        dcgan.train(
            epochs=int(args.epochs),
            image_path=args.data,
            batch_size=int(args.batch_size),
            save_interval=int(args.save_interval),
        )
    else:
        dcgan.generate_images_thresholded(
            int(args.sample), literal_eval(args.sample_thresholds), ""
        )
