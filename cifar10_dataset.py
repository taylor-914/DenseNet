import pickle
import os
import numpy as np
import tensorflow as tf

class Cifar10Dataset:

    # Each image is 32 by 32 pixels with 3 color channels
    img_size = 32

    # The images have 3 colour channels
    img_channels = 3

    # The images are stored in one dimensional arrays of this lengeth
    img_size_flat = img_size * img_size * img_channels

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # Tuple with height, width and depth used to reshape arrays.
    # This is used for reshaping in Keras.
    img_shape_full = (img_size, img_size, img_channels)

    # There are 10 classes in cifar10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    num_classes = 10

    # list of classes in the order corresponding to their numerical category in the labeled data
    class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Class dictionary allows for the easy lookup of a class label based on the index
    class_dict = {ind: category for ind, category in enumerate(class_list)}

    def __init__(self, data_dir='./cifar-10-batches-py', batch_size=64, augment=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment

        self.num_train = 50000
        self.num_test = 10000

        # Images will be normalized based on the channel mean and std
        self.channel_mean = None
        self.channel_std = None

        # Load data into self.x_train, self.y_train, self.x_test, self.y_test
        self._load_data()

        #with tf.device('/cpu:0'):
        # fill self.train_ds,
        self.train_ds, self.test_ds = self._construct_datasets()

        self.iterator = tf.data.Iterator.from_structure(self.train_ds.output_types, self.train_ds.output_shapes)

        self.training_init_op = self.iterator.make_initializer(self.train_ds)
        self.testing_init_op = self.iterator.make_initializer(self.test_ds)


    def _load_data(self):
        '''
        loads the data from a folder, will load all cifar data

        :return:
        '''

        def _unpickle(file):
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            return data_dict

        # Load training data
        data = []
        labels = []

        for i in range(1, 6):
            file_path = os.path.join(self.data_dir, 'data_batch_{}'.format(i))
            data_dict = _unpickle(file_path)

            data.append(data_dict[b'data'])
            labels.append(data_dict[b'labels'])
        # will use to demean and normalize data

        self.x_train = self._reshape_data(np.vstack(data)).astype(np.float32)
        self.y_train = np.hstack(labels).astype(np.int)

        # Load testing Data
        file_path = os.path.join(self.data_dir, 'test_batch')
        test_dict = _unpickle(file_path)

        self.x_test = self._reshape_data(np.array(test_dict[b'data'])).astype(np.float32)
        self.y_test = np.hstack(test_dict[b'labels']).astype(np.int)

        # Normalize data based on the channel means
        self.channel_mean = []
        self.channel_std = []

        for i in range(3):
            self.channel_mean.append(np.mean(self.x_train[:,:,:,i]))
            self.channel_std.append(np.std(self.x_train[:,:,:,i]))

            self.x_test[:, :, :, i] = (self.x_test[:, :, :, i] - self.channel_mean[i])/self.channel_std[i]
            self.x_train[:, :, :, i] = (self.x_train[:, :, :, i] - self.channel_mean[i])/self.channel_std[i]

    def _reshape_data(self, data):
        ''' Reshapes data into images that can be trained on

        Args:
            data (numpy array): Each row is an image, each column is a pixel value for one channel

        Return: data reshaped into images (channel last)
        '''

        # The cifar images are loaded in a channel first format
        data = data.reshape((-1, 3, 32, 32))
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 1, 2)

        return data

    def _construct_datasets(self):
        ''' Constructs the training and test dataset based on the data in numpy arrays

        '''

        def apply_settings(dataset, img_num):
            dataset = dataset.shuffle(buffer_size=img_num)
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size).prefetch(2)
            return dataset

        def zipped_dataset(images, labels):
            img_ds = tf.data.Dataset.from_tensor_slices(images)
            label_ds = tf.data.Dataset.from_tensor_slices(labels).map(lambda x: tf.one_hot(x, depth=10))
            return tf.data.Dataset.zip((img_ds, label_ds))

        # Construct training dataset
        train_ds = zipped_dataset(self.x_train, self.y_train)
        test_ds = zipped_dataset(self.x_test, self.y_test)

        # Apply settings to data sets
        train_ds = apply_settings(train_ds, img_num=self.num_train)
        test_ds = apply_settings(test_ds, img_num=self.num_test)

        # Augment training dataset
        if self.augment:
            train_ds = self._apply_augmentation(train_ds)

        return train_ds, test_ds

    def _apply_augmentation(self, dataset):
        ''' Applies data augmentation to the dataset passed in

        Args:
            dataset (tf.data.Dataset): Tensorflow Dataset to be augmented

        Return: dataset with augmentations mapped
        '''

        dataset = dataset.map(self._horizontal_flip, num_parallel_calls=2)
        #dataset = dataset.map(self._random_resize_and_crop, num_parallel_calls=2)
        dataset = dataset.map(self._image_modify, num_parallel_calls=2)

        # research other common augmentations
        return dataset

    def _image_modify(self, image, label):
        img = tf.image.random_brightness(image, max_delta=0.3)
        img = tf.image.resize_image_with_crop_or_pad(img, 36, 36)

        img = tf.image.random_crop(img, size=[self.batch_size, 32, 32, 3])
        return img, label

    def _resize_and_crop(self, image):
        img = tf.image.resize_images(image, size=[36, 36])

        boxes = [[x1, y1, x1+31, y1+31] for x1 in range(5) for y1 in range(5)]

        img = tf.image.random_crop(img, [32, 32, 3])
        return img

    def _random_resize_and_crop(self, image, label):
        random_val = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
        cond_value = tf.math.less(random_val, 0.5)
        boxes = [[y1, x1, y1+27, x1+27] for y1 in range(5) for x1 in range(5) for zoom in range(5)]
        img = tf.cond(cond_value, lambda: image, lambda: tf.images.crop_and_resize(image, boxes=boxes, box_ind_=0, crop_size=[32, 32]))
        return img, label


    def _horizontal_flip(self, image, label):
        return tf.image.random_flip_left_right(image), label

if __name__=='__main__':
    from matplotlib import pyplot as plt

    ci = Cifar10Dataset(augment=True)

    with tf.Session() as sess:

        sess.run(ci.training_init_op)

        for i in range(5):
            img, label = sess.run(ci.iterator.get_next())

            print(np.max(img))
            print(np.min(img))
            print(img.shape)
            print(label.shape)


            for i in range(1):
                plt.imshow((img[i, :, :, :]+2.5)/5.0)
                plt.show()



