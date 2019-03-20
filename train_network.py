import os
import tensorflow as tf
import time

from dense_net import DenseNet
from cifar10_dataset import Cifar10Dataset

class TrainNetwork:
    '''
        Handles the training of a neural network. For now this will only work with a DENSENET from dense_net.py
    '''

    def __init__(self, theta=1.0, growth_rate=12, depth=40, weight_decay=1e-4,use_bottleneck=False, batch_size=64):

        self.theta = theta
        self.growth_rate = growth_rate
        self.depth = depth
        self.weight_decay = weight_decay
        self.use_bottleneck = use_bottleneck
        self.data_augmentation = True

        self.batch_size = batch_size

        self.data = Cifar10Dataset(batch_size=self.batch_size, augment=self.data_augmentation)
        self.dense_net = DenseNet(self.data.iterator, theta, growth_rate, depth, weight_decay, use_bottleneck)

        self.train_step = self.dense_net.train_step
        self.summary = self.dense_net.summary
        self.accuracy = self.dense_net.accuracy

        self.subdir = 'DN{}_GR_{}_DA_{}_Theta_{:.2f}_Bottle_{}_Batch_size_{}'.format(depth, growth_rate, self.data_augmentation, theta, use_bottleneck, self.batch_size)


    def train(self):
        ''' Handles the training of a neural network
        Requirments:
            - Log summaries for tensorboard
            - Schedule learning rate
            - save models if desirable

            switch between training and test data for logs

            - Able to train from scratch or import existing model

        '''

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, save_relative_paths=True)
        model_path = './models/{}/'.format(self.subdir)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            train_writer = tf.summary.FileWriter(logdir='./logs/{}/Train/'.format(self.subdir), graph=sess.graph)
            test_writer = tf.summary.FileWriter(logdir='./logs/{}/Test/'.format(self.subdir), graph=sess.graph)

            # Training parameters
            batches_per_epoch = self.data.num_train // self.data.batch_size
            test_batches = self.data.num_test // self.data.batch_size

            lr = 0.1

            # Initial file save
            saver.save(sess, save_path=model_path + 'model', global_step=0, write_meta_graph=True)

            for epoch in range(1, 301):
                if not epoch % 10:
                    print('Model saved')
                    saver.save(sess, save_path=model_path+'model', global_step=epoch, write_meta_graph=False)

                # Initialize training data into iterator
                sess.run(self.data.training_init_op)
                sess.run(tf.local_variables_initializer())

                # Update learning rate, Learning rate scheduling should be moved to another function
                if epoch == 150:
                    lr = lr/10
                if epoch == 225:
                    lr = lr/10

                ta = time.time()
                for batch in range(batches_per_epoch):
                    # Training
                    feed_dict = {'lr:0': lr, 'is_training:0': True}

                    sess.run(self.train_step, feed_dict=feed_dict)

                    if not batch % 100:
                        tr_summary = sess.run(self.summary, feed_dict=feed_dict)
                        train_writer.add_summary(tr_summary, batch + (epoch-1)*batches_per_epoch)

                        print('Epoch: {}; Batch: {}/{}, Batches per sec: {:.2f}'.format(epoch, batch, batches_per_epoch,
                                                                                        100.0/(time.time()-ta)))
                        ta = time.time()
                else:
                    print('Initialized Testing')
                    sess.run(self.data.testing_init_op)
                    sess.run(tf.local_variables_initializer())

                    feed_dict = {'lr:0': 0.0, 'is_training:0': False}

                    for batch in range(test_batches):
                        sess.run(self.accuracy, feed_dict=feed_dict)
                    else:
                        test_summary = sess.run(self.summary, feed_dict=feed_dict)
                        test_writer.add_summary(test_summary, epoch*batches_per_epoch)

if __name__=='__main__':
    tn = TrainNetwork(theta=1.0, growth_rate=12, depth=40, weight_decay=1e-4, use_bottleneck=False, batch_size=32)
    tn.train()

