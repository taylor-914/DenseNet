
import tensorflow as tf

class DenseNet:
    ''' Primary job is to build the DenseNet graph

        This implimentation assumes there are 3 blocks, each with an equal number of layers

        Args:
            iterator (tf.data.Iterator): Data feed for the neural network
            theta (float): Compression factor applied in the
            growth_rate (int): The number of new feature maps added by each convolution inside a block
            depth (int): The number of trainable layers in the network
            weight_decay (float): The weight decay factor applied to the L2 regularization cost
            use_bottleneck (bool): If true, each layer in a block is composed of conv(1x1) followed by conv(3x3)
            use_compression (bool): If False, no compression is applied between layers, only the average down sampling

    '''

    def __init__(self, iterator,  theta=0.5, growth_rate=12, depth=100, weight_decay=1e-4, use_bottleneck=True):
        assert isinstance(iterator, tf.data.Iterator), 'iterator must be a tensorflow.data.Iterator'
        assert 0 < theta <= 1.0, 'theta must be larger than zero and less than or equal to 1.0: Recieved {}'.format(theta)
        assert isinstance(growth_rate, int), 'growth_rate must be an integer: Recieved {}'.format(type(growth_rate))
        assert isinstance(depth, int), 'depth must be an integer: Recieved {}'.format(type(depth))
        assert isinstance(weight_decay, float), 'weight_decay must be an float: Recieved {}'.format(type(weight_decay))
        assert isinstance(use_bottleneck, bool), 'use_bottleneck must be an boolean: Recieved {}'.format(type(use_bottleneck))

        self.iterator = iterator
        self.theta = theta
        self.growth_rate = growth_rate
        self.depth = depth
        self.weight_decay = weight_decay
        self.use_bottleneck = use_bottleneck

        layers_per_block = (depth-4)//3
        self.layers_per_block = layers_per_block // 2 if self.use_bottleneck else layers_per_block

        self.num_layers = 1
        self.layer_activations = []

        self.build_graph()

    def get_next(self):
        return self.iterator.get_next()

    def build_graph(self):

        lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr') # Learning rate
        is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training') # Tells Batchnorm how to modify the batch

        img, y_true = self.get_next()

        with tf.variable_scope('Preconvolution'):
            num_filters = 2 * self.growth_rate if self.use_bottleneck else 16
            cp = self._conv(img, num_filters, kernel_size=3, preActivation=False, batch_norm=False, is_training=is_training)

        b1 = self._add_block(input=cp, is_training=is_training, block_num=1)
        t1 = self._transition_layer(input=b1, is_training=is_training, transition_num=1)
        b2 = self._add_block(input=t1, is_training=is_training, block_num=2)
        t2 = self._transition_layer(input=b2, is_training=is_training, transition_num=2)
        b3 = self._add_block(input=t2, is_training=is_training, block_num=3)

        if True:
            bn = tf.layers.batch_normalization(b3, training=is_training)
            relu = tf.nn.relu(bn)
            b3 = relu

        gap = self.global_average(b3)

        logits = self.dense(input=gap, is_training=is_training)

        y_score = tf.nn.softmax(logits, axis=1)

        # Add Metrics
        with tf.variable_scope('Metrics'):
            acc, acc_update_op = tf.metrics.accuracy(labels=tf.argmax(y_true, axis=1),
                                                     predictions=tf.argmax(y_score, axis=1))
            self.accuracy = tf.group([acc, acc_update_op])

        with tf.variable_scope('Cost'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
            cost = tf.reduce_mean(cross_entropy)

            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'gamma' not in var.name and 'beta' not in var.name])
            total_cost = cost + self.weight_decay * l2

        with tf.variable_scope('Optimizer_Setup'):

            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, use_nesterov=True, momentum=0.9)
            train = optimizer.minimize(total_cost)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.train_step = tf.group([train, update_ops, acc_update_op])

        with tf.variable_scope('Summary'):
            # Add Histograms of all trainable variables
            for var in tf.trainable_variables():
                if 'kernel' in var.name:
                    tf.summary.histogram(var.name, var)

            # Add histograms of layer activations
            for layer_act in self.layer_activations:
                tf.summary.histogram(layer_act.name, layer_act)

            # add logits distrobution
            tf.summary.histogram('logits', logits)
            tf.summary.histogram('GAP', gap)
            tf.summary.histogram('Images', img)

            # Add training image samples
            tf.summary.image('Training_Images', img)

            # Add scalar metrics
            tf.summary.scalar('Accuracy', acc)
            tf.summary.scalar('Error_Rate', 1.0-acc)
            tf.summary.scalar('Cross_entropy_cost', cost)
            tf.summary.scalar('Total_cost', total_cost)
            tf.summary.scalar('l2_loss', total_cost-cost)

            self.summary = tf.summary.merge_all()


    def dense(self, input, is_training):
        ''' Dense classifier on the output of our NN
            No batch normalization Allowed!!

        '''
        with tf.variable_scope('Classification'):
            init = tf.contrib.layers.xavier_initializer()
            input_dim = input.get_shape().as_list()[-1]

            #bn = tf.layers.batch_normalization(input, training=is_training)

            weights = tf.get_variable(name='weights', shape=[input_dim, 10], dtype=tf.float32, initializer=init)
            bias = tf.get_variable(name='bias', shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())

            return tf.matmul(input, weights) + bias

    def global_average(self, input):
        '''bn  -> avg_pooling entire feature
        Args:
            input (tf.Tensor): tensorflow tensor

        Return (tf.Tensor): flattened tensorflow tensor
        '''
        with tf.variable_scope('Global_Average_Pooling'):
            feature_size = input.get_shape()[1]
            k_shape = [1, feature_size, feature_size, 1]
            gap = tf.nn.avg_pool(value=input, ksize=k_shape, strides=k_shape, padding='VALID', name='GAP')
            flat = tf.layers.flatten(gap)

            self.layer_activations.append(flat)

        return flat

    def _add_block(self, input, is_training, block_num):
        ''' Adds all the layers in a block
        Args:
            input (tf.Tensor): tensorflow tensor

        Return (tf.Tensor): tensorflow tensor
        '''
        c = input
        with tf.variable_scope('Block_{}'.format(block_num)):
            for l in range(self.layers_per_block):
                with tf.variable_scope('Layer_{}'.format(l)):
                    if self.use_bottleneck:
                        c_out = self._bottleneck_layer(c, is_training=is_training)
                    else:
                        c_out = self._conv(c, self.growth_rate, 3, is_training=is_training)

                    c = tf.concat([c_out, c], axis=3)

        return c

    def _transition_layer(self, input, is_training, transition_num):
        ''' bn -> relu -> conv2d(1,1) -> avg_pooling(2,2)
        Args:
            input (tf.Tensor): tensorflow tensor

        Return (tf.Tensor): tensorflow tensor
        '''
        with tf.variable_scope('Transition_{}'.format(transition_num)):
            # m is the number of feature maps from the previous block
            m = input.get_shape().as_list()[-1]
            num_filters = int(self.theta * m)
            c = self._conv(input, filters=num_filters, kernel_size=1, is_training=is_training, preActivation=False)

            c_out = tf.nn.avg_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        return c_out

    def _bottleneck_layer(self, input, is_training):
        ''' bn -> relu -> conv2d(1,1) -> dropout -> bn -> relu -> conv2d(3,3) -> dropout
        Args:
            input (tf.Tensor): tensorflow tensor

        Return (tf.Tensor): tf.concat([input, new_conv_features], axis=3)
        '''

        c_bn = self._conv(input, filters=4*self.growth_rate, kernel_size=1, is_training=is_training)
        c = self._conv(c_bn, filters=self.growth_rate, kernel_size=3, is_training=is_training)

        return c

    def _conv(self, input, filters, kernel_size, is_training,
              preActivation=True, batch_norm=True):
        ''' All convolutions are a batch_normalization -> relu -> convolution_2d -> dropout
        Args:
            input (tf.Tensor): the input tensor
            filters (int): the number of unique filters in this conv
            kernel_size (int): the height and width of the kernel

        Return (tf.Tensor): tensorflow tensor
        '''
        # keep track of the number of conv layers in graph
        self.num_layers += 1

        bn = tf.layers.batch_normalization(input, scale=True, training=is_training) if batch_norm else input

        relu = tf.nn.relu(bn) if preActivation else bn

        c = tf.layers.conv2d(relu, filters=filters, kernel_size=kernel_size, padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             use_bias=False, activation=tf.identity)

        self.layer_activations.append(c)

        return c

    def count_trainable_parameters(self):
        total_parameters = 0

        for var in tf.trainable_variables():
            #print(var.name, var.get_shape())

            shape = var.get_shape()
            var_parameters = 1
            for dim in shape:
                var_parameters *= dim

            total_parameters += var_parameters

        print('There are {:.3f} million trainable parameters'.format(int(total_parameters)/1e6))


if __name__=='__main__':
    from cifar10_dataset import Cifar10Dataset
    ci = Cifar10Dataset()

    dn = DenseNet(ci.iterator, theta=1.0, growth_rate=12, depth=40, weight_decay=1e-4, use_bottleneck=False)

    dn.count_trainable_parameters()





