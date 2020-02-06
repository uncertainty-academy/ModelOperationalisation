# 3rd Party
import tensorflow as tf


class BaseSoftmaxClassifier(object):
    def __init__(self, input_size, output_size, l2_lambda):
        self.input_size = input_size
        self.output_size = output_size
        self.reg_lambda = l2_lambda

        self.x = tf.placeholder(tf.float64, [None, input_size])
        self.y = tf.placeholder(tf.float64, [None, output_size])

        self._all_weights = []

        self.predictions = None
        self.loss = None

        self.build_model()

    def dense_linear_layer(self, inputs, layer_name, input_size, output_size):
        with tf.variable_scope(layer_name, reuse=False):
            layer_weights = tf.get_variable("weights",
                                            shape=[input_size, output_size],
                                            dtype=tf.float64,
                                            initializer=tf.random_normal_initializer())

            layer_biases = tf.get_variable("biases",
                                           shape=[output_size],
                                           dtype=tf.float64,
                                           initializer=tf.random_normal_initializer())

            logits = tf.add(tf.matmul(inputs, layer_weights), layer_biases)

        return logits, layer_weights

    def build_hidden_layers(self, inputs, hidden_layers, inputs_size):
        input_size = inputs_size
        for layer, units in enumerate(hidden_layers):
            layer_name = 'hidden_layer_' + str(layer + 1)
            output_size = units
            with tf.variable_scope(layer_name, reuse=False):
                layer_weights = tf.get_variable("weights",
                                                shape=[input_size, output_size],
                                                dtype=tf.float64,
                                                initializer=tf.random_normal_initializer())

                layer_biases = tf.get_variable("biases",
                                               shape=[output_size],
                                               dtype=tf.float64,
                                               initializer=tf.random_normal_initializer())

                logits = tf.add(tf.matmul(inputs, layer_weights), layer_biases)
                layer_output = tf.nn.relu(logits)

            inputs = layer_output
            input_size = units

        return layer_output, input_size

    def build_model(self):
        raise NotImplementedError('Subclass should implement this function')

    def compute_loss(self, logits):
        data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        reg_loss = 0.
        for w in self._all_weights:
            reg_loss += tf.nn.l2_loss(w) + self.reg_lambda

        return data_loss + reg_loss

    def accuracy(self):
        assert self.predictions is not None
        correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

        return accuracy


class LinearSoftMaxClassier(BaseSoftmaxClassifier):
    def __init__(self, input_size=784, output_size=10, l2_lambda=0.):
        super(LinearSoftMaxClassier, self).__init__(input_size, output_size, l2_lambda)

    def build_model(self):
        logits, self.weights = self.dense_linear_layer(self.x, 'linear_layer', self.input_size, self.output_size)

        self._all_weights.append(self.weights)

        self.predictions = tf.nn.softmax(logits)

        self.loss = self.compute_loss(logits)


class DnnSoftMaxClassier(BaseSoftmaxClassifier):
    def __init__(self, hidden_layers=None, input_size=784, output_size=10, l2_lambda=0.):
        self.hidden_layers = hidden_layers
        super(DnnSoftMaxClassier, self).__init__(input_size, output_size, l2_lambda)

    def build_model(self):
        layer_output, hidden_layers_out_size = self.build_hidden_layers(self.x, self.hidden_layers, self.input_size)

        logits, self.weights = self.dense_linear_layer(layer_output, 'linear_layer', hidden_layers_out_size,
                                                       self.output_size)

        self._all_weights.append(self.weights)

        self.predictions = tf.nn.softmax(logits)

        self.loss = self.compute_loss(logits)
