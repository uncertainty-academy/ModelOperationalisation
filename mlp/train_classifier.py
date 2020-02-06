# internal
import dnn_classifier

# 3rd Party
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def train_model_tf(data_set,
                   tf_model,
                   session,
                   num_epochs,
                   batch_size=50,
                   train_only_on_fraction=1.0,
                   optimiser_fn=None,
                   report_every=1,
                   eval_every=1,
                   stop_early=True,
                   verbose=True):
    x, y = tf_model.x, tf_model.y
    loss = tf_model.loss
    accuracy = tf_model.accuracy()

    if optimiser_fn is None:
        optimiser_fn = tf.train.AdamOptimizer()
    optimiser_step = optimiser_fn.minimize(loss)

    init = tf.global_variables_initializer()

    session.run(init)

    train_costs = []
    train_acc = []
    val_costs = []
    val_acc = []

    mnist_train_data = data_set.train

    prev_c_eval = 1000000

    for epoch in range(num_epochs):

        avg_cost = 0.0
        avg_acc = 0.0

        total_batch = int(train_only_on_fraction * mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist_train_data.next_batch(batch_size)

            feed_dict = {x: batch_x, y: batch_y}

            _, c, a = session.run([optimiser_step, loss, accuracy], feed_dict=feed_dict)

            avg_cost += c / total_batch
            avg_acc += a / total_batch

        train_costs.append((epoch, avg_cost))
        train_acc.append((epoch, avg_acc))

        if epoch % report_every == 0 and verbose:
            print 'Epoch:', '%04d' % (epoch + 1), 'Training cost =', '{:.9f}'.format(avg_cost)

        if epoch % eval_every == 0:
            val_x, val_y = mnist.validation.images, mnist.validation.labels

            feed_dict = {x: val_x, y: val_y}

            c_eval, a_eval = session.run([loss, accuracy], feed_dict=feed_dict)

            if verbose:
                print 'Epoch:', '%04d' % (epoch + 1), 'Validation accuracy =', '{:.9f}'.format(a_eval)

            if c_eval > prev_c_eval and stop_early:
                print 'Validation loss stopped improving. Stopping validation after %04d epochs !' % (epoch + 1)
                break

            prev_c_eval = c_eval

            val_costs.append((epoch, c_eval))
            val_acc.append((epoch, a_eval))

    print 'Optimisation finished !'

    test_x, test_y = mnist.test.images, mnist.test.labels

    feed_dict = {x: test_x, y: test_y}

    c_test, a_test = session.run([loss, accuracy], feed_dict=feed_dict)

    print 'Accuracy on test set:', '{:.9f}'.format(a_test)

    return train_costs, train_acc, val_costs, val_acc


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# tf_linear_model = dnn_classifier.LinearSoftMaxClassier(l2_lambda=1.0)
tf_dnn_model = dnn_classifier.DnnSoftMaxClassier(hidden_layers=[512], l2_lambda=1.0)
sess = tf.Session()
trn_costs, trn_accs, val_costs, val_accs = train_model_tf(data_set=mnist,
                                                          tf_model=tf_dnn_model,
                                                          session=sess,
                                                          num_epochs=100,
                                                          batch_size=50,
                                                          train_only_on_fraction=1.0,
                                                          optimiser_fn=tf.train.GradientDescentOptimizer(
                                                              learning_rate=1e-3),
                                                          report_every=1,
                                                          eval_every=2,
                                                          stop_early=True,
                                                          verbose=True)


def my_plot(list_of_tuples):
    plt.plot(*zip(*list_of_tuples))


def plot_multi(values_lst_1, values_lst_2, labels_lst, y_label_1, y_label_2, x_label='epoch'):
    plt.figure(1)
    plt.subplot(211)

    for v in values_lst_1:
        my_plot(v)
    plt.legend(labels_lst, loc='upper_left')

    plt.xlabel(x_label)
    plt.ylabel(y_label_1)

    plt.subplot(212)

    for v in values_lst_2:
        my_plot(v)
    plt.legend(labels_lst, loc='upper_left')

    plt.xlabel(x_label)
    plt.ylabel(y_label_2)

    plt.show()


# Plot losses and accuracies.
plot_multi([trn_costs, val_costs], [trn_accs, val_accs], ['train', 'val'], 'loss', 'accuracy', 'epoch')
# plot_multi([trn_accs, val_accs], ['train', 'val'], 'accuracy', 'epoch')
