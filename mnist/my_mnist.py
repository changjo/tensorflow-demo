'''
<https://www.tensorflow.org/get_started/mnist/pros>

Modified
  by Chang Jo Kim


'''

# coding: utf-8


import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import time, os


TRAIN = False
ADD_IMAGE_SUMMARY = False

if TRAIN:

    #Load MNIST Data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Build a Multilayer Convolutional Network

### Weight initialization
# One should generally initialize weights with a small amount of noise for symmetry breaking,
#   and to prevent 0 gradients.
# Since we're using ReLU neurons,
#   it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


### Convolution and Pooling
# Our convolutions uses a stride of one and are zero padded
#   so that the output is the same size as the input.
# Our pooling is plain old max pooling over 2x2 blocks.

# An input tensor: [batch, in_height, in_width, in_channels]
# A filter / kernel tensor: [filter_height, filter_width, in_channels, out_channels]
# strides: [batch, height, width, channels] for data_format="NHWC"
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)


def h_image_concat(h):
    
    num = h.shape[-1]
    h_list = []
    for i in range(num):
        #h_conv1_concat = tf.concat([h_conv1_concat, h_conv1[:, :, :, i]], 1)
        h_list.append(h[:, :, :, i])
    h_concat = tf.concat(h_list, 1)
    
    return tf.expand_dims(h_concat, 3)



def model(x, keep_prob):    

    # Placeholders

    ### First Convolutional Layer
    # Patch size: 5x5, 32 features
    with tf.name_scope('conv_1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='weights')   # [filter height, filter width, # input channels, # output channels]
        b_conv1 = bias_variable([32], name='biases')

        #print x.shape
        #  (?, 784)
        # To apply the layer, we first reshape x to a 4d tensor.
        x_image = tf.reshape(x, [-1, 28, 28, 1])   # [batch, image width, image height, # color channels]

        #print x_image.shape
        #  (?, 28, 28, 1)
        #print conv2d(x_image, W_conv1).shape
        #  (?, 28, 28, 32)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)   
        
        if ADD_IMAGE_SUMMARY:
            h_conv1_image = h_image_concat(h_conv1)
            tf.summary.image('h_conv1', h_conv1_image, 10)
        
        
    with tf.name_scope('pool_1'):
        h_pool1 = max_pool_2x2(h_conv1, name='max_pool')            # This will reduce the image size to 14x14
        #print h_pool1.shape
        #  (?, 14, 14, 32)
        
        if ADD_IMAGE_SUMMARY:
            h_pool1_image = h_image_concat(h_pool1)
            tf.summary.image('h_pool1', h_pool1_image, 10)


    ### Second Convolutional Layer
    # Patch size: 5x5, 64 features
    with tf.name_scope('conv_2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='weights')
        b_conv2 = bias_variable([64], name='biases')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        
        
        if ADD_IMAGE_SUMMARY:
            h_conv2_image = h_image_concat(h_conv2)
            tf.summary.image('h_conv2', h_conv2_image, 10)

        
    with tf.name_scope('pool_2'):    
        h_pool2 = max_pool_2x2(h_conv2)            # This will reduce the image size to 7x7
        #print h_conv2.shape
        #  (?, 14, 14, 64)
        #print h_pool2.shape
        #  (?, 7, 7, 64)
        #h_pool2_image = tf.reshape(h_pool2, [-1, 7*64, 7, 1])
        
        if ADD_IMAGE_SUMMARY:
            h_pool2_image = h_image_concat(h_pool2)
            tf.summary.image('h_pool2', h_pool2_image, 10)

    ### Densely Connected Layer
    # We add a fully-connected layer with 1024 neurons to allow processing on the entire image
    with tf.name_scope('full_connected'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='weights')
        b_fc1 = bias_variable([1024], name='biases')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #print h_pool2_flat.shape
    #  (?, 3136)
    #print h_fc1.shape
    #  (?, 1024)

    # Dropout
    # To reduce overfitting, we will apply dropout before the readout layer.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #sess.run(tf.global_variables_initializer())
    #batch = mnist.train.next_batch(50)
    #print sess.run(h_fc1_drop, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}).shape
    #print sess.run(h_pool2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}).shape

    # Readout Layer
    # Softmax
    with tf.name_scope('softmax'):
        W_fc2 = weight_variable([1024, 10], name='weights')
        b_fc2 = bias_variable([10], name='biases')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.identity(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')


    return y_conv    



def loss_fn(labels, logits):

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    tf.summary.scalar('loss', loss)

    return loss


def train(loss):

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

    return train_step


def eval(labels, logits):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
    tf.summary.scalar('accuracy', accuracy)
    
    return accuracy



def run():

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = model(x, keep_prob)

    loss = loss_fn(y_, y_conv)
    accuracy = eval(y_, y_conv)
    summary = tf.summary.merge_all()

    train_step = train(loss)

    saver = tf.train.Saver()

    log_dir = '/tmp/tensorflow/mnist/logs/deep_mnist_for_experts'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)


    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(20000):

            start_time = time.time()

            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            duration = time.time() - start_time


            if step % 100 == 0:

                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' %(step, train_accuracy))

                summary_str = sess.run(summary, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()


            if (step + 1) % 1000 == 0 or (step + 1) == 20000:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)


        print('test accuracy %g' %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    run()

