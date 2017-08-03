'''
Predict mnist digit image with a mnist trained model

  by Chang Jo Kim
'''


import tensorflow as tf
import my_mnist
import scipy
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## Please install
# scipy, numpy, pillow


def tensorflow_session():

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    session = tf.Session(config=config)

    return session


def restore_checkpoints(session, ckpt_filename):

    #saver = tf.train.import_meta_graph(ckpt_filename + '.meta')
    saver = tf.train.Saver()
    saver.restore(session, ckpt_filename)


def load_image(filename_list):

    x_imgs = np.array([]).reshape(0, 784)
    for filename in filename_list:
        x_img = scipy.misc.imread(filename)

        # Convert 0 <= x <= 255 to 0.0 <= x <= 1.0
        x_img = np.reshape(x_img / 255.0, (1, 784))
        x_imgs = np.concatenate((x_imgs, x_img), axis=0)

    #filename_queue = tf.train.string_input_producer([filename])
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    #x_img = tf.image.decode_png(value, channels=1)
    
    return x_imgs


def start():

    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    y_conv = my_mnist.model(x, keep_prob)
    
    session = tensorflow_session()
    session.run(tf.global_variables_initializer())

    ckpt_filename = 'checkpoints/model.ckpt'
    restore_checkpoints(session, ckpt_filename)

    def predict(image_filename):
        x_imgs = load_image(image_filename)
        logits = session.run(y_conv, feed_dict={x: x_imgs, keep_prob: 1.0})
        predicted_digits = session.run(tf.argmax(logits, 1))

        return predicted_digits

    #print(predict(image_filename))

    #session.run(y_conv, feed_dict={x: mnist.test.images, keep_prob: 1.0})
    #logits = session.run(y_conv, feed_dict={x: mnist.test.images, keep_prob: 1.0})
    
    #print("Predicted digit: ", predicted_digit)
    
    #correct_prediction = tf.equal(predicted_digit, labels)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #accuracy.eval(session=session, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    
    return predict

    
if __name__ == '__main__':

    # image_files = []
    # for i in range(10):
    #     for j in range(1, 11):
    #         image_files.append("test_examples/" + str(i) +"/" + str(j) + ".png")
    
    image_files = ["test_examples/8/1.png", "test_examples/0/1.png", "test_examples/3/1.png"]

    # Start and get a predict function
    predict = start()

    # Perform prediction
    predicted_digits = predict(image_files)

    for i in range(len(predicted_digits)):
        print(image_files[i], "->", predicted_digits[i]) 
