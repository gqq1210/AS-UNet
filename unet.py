'''

'''
from uNet.layer import (conv2d, deconv2d, max_pooling_2x2, crop_and_concat, weight_xavier_init, bias_variable)
import tensorflow as tf
import numpy as np
import os
import cv2


def deconv_relu(x, kernal, scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal, n_inputs = kernal[0] * kernal[1] * kernal[-1], n_outputs = kernal[-2],
                               activefunction = 'relu', variable_name = scope + 'W')
        B = bias_variable([kernal[-2]], variable_name = scope + 'B')
        deconv = deconv2d(x, W) + B
        deconv = tf.nn.relu(deconv)
        return deconv


def normalizationlayer(x, is_train, norm_type = None, scope = None):
    with tf.name_scope(scope + norm_type):
        if norm_type == None:
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(x, center = True, scale = True, is_training = is_train)
        return output

def conv_bn_relu_drop(x, kernal, phase, drop, scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal, n_inputs = kernal[0] * kernal[1] * kernal[2], n_outputs = kernal[-1],
                               activefunction = 'relu', variable_name = scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name = scope + 'conv_B')
        conv = conv2d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, norm_type='batch', scope = scope + 'normalization')
        conv = tf.nn.dropout(tf.nn.relu(conv), drop, name = scope + 'conv_dropout')
        return conv

def _create_conv_net(X, image_width, image_height, image_channel, phase, drop_conv, n_class=1):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # UNet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x = inputX, kernal = [3, 3, image_channel, 32], phase= phase, drop = drop_conv, scope = 'layer0')
    layer1 = conv_bn_relu_drop(x = layer0, kernal= [3, 3, 32, 32], phase = phase, drop=drop_conv, scope = 'layer1')
    # print(layer1.get_shape().as_list())

    """
    W1_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32)
    B1_2 = bias_variable([32])
    conv1_2 = conv2d(conv1_1, W1_2) + B1_2
    # conv1_2 = tf.contrib.layers.batch_norm(conv1_2, epsilon=1e-5, scope='bn2')
    conv1_2 = tf.contrib.layers.batch_norm(conv1_2, center=True, scale=True, is_training=phase, scope='bn2')
    conv1_2 = tf.nn.dropout(tf.nn.relu(conv1_2), drop_conv)
    """

    pool1 = max_pooling_2x2(layer1)
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x = pool1, kernal = [3, 3, 32, 64], phase= phase, drop = drop_conv, scope= 'layer2_1')
    """
    W2_1 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 32, n_outputs=64)
    B2_1 = bias_variable([64])
    conv2_1 = conv2d(pool1, W2_1) + B2_1
    # conv2_1 = tf.contrib.layers.batch_norm(conv2_1, epsilon=1e-5, scope='bn3')
    conv2_1 = tf.contrib.layers.batch_norm(conv2_1, center=True, scale=True, is_training=phase, scope='bn3')
    conv2_1 = tf.nn.dropout(tf.nn.relu(conv2_1), drop_conv)
    """

    layer2 = conv_bn_relu_drop(x = layer2, kernal=[3, 3, 64, 64], phase = phase, drop = drop_conv, scope = 'layer2_2')
    # print(layer2.get_shape().as_list())
    """
    W2_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64)
    B2_2 = bias_variable([64])
    conv2_2 = conv2d(conv2_1, W2_2) + B2_2
    # conv2_2 = tf.contrib.layers.batch_norm(conv2_2, epsilon=1e-5, scope='bn4')
    conv2_2 = tf.contrib.layers.batch_norm(conv2_2, center=True, scale=True, is_training=phase, scope='bn4')
    conv2_2 = tf.nn.dropout(tf.nn.relu(conv2_2), drop_conv)
    """
    pool2 = max_pooling_2x2(layer2)

    # layer3->convolution
    layer3 = conv_bn_relu_drop(x = pool2, kernal= [3, 3, 64, 128], phase = phase, drop=drop_conv, scope = 'layer3_1')
    """
    W3_1 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 64, n_outputs=128)
    B3_1 = bias_variable([128])
    conv3_1 = conv2d(pool2, W3_1) + B3_1
    # conv3_1 = tf.contrib.layers.batch_norm(conv3_1, epsilon=1e-5, scope='bn5')
    conv3_1 = tf.contrib.layers.batch_norm(conv3_1, center=True, scale=True, is_training=phase, scope='bn5')
    conv3_1 = tf.nn.dropout(tf.nn.relu(conv3_1), drop_conv)
    """
    layer3 = conv_bn_relu_drop(x = layer3, kernal= [3, 3, 128, 128], phase = phase, drop=drop_conv, scope = 'layer3_2')
    """
    W3_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128)
    B3_2 = bias_variable([128])
    conv3_2 = conv2d(conv3_1, W3_2) + B3_2
    # conv3_2 = tf.contrib.layers.batch_norm(conv3_2, epsilon=1e-5, scope='bn6')
    conv3_2 = tf.contrib.layers.batch_norm(conv3_2, center=True, scale=True, is_training=phase, scope='bn6')
    conv3_2 = tf.nn.dropout(tf.nn.relu(conv3_2), drop_conv)
    """
    pool3 = max_pooling_2x2(layer3)

    # layer4->convolution
    layer4 = conv_bn_relu_drop(x = pool3, kernal= [3, 3, 128, 256], phase=phase, drop=drop_conv, scope = 'layer4_1')
    """
    W4_1 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 128, n_outputs=256)
    B4_1 = bias_variable([256])
    conv4_1 = conv2d(pool3, W4_1) + B4_1
    # conv4_1 = tf.contrib.layers.batch_norm(conv4_1, epsilon=1e-5, scope='bn7')
    conv4_1 = tf.contrib.layers.batch_norm(conv4_1, center=True, scale=True, is_training=phase, scope='bn7')
    conv4_1 = tf.nn.dropout(tf.nn.relu(conv4_1), drop_conv)
    """
    layer4 = conv_bn_relu_drop(x = layer4, kernal=[3, 3, 256, 256], phase=phase, drop=drop_conv, scope= 'layer4_2')
    """
    W4_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256)
    B4_2 = bias_variable([256])
    conv4_2 = conv2d(conv4_1, W4_2) + B4_2
    # conv4_2 = tf.contrib.layers.batch_norm(conv4_2, epsilon=1e-5, scope='bn8')
    conv4_2 = tf.contrib.layers.batch_norm(conv4_2, center=True, scale=True, is_training=phase, scope='bn8')
    conv4_2 = tf.nn.dropout(tf.nn.relu(conv4_2), drop_conv)
    """
    pool4 = max_pooling_2x2(layer4)

    # layer5->convolution
    layer5 = conv_bn_relu_drop(x = pool4, kernal= [3, 3, 256, 512], phase=phase, drop=drop_conv, scope='layer5_1')
    """
    W5_1 = weight_xavier_init(shape=[3, 3, 256, 512], n_inputs=3 * 3 * 256, n_outputs=512)
    B5_1 = bias_variable([512])
    conv5_1 = conv2d(pool4, W5_1) + B5_1
    # conv5_1 = tf.contrib.layers.batch_norm(conv5_1, epsilon=1e-5, scope='bn9')
    conv5_1 = tf.contrib.layers.batch_norm(conv5_1, center=True, scale=True, is_training=phase, scope='bn9')
    conv5_1 = tf.nn.dropout(tf.nn.relu(conv5_1), drop_conv)
    """
    layer5 = conv_bn_relu_drop(x = layer5, kernal=[3, 3, 512, 512], phase=phase, drop = drop_conv, scope='layer5_2')
    """
    W5_2 = weight_xavier_init(shape=[3, 3, 512, 512], n_inputs=3 * 3 * 512, n_outputs=512)
    B5_2 = bias_variable([512])
    conv5_2 = conv2d(conv5_1, W5_2) + B5_2
    # conv5_2 = tf.contrib.layers.batch_norm(conv5_2, epsilon=1e-5, scope='bn10')
    conv5_2 = tf.contrib.layers.batch_norm(conv5_2, center=True, scale=True, is_training=phase, scope='bn10')
    conv5_2 = tf.nn.dropout(tf.nn.relu(conv5_2), drop_conv)
    """

    # layer6->deconvolution

    deconv1 = deconv_relu(x = layer5, kernal=[3, 3, 256, 512], scope = 'deconv1')
    """
    W6 = weight_xavier_init(shape=[3, 3, 256, 512], n_inputs=3 * 3 * 512, n_outputs=256)
    B6 = bias_variable([256])
    dconv1 = tf.nn.relu(deconv2d(conv5_2, W6) + B6)
    """
    layer6 = crop_and_concat(layer4, deconv1)


    # layer7->convolution
    layer6 = conv_bn_relu_drop(x = layer6, kernal= [3, 3, 512, 256], phase=phase, drop=drop_conv, scope = 'layer6_1')
    """
    W7_1 = weight_xavier_init(shape=[3, 3, 512, 256], n_inputs=3 * 3 * 512, n_outputs=256)
    B7_1 = bias_variable([256])
    conv7_1 = conv2d(dconv_concat1, W7_1) + B7_1
    # conv7_1 = tf.contrib.layers.batch_norm(conv7_1, epsilon=1e-5, scope='bn11')
    conv7_1 = tf.contrib.layers.batch_norm(conv7_1, center=True, scale=True, is_training=phase, scope='bn11')
    conv7_1 = tf.nn.dropout(tf.nn.relu(conv7_1), drop_conv)
    """
    layer6 = conv_bn_relu_drop(x = layer6, kernal= [3, 3, 256, 256], phase= phase, drop=drop_conv, scope = 'layer6_2')
    """
    W7_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256)
    B7_2 = bias_variable([256])
    conv7_2 = conv2d(conv7_1, W7_2) + B7_2
    # conv7_2 = tf.contrib.layers.batch_norm(conv7_2, epsilon=1e-5, scope='bn12')
    conv7_2 = tf.contrib.layers.batch_norm(conv7_2, center=True, scale=True, is_training=phase, scope='bn12')
    conv7_2 = tf.nn.dropout(tf.nn.relu(conv7_2), drop_conv)
    """
    deconv2 = deconv_relu(layer6, kernal=[3, 3, 128, 256], scope = 'deconv2')
    """
    # layer8->deconvolution
    W8 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 256, n_outputs=128)
    B8 = bias_variable([128])
    dconv2 = tf.nn.relu(deconv2d(conv7_2, W8) + B8)
    """
    layer7 = crop_and_concat(layer3, deconv2)

    # layer9->convolution
    layer7 = conv_bn_relu_drop(x = layer7, kernal= [3, 3, 256, 128], phase=phase, drop=drop_conv, scope = 'layer7_1')
    """
    W9_1 = weight_xavier_init(shape=[3, 3, 256, 128], n_inputs=3 * 3 * 256, n_outputs=128)
    B9_1 = bias_variable([128])
    conv9_1 = conv2d(dconv_concat2, W9_1) + B9_1
    # conv9_1 = tf.contrib.layers.batch_norm(conv9_1, epsilon=1e-5, scope='bn13')
    conv9_1 = tf.contrib.layers.batch_norm(conv9_1, center=True, scale=True, is_training=phase, scope='bn13')
    conv9_1 = tf.nn.dropout(tf.nn.relu(conv9_1), drop_conv)
    """
    layer7 = conv_bn_relu_drop(x = layer7, kernal = [3, 3, 128, 128], phase=phase, drop=drop_conv, scope='layer7_2')
    """
    W9_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128)
    B9_2 = bias_variable([128])
    conv9_2 = conv2d(conv9_1, W9_2) + B9_2
    # conv9_2 = tf.contrib.layers.batch_norm(conv9_2, epsilon=1e-5, scope='bn14')
    conv9_2 = tf.contrib.layers.batch_norm(conv9_2, center=True, scale=True, is_training=phase, scope='bn14')
    conv9_2 = tf.nn.dropout(tf.nn.relu(conv9_2), drop_conv)
    """

    # layer10->deconvolution
    deconv3 = deconv_relu(x = layer7, kernal= [3, 3, 64, 128], scope = 'deconv3')
    """
    W10 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 128, n_outputs=64)
    B10 = bias_variable([64])
    dconv3 = tf.nn.relu(deconv2d(conv9_2, W10) + B10)
    """
    layer8 = crop_and_concat(layer2, deconv3)

    # layer11->convolution
    layer8 = conv_bn_relu_drop(x = layer8, kernal= [3, 3, 128, 64], phase=phase, drop=drop_conv, scope='layer8_1')
    """
    W11_1 = weight_xavier_init(shape=[3, 3, 128, 64], n_inputs=3 * 3 * 128, n_outputs=64)
    B11_1 = bias_variable([64])
    conv11_1 = conv2d(dconv_concat3, W11_1) + B11_1
    # conv11_1 = tf.contrib.layers.batch_norm(conv11_1, epsilon=1e-5, scope='bn15')
    conv11_1 = tf.contrib.layers.batch_norm(conv11_1, center=True, scale=True, is_training=phase, scope='bn15')
    conv11_1 = tf.nn.dropout(tf.nn.relu(conv11_1), drop_conv)
    """
    layer8 = conv_bn_relu_drop(x = layer8, kernal= [3, 3, 64, 64], phase=phase, drop=drop_conv, scope='layer8_2')
    """
    W11_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64)
    B11_2 = bias_variable([64])
    conv11_2 = conv2d(conv11_1, W11_2) + B11_2
    # conv11_2 = tf.contrib.layers.batch_norm(conv11_2, epsilon=1e-5, scope='bn16')
    conv11_2 = tf.contrib.layers.batch_norm(conv11_2, center=True, scale=True, is_training=phase, scope='bn16')
    conv11_2 = tf.nn.dropout(tf.nn.relu(conv11_2), drop_conv)
    """
    deconv4 = deconv_relu(layer8, kernal=[3, 3, 32, 64], scope='deconv4')
    """
    # layer 12->deconvolution
    W12 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 64, n_outputs=32)
    B12 = bias_variable([32])
    dconv4 = tf.nn.relu(deconv2d(conv11_2, W12) + B12)
    """
    layer9 = crop_and_concat(layer1, deconv4)

    # layer 13->convolution
    layer9 = conv_bn_relu_drop(x = layer9, kernal=[3, 3, 64, 32], phase=phase, drop=drop_conv, scope='layer9_1')
    """
    W13_1 = weight_xavier_init(shape=[3, 3, 64, 32], n_inputs=3 * 3 * 64, n_outputs=32)
    B13_1 = bias_variable([32])
    conv13_1 = conv2d(dconv_concat4, W13_1) + B13_1
    # conv13_1 = tf.contrib.layers.batch_norm(conv13_1, epsilon=1e-5, scope='bn17')
    conv13_1 = tf.contrib.layers.batch_norm(conv13_1, center=True, scale=True, is_training=phase, scope='bn17')
    conv13_1 = tf.nn.dropout(tf.nn.relu(conv13_1), drop_conv)
    """
    layer9 = conv_bn_relu_drop(x = layer9, kernal=[3, 3, 32, 32], phase=phase, drop=drop_conv, scope='layer9_2')
    """
    W13_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32)
    B13_2 = bias_variable([32])
    conv13_2 = conv2d(conv13_1, W13_2) + B13_2
    # conv13_2 = tf.contrib.layers.batch_norm(conv13_2, epsilon=1e-5, scope='bn18')
    conv13_2 = tf.contrib.layers.batch_norm(conv13_2, center=True, scale=True, is_training=phase, scope='bn18')
    conv13_2 = tf.nn.dropout(tf.nn.relu(conv13_2), drop_conv)
    """
    # layer14->output
    W14 = weight_xavier_init(shape=[1, 1, 32, n_class], n_inputs=1 * 1 * 32, n_outputs=n_class)
    B14 = bias_variable([n_class])
    output_map = tf.nn.sigmoid(conv2d(layer9, W14) + B14, name='output')
    # print(output_map)

    return output_map


def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class unet2dModule(object):
    """
    A unet2d implementation

    :param image_height: number of height in the input image
    :param image_width: number of width in the input image
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param costname: name of the cost function.Default is "cross_entropy"
    """

    def __init__(self, image_height, image_width, channels=1, costname="dice coefficient"):
        self.image_with = image_width
        self.image_height = image_height
        self.channels = channels

        self.X = tf.placeholder("float", shape=[None, image_height, image_width, channels], name="Input")
        self.Y_gt = tf.placeholder("float", shape=[None, image_height, image_width, 1], name="Output_GT")
        self.lr = tf.placeholder('float', name="Learning_rate")
        self.phase = tf.placeholder(tf.bool, name="Phase")
        self.drop_conv = tf.placeholder('float', name="DropOut")

        self.Y_pred = _create_conv_net(self.X, image_width, image_height, channels, self.phase, self.drop_conv)

        self.cost = self.__get_cost(costname)
        self.accuracy = -self.__get_cost(costname)

    def __get_cost(self, cost_name):
        H, W, C = self.Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(self.Y_pred, [-1, H * W * C])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
        if cost_name == "pixelwise_cross entroy":
            assert (C == 1)
            flat_logit = tf.reshape(self.Y_pred, [-1])
            flat_label = tf.reshape(self.Y_gt, [-1])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit, labels=flat_label))
        return loss

    def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=1000, batch_size=2):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables())

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)
        if os.path.isfile(model_path):
            saver.restore(sess, model_path)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        for i in range(train_epochs):
            # get new batch
            batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, train_lanbels, batch_size,
                                                                       index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_height, self.image_with, self.channels))
            batch_ys = np.empty((len(batch_ys_path), self.image_height, self.image_with, 1))

            for num in range(len(batch_xs_path)):
                image = cv2.imread(batch_xs_path[num][0], cv2.IMREAD_COLOR)
               # cv2.imwrite('image_src.bmp', image)
                label = cv2.imread(batch_ys_path[num][0], cv2.IMREAD_GRAYSCALE)
                #cv2.imwrite('zbqzbq.bmp', label)
                batch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_with, self.channels))
                batch_ys[num, :, :, :] = np.reshape(label, (self.image_height, self.image_with, 1))
               # print("ffffffffffffffffffff")
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = sess.run([self.cost, self.accuracy], feed_dict={self.X: batch_xs,
                                                                                         self.Y_gt: batch_ys,
                                                                                         self.lr: learning_rate,
                                                                                         self.phase: 1,
                                                                                         self.drop_conv: dropout_conv})
                pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
                                                    self.Y_gt: batch_ys,
                                                    self.phase: 1,
                                                    self.drop_conv: 1})
                result = np.reshape(pred[0], (512, 512))
                result = result.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')
                cv2.imwrite("result.bmp", result)
                print('epochs %d training_loss ,Training_accuracy ,Train_hausdorff => %.5f,%.5f,%.5f ' % (
                i, train_loss, train_accuracy, train_accuracy / (2 - train_accuracy)))
                with open("loss.txt", "a+") as f:
                    f.write(str(train_loss))
                    f.write("\n")

                with open("accuracy.txt", "a+") as f:
                    f.write(str(train_accuracy))
                    f.write("\n")

                with open("hausdorff.txt", "a+") as f:
                    f.write(str(train_accuracy / (2 - train_accuracy)))
                    f.write("\n")
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop_conv: dropout_conv})
            summary_writer.add_summary(summary, i)
            if (i % 1000 == 0) or (i + 1) == train_epochs:
                save_path = saver.save(sess, model_path + str(i))
                print("Model saved in file:", save_path)
        summary_writer.close()

        #summary_writer.close()

        #save_path = saver.save(sess, model_path)
        #print("Model saved in file:", save_path)

    def prediction(self, model_path, test_images):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)

        test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], self.channels))
        # test_label = cv2.imread("D:\Data\GlandCeil\Test\Mask\\train_37_anno.bmp", 0)
        # test_label = np.multiply(test_label, 1.0 / 255.0)
        # test_label = np.reshape(test_label, (1, test_label.shape[0], test_label.shape[1], 1))
        pred = sess.run(self.Y_pred, feed_dict={self.X: test_images,
                                                self.phase: 1,
                                                self.drop_conv: 1})
        result = np.reshape(pred, (test_images.shape[1], test_images.shape[2]))
        result = result.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        tf.reset_default_graph()
        return result

















# from uNet.layer import (conv2d, deconv2d, max_pool_2x2, crop_and_concat, weight_xavier_init, bias_variable)
# import tensorflow as tf
# import numpy as np
# import os
# import cv2
#
#
# def deconv_relu(x, kernal, scope = None):
#     with tf.name_scope(scope):
#         W = weight_xavier_init(shape = kernal, n_inputs = kernal[0] * kernal[1] * kernal[-1], n_outputs = kernal[-2],
#                                activefunction = 'relu', variable_name = scope + 'W')
#         B = bias_variable([kernal[-2]], variable_name = scope + 'B')
#         deconv = deconv2d(x, W) + B
#         deconv = tf.nn.relu(deconv)
#         return deconv
#
#
# def normalizationlayer(x, is_train, norm_type = None, scope = None):
#     with tf.name_scope(scope + norm_type):
#         if norm_type == None:
#             output = x
#         elif norm_type == 'batch':
#             output = tf.contrib.layers.batch_norm(x, center = True, scale = True, is_training = is_train)
#         return output
#
# def conv_bn_relu_drop(x, kernal, phase, drop, scope = None):
#     with tf.name_scope(scope):
#         W = weight_xavier_init(shape = kernal, n_inputs = kernal[0] * kernal[1] * kernal[2], n_outputs = kernal[-1],
#                                activefunction = 'relu', variable_name = scope + 'conv_W')
#         B = bias_variable([kernal[-1]], variable_name = scope + 'conv_B')
#         conv = conv2d(x, W) + B
#         conv = normalizationlayer(conv, is_train=phase, norm_type='batch', scope = scope + 'normalization')
#         conv = tf.nn.dropout(tf.nn.relu(conv), drop, name = scope + 'conv_dropout')
#         return conv
#
# def _create_conv_net(X, image_width, image_height, image_channel, phase, drop_conv, n_class=1):
#     inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
#     # UNet model
#     # layer1->convolution
#     layer0 = conv_bn_relu_drop(x = inputX, kernal = [3, 3, image_channel, 32], phase= phase, drop = drop_conv, scope = 'layer0')
#     layer1 = conv_bn_relu_drop(x = layer0, kernal= [3, 3, 32, 32], phase = phase, drop=drop_conv, scope = 'layer1')
#     # print(layer1.get_shape().as_list())
#
#     """
#     W1_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32)
#     B1_2 = bias_variable([32])
#     conv1_2 = conv2d(conv1_1, W1_2) + B1_2
#     # conv1_2 = tf.contrib.layers.batch_norm(conv1_2, epsilon=1e-5, scope='bn2')
#     conv1_2 = tf.contrib.layers.batch_norm(conv1_2, center=True, scale=True, is_training=phase, scope='bn2')
#     conv1_2 = tf.nn.dropout(tf.nn.relu(conv1_2), drop_conv)
#     """
#
#     pool1 = max_pool_2x2(layer1)
#     # layer2->convolution
#     layer2 = conv_bn_relu_drop(x = pool1, kernal = [3, 3, 32, 64], phase= phase, drop = drop_conv, scope= 'layer2_1')
#     """
#     W2_1 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 32, n_outputs=64)
#     B2_1 = bias_variable([64])
#     conv2_1 = conv2d(pool1, W2_1) + B2_1
#     # conv2_1 = tf.contrib.layers.batch_norm(conv2_1, epsilon=1e-5, scope='bn3')
#     conv2_1 = tf.contrib.layers.batch_norm(conv2_1, center=True, scale=True, is_training=phase, scope='bn3')
#     conv2_1 = tf.nn.dropout(tf.nn.relu(conv2_1), drop_conv)
#     """
#
#     layer2 = conv_bn_relu_drop(x = layer2, kernal=[3, 3, 64, 64], phase = phase, drop = drop_conv, scope = 'layer2_2')
#     # print(layer2.get_shape().as_list())
#     """
#     W2_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64)
#     B2_2 = bias_variable([64])
#     conv2_2 = conv2d(conv2_1, W2_2) + B2_2
#     # conv2_2 = tf.contrib.layers.batch_norm(conv2_2, epsilon=1e-5, scope='bn4')
#     conv2_2 = tf.contrib.layers.batch_norm(conv2_2, center=True, scale=True, is_training=phase, scope='bn4')
#     conv2_2 = tf.nn.dropout(tf.nn.relu(conv2_2), drop_conv)
#     """
#     pool2 = max_pool_2x2(layer2)
#
#     # layer3->convolution
#     layer3 = conv_bn_relu_drop(x = pool2, kernal= [3, 3, 64, 128], phase = phase, drop=drop_conv, scope = 'layer3_1')
#     """
#     W3_1 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 64, n_outputs=128)
#     B3_1 = bias_variable([128])
#     conv3_1 = conv2d(pool2, W3_1) + B3_1
#     # conv3_1 = tf.contrib.layers.batch_norm(conv3_1, epsilon=1e-5, scope='bn5')
#     conv3_1 = tf.contrib.layers.batch_norm(conv3_1, center=True, scale=True, is_training=phase, scope='bn5')
#     conv3_1 = tf.nn.dropout(tf.nn.relu(conv3_1), drop_conv)
#     """
#     layer3 = conv_bn_relu_drop(x = layer3, kernal= [3, 3, 128, 128], phase = phase, drop=drop_conv, scope = 'layer3_2')
#     """
#     W3_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128)
#     B3_2 = bias_variable([128])
#     conv3_2 = conv2d(conv3_1, W3_2) + B3_2
#     # conv3_2 = tf.contrib.layers.batch_norm(conv3_2, epsilon=1e-5, scope='bn6')
#     conv3_2 = tf.contrib.layers.batch_norm(conv3_2, center=True, scale=True, is_training=phase, scope='bn6')
#     conv3_2 = tf.nn.dropout(tf.nn.relu(conv3_2), drop_conv)
#     """
#     pool3 = max_pool_2x2(layer3)
#
#     # layer4->convolution
#     layer4 = conv_bn_relu_drop(x = pool3, kernal= [3, 3, 128, 256], phase=phase, drop=drop_conv, scope = 'layer4_1')
#     """
#     W4_1 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 128, n_outputs=256)
#     B4_1 = bias_variable([256])
#     conv4_1 = conv2d(pool3, W4_1) + B4_1
#     # conv4_1 = tf.contrib.layers.batch_norm(conv4_1, epsilon=1e-5, scope='bn7')
#     conv4_1 = tf.contrib.layers.batch_norm(conv4_1, center=True, scale=True, is_training=phase, scope='bn7')
#     conv4_1 = tf.nn.dropout(tf.nn.relu(conv4_1), drop_conv)
#     """
#     layer4 = conv_bn_relu_drop(x = layer4, kernal=[3, 3, 256, 256], phase=phase, drop=drop_conv, scope= 'layer4_2')
#     """
#     W4_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256)
#     B4_2 = bias_variable([256])
#     conv4_2 = conv2d(conv4_1, W4_2) + B4_2
#     # conv4_2 = tf.contrib.layers.batch_norm(conv4_2, epsilon=1e-5, scope='bn8')
#     conv4_2 = tf.contrib.layers.batch_norm(conv4_2, center=True, scale=True, is_training=phase, scope='bn8')
#     conv4_2 = tf.nn.dropout(tf.nn.relu(conv4_2), drop_conv)
#     """
#     pool4 = max_pool_2x2(layer4)
#
#     # layer5->convolution
#     layer5 = conv_bn_relu_drop(x = pool4, kernal= [3, 3, 256, 512], phase=phase, drop=drop_conv, scope='layer5_1')
#     """
#     W5_1 = weight_xavier_init(shape=[3, 3, 256, 512], n_inputs=3 * 3 * 256, n_outputs=512)
#     B5_1 = bias_variable([512])
#     conv5_1 = conv2d(pool4, W5_1) + B5_1
#     # conv5_1 = tf.contrib.layers.batch_norm(conv5_1, epsilon=1e-5, scope='bn9')
#     conv5_1 = tf.contrib.layers.batch_norm(conv5_1, center=True, scale=True, is_training=phase, scope='bn9')
#     conv5_1 = tf.nn.dropout(tf.nn.relu(conv5_1), drop_conv)
#     """
#     layer5 = conv_bn_relu_drop(x = layer5, kernal=[3, 3, 512, 512], phase=phase, drop = drop_conv, scope='layer5_2')
#     """
#     W5_2 = weight_xavier_init(shape=[3, 3, 512, 512], n_inputs=3 * 3 * 512, n_outputs=512)
#     B5_2 = bias_variable([512])
#     conv5_2 = conv2d(conv5_1, W5_2) + B5_2
#     # conv5_2 = tf.contrib.layers.batch_norm(conv5_2, epsilon=1e-5, scope='bn10')
#     conv5_2 = tf.contrib.layers.batch_norm(conv5_2, center=True, scale=True, is_training=phase, scope='bn10')
#     conv5_2 = tf.nn.dropout(tf.nn.relu(conv5_2), drop_conv)
#     """
#
#     # layer6->deconvolution
#
#     deconv1 = deconv_relu(x = layer5, kernal=[3, 3, 256, 512], scope = 'deconv1')
#     """
#     W6 = weight_xavier_init(shape=[3, 3, 256, 512], n_inputs=3 * 3 * 512, n_outputs=256)
#     B6 = bias_variable([256])
#     dconv1 = tf.nn.relu(deconv2d(conv5_2, W6) + B6)
#     """
#     layer6 = crop_and_concat(layer4, deconv1)
#
#
#     # layer7->convolution
#     layer6 = conv_bn_relu_drop(x = layer6, kernal= [3, 3, 512, 256], phase=phase, drop=drop_conv, scope = 'layer6_1')
#     """
#     W7_1 = weight_xavier_init(shape=[3, 3, 512, 256], n_inputs=3 * 3 * 512, n_outputs=256)
#     B7_1 = bias_variable([256])
#     conv7_1 = conv2d(dconv_concat1, W7_1) + B7_1
#     # conv7_1 = tf.contrib.layers.batch_norm(conv7_1, epsilon=1e-5, scope='bn11')
#     conv7_1 = tf.contrib.layers.batch_norm(conv7_1, center=True, scale=True, is_training=phase, scope='bn11')
#     conv7_1 = tf.nn.dropout(tf.nn.relu(conv7_1), drop_conv)
#     """
#     layer6 = conv_bn_relu_drop(x = layer6, kernal= [3, 3, 256, 256], phase= phase, drop=drop_conv, scope = 'layer6_2')
#     """
#     W7_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256)
#     B7_2 = bias_variable([256])
#     conv7_2 = conv2d(conv7_1, W7_2) + B7_2
#     # conv7_2 = tf.contrib.layers.batch_norm(conv7_2, epsilon=1e-5, scope='bn12')
#     conv7_2 = tf.contrib.layers.batch_norm(conv7_2, center=True, scale=True, is_training=phase, scope='bn12')
#     conv7_2 = tf.nn.dropout(tf.nn.relu(conv7_2), drop_conv)
#     """
#     deconv2 = deconv_relu(layer6, kernal=[3, 3, 128, 256], scope = 'deconv2')
#     """
#     # layer8->deconvolution
#     W8 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 256, n_outputs=128)
#     B8 = bias_variable([128])
#     dconv2 = tf.nn.relu(deconv2d(conv7_2, W8) + B8)
#     """
#     layer7 = crop_and_concat(layer3, deconv2)
#
#     # layer9->convolution
#     layer7 = conv_bn_relu_drop(x = layer7, kernal= [3, 3, 256, 128], phase=phase, drop=drop_conv, scope = 'layer7_1')
#     """
#     W9_1 = weight_xavier_init(shape=[3, 3, 256, 128], n_inputs=3 * 3 * 256, n_outputs=128)
#     B9_1 = bias_variable([128])
#     conv9_1 = conv2d(dconv_concat2, W9_1) + B9_1
#     # conv9_1 = tf.contrib.layers.batch_norm(conv9_1, epsilon=1e-5, scope='bn13')
#     conv9_1 = tf.contrib.layers.batch_norm(conv9_1, center=True, scale=True, is_training=phase, scope='bn13')
#     conv9_1 = tf.nn.dropout(tf.nn.relu(conv9_1), drop_conv)
#     """
#     layer7 = conv_bn_relu_drop(x = layer7, kernal = [3, 3, 128, 128], phase=phase, drop=drop_conv, scope='layer7_2')
#     """
#     W9_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128)
#     B9_2 = bias_variable([128])
#     conv9_2 = conv2d(conv9_1, W9_2) + B9_2
#     # conv9_2 = tf.contrib.layers.batch_norm(conv9_2, epsilon=1e-5, scope='bn14')
#     conv9_2 = tf.contrib.layers.batch_norm(conv9_2, center=True, scale=True, is_training=phase, scope='bn14')
#     conv9_2 = tf.nn.dropout(tf.nn.relu(conv9_2), drop_conv)
#     """
#
#     # layer10->deconvolution
#     deconv3 = deconv_relu(x = layer7, kernal= [3, 3, 64, 128], scope = 'deconv3')
#     """
#     W10 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 128, n_outputs=64)
#     B10 = bias_variable([64])
#     dconv3 = tf.nn.relu(deconv2d(conv9_2, W10) + B10)
#     """
#     layer8 = crop_and_concat(layer2, deconv3)
#
#     # layer11->convolution
#     layer8 = conv_bn_relu_drop(x = layer8, kernal= [3, 3, 128, 64], phase=phase, drop=drop_conv, scope='layer8_1')
#     """
#     W11_1 = weight_xavier_init(shape=[3, 3, 128, 64], n_inputs=3 * 3 * 128, n_outputs=64)
#     B11_1 = bias_variable([64])
#     conv11_1 = conv2d(dconv_concat3, W11_1) + B11_1
#     # conv11_1 = tf.contrib.layers.batch_norm(conv11_1, epsilon=1e-5, scope='bn15')
#     conv11_1 = tf.contrib.layers.batch_norm(conv11_1, center=True, scale=True, is_training=phase, scope='bn15')
#     conv11_1 = tf.nn.dropout(tf.nn.relu(conv11_1), drop_conv)
#     """
#     layer8 = conv_bn_relu_drop(x = layer8, kernal= [3, 3, 64, 64], phase=phase, drop=drop_conv, scope='layer8_2')
#     """
#     W11_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64)
#     B11_2 = bias_variable([64])
#     conv11_2 = conv2d(conv11_1, W11_2) + B11_2
#     # conv11_2 = tf.contrib.layers.batch_norm(conv11_2, epsilon=1e-5, scope='bn16')
#     conv11_2 = tf.contrib.layers.batch_norm(conv11_2, center=True, scale=True, is_training=phase, scope='bn16')
#     conv11_2 = tf.nn.dropout(tf.nn.relu(conv11_2), drop_conv)
#     """
#     deconv4 = deconv_relu(layer8, kernal=[3, 3, 32, 64], scope='deconv4')
#     """
#     # layer 12->deconvolution
#     W12 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 64, n_outputs=32)
#     B12 = bias_variable([32])
#     dconv4 = tf.nn.relu(deconv2d(conv11_2, W12) + B12)
#     """
#     layer9 = crop_and_concat(layer1, deconv4)
#
#     # layer 13->convolution
#     layer9 = conv_bn_relu_drop(x = layer9, kernal=[3, 3, 64, 32], phase=phase, drop=drop_conv, scope='layer9_1')
#     """
#     W13_1 = weight_xavier_init(shape=[3, 3, 64, 32], n_inputs=3 * 3 * 64, n_outputs=32)
#     B13_1 = bias_variable([32])
#     conv13_1 = conv2d(dconv_concat4, W13_1) + B13_1
#     # conv13_1 = tf.contrib.layers.batch_norm(conv13_1, epsilon=1e-5, scope='bn17')
#     conv13_1 = tf.contrib.layers.batch_norm(conv13_1, center=True, scale=True, is_training=phase, scope='bn17')
#     conv13_1 = tf.nn.dropout(tf.nn.relu(conv13_1), drop_conv)
#     """
#     layer9 = conv_bn_relu_drop(x = layer9, kernal=[3, 3, 32, 32], phase=phase, drop=drop_conv, scope='layer9_2')
#     """
#     W13_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32)
#     B13_2 = bias_variable([32])
#     conv13_2 = conv2d(conv13_1, W13_2) + B13_2
#     # conv13_2 = tf.contrib.layers.batch_norm(conv13_2, epsilon=1e-5, scope='bn18')
#     conv13_2 = tf.contrib.layers.batch_norm(conv13_2, center=True, scale=True, is_training=phase, scope='bn18')
#     conv13_2 = tf.nn.dropout(tf.nn.relu(conv13_2), drop_conv)
#     """
#     # layer14->output
#     W14 = weight_xavier_init(shape=[1, 1, 32, n_class], n_inputs=1 * 1 * 32, n_outputs=n_class)
#     B14 = bias_variable([n_class])
#     output_map = tf.nn.sigmoid(conv2d(layer9, W14) + B14, name='output')
#
#     return output_map
#
#
# def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
#     start = index_in_epoch
#     index_in_epoch += batch_size
#
#     num_examples = train_images.shape[0]
#     # when all trainig data have been already used, it is reorder randomly
#     if index_in_epoch > num_examples:
#         # shuffle the data
#         perm = np.arange(num_examples)
#         np.random.shuffle(perm)
#         train_images = train_images[perm]
#         train_labels = train_labels[perm]
#         # start next epoch
#         start = 0
#         index_in_epoch = batch_size
#         assert batch_size <= num_examples
#     end = index_in_epoch
#     return train_images[start:end], train_labels[start:end], index_in_epoch
#
#
# class unet2dModule(object):
#     """
#     A unet2d implementation
#
#     :param image_height: number of height in the input image
#     :param image_width: number of width in the input image
#     :param channels: number of channels in the input image
#     :param n_class: number of output labels
#     :param costname: name of the cost function.Default is "cross_entropy"
#     """
#
#     def __init__(self, image_height, image_width, channels=1, costname="dice coefficient"):
#         self.image_with = image_width
#         self.image_height = image_height
#         self.channels = channels
#
#         self.X = tf.placeholder("float", shape=[None, image_height, image_width, channels], name="Input")
#         self.Y_gt = tf.placeholder("float", shape=[None, image_height, image_width, 1], name="Output_GT")
#         self.lr = tf.placeholder('float', name="Learning_rate")
#         self.phase = tf.placeholder(tf.bool, name="Phase")
#         self.drop_conv = tf.placeholder('float', name="DropOut")
#
#         self.Y_pred = _create_conv_net(self.X, image_width, image_height, channels, self.phase, self.drop_conv)
#
#         self.cost = self.__get_cost(costname)
#         self.accuracy = -self.__get_cost(costname)
#
#     def __get_cost(self, cost_name):
#         H, W, C = self.Y_gt.get_shape().as_list()[1:]
#         if cost_name == "dice coefficient":
#             smooth = 1e-5
#             pred_flat = tf.reshape(self.Y_pred, [-1, H * W * C])
#             true_flat = tf.reshape(self.Y_gt, [-1, H * W * C])
#             intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
#             denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
#             loss = -tf.reduce_mean(intersection / denominator)
#         if cost_name == "pixelwise_cross entroy":
#             assert (C == 1)
#             flat_logit = tf.reshape(self.Y_pred, [-1])
#             flat_label = tf.reshape(self.Y_gt, [-1])
#             loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit, labels=flat_label))
#         return loss
#
#     def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
#               dropout_conv=0.8, train_epochs=1000, batch_size=2):
#         train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
#
#         init = tf.global_variables_initializer()
#         saver = tf.train.Saver(tf.all_variables())
#
#         tf.summary.scalar("loss", self.cost)
#         tf.summary.scalar("accuracy", self.accuracy)
#         merged_summary_op = tf.summary.merge_all()
#         sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
#         summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
#         sess.run(init)
#         if os.path.isfile(model_path):
#             saver.restore(sess, model_path)
#
#         DISPLAY_STEP = 1
#         index_in_epoch = 0
#
#         for i in range(train_epochs):
#             # get new batch
#             batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, train_lanbels, batch_size,
#                                                                        index_in_epoch)
#             batch_xs = np.empty((len(batch_xs_path), self.image_height, self.image_with, self.channels))
#             batch_ys = np.empty((len(batch_ys_path), self.image_height, self.image_with, 1))
#
#             for num in range(len(batch_xs_path)):
#                 image = cv2.imread(batch_xs_path[num][0], cv2.IMREAD_COLOR)
#                # cv2.imwrite('image_src.bmp', image)
#                 label = cv2.imread(batch_ys_path[num][0], cv2.IMREAD_GRAYSCALE)
#                 #cv2.imwrite('zbqzbq.bmp', label)
#                 batch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_with, self.channels))
#                 batch_ys[num, :, :, :] = np.reshape(label, (self.image_height, self.image_with, 1))
#                # print("ffffffffffffffffffff")
#             # Extracting images and labels from given data
#             batch_xs = batch_xs.astype(np.float)
#             batch_ys = batch_ys.astype(np.float)
#             # Normalize from [0:255] => [0.0:1.0]
#             batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
#             batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
#             # check progress on every 1st,2nd,...,10th,20th,...,100th... step
#             if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
#                 train_loss, train_accuracy = sess.run([self.cost, self.accuracy], feed_dict={self.X: batch_xs,
#                                                                                              self.Y_gt: batch_ys,
#                                                                                              self.lr: learning_rate,
#                                                                                              self.phase: 1,
#                                                                                              self.drop_conv: dropout_conv})
#                 pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
#                                                         self.Y_gt: batch_ys,
#                                                         self.phase: 1,
#                                                         self.drop_conv: 1})
#                 print(pred)
#                 result = np.reshape(pred[0], (512, 512))
#                 result = result.astype(np.float32) * 255.
#                 result = np.clip(result, 0, 255).astype('uint8')
#                 cv2.imwrite("result.bmp", result)
#                 print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))
#                 if i % (DISPLAY_STEP * 10) == 0 and i:
#                     DISPLAY_STEP *= 10
#
#                     # train on batch
#             _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
#                                                                             self.Y_gt: batch_ys,
#                                                                             self.lr: learning_rate,
#                                                                             self.phase: 1,
#                                                                             self.drop_conv: dropout_conv})
#             summary_writer.add_summary(summary, i)
#         summary_writer.close()
#
#         save_path = saver.save(sess, model_path)
#         print("Model saved in file:", save_path)
#
#     def prediction(self, model_path, test_images):
#
#         init = tf.global_variables_initializer()
#         saver = tf.train.Saver()
#         sess = tf.InteractiveSession()
#         sess.run(init)
#         saver.restore(sess, model_path)
#
#         test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], self.channels))
#         # test_label = cv2.imread("D:\Data\GlandCeil\Test\Mask\\train_37_anno.bmp", 0)
#         # test_label = np.multiply(test_label, 1.0 / 255.0)
#         # test_label = np.reshape(test_label, (1, test_label.shape[0], test_label.shape[1], 1))
#         pred = sess.run(self.Y_pred, feed_dict={self.X: test_images,
#                                                 self.phase: 1,
#                                                 self.drop_conv: 1})
#         result = np.reshape(pred, (test_images.shape[1], test_images.shape[2]))
#         result = result.astype(np.float32) * 255.
#         result = np.clip(result, 0, 255).astype('uint8')
#         tf.reset_default_graph()
#
#         return result
