from uNet.DRUnet_layer import (conv_bn, resnet_ADD, crop_and_concat, max_pooling_2x2, deconv_relu)
import tensorflow as tf
import numpy as np
import os
import cv2

def _create_conv_net(X, phase, drop):
    _, width, height, channel = X.get_shape().as_list()  # 拿出 输入 tensor 的 最后一维:也就是通道数
    inputX = tf.reshape(X, [-1, width, height, channel])  # 将图片转换成tf识别格式

    inputX = conv_bn(x=inputX, kernal=(7, 7, channel, 16), phase=phase, scope='input')
    # layer1 -> convolution
    layer0 = conv_bn(x=inputX, kernal=(3, 3, 16, 16), phase=phase, scope='layer0')
    layer0_relu = tf.nn.leaky_relu(layer0)
    layer1_2 = conv_bn(x=layer0_relu, kernal=(3, 3, 16, 16), phase=phase, scope='layer1')
    layer1 = resnet_ADD(x1=layer0, x2=layer1_2)
    layer1 = resnet_ADD(x1=layer1, x2=inputX)
    layer1_relu = tf.nn.leaky_relu(layer1)
    layer1_out = crop_and_concat(layer1_relu, inputX)

    pool1 = max_pooling_2x2(layer1_out)

    # layer2 -> convolution
    layer2_1 = conv_bn(x=pool1, kernal=(3, 3, 32, 32), phase=phase, scope='layer2_1')
    layer2_relu = tf.nn.leaky_relu(layer2_1)
    layer2_2 = conv_bn(x=layer2_relu, kernal=(3, 3, 32, 32), phase=phase, scope='layer2_2')
    layer2 = resnet_ADD(x1=layer2_1, x2=layer2_2)
    # layer2 = resnet_ADD(x1=layer2, x2=pool1)
    layer2 = layer2 + pool1
    layer2_relu = tf.nn.leaky_relu(layer2)
    layer2_out = crop_and_concat(layer2_relu, pool1)

    pool2 = max_pooling_2x2(layer2_out)

    # layer3 -> convolution
    layer3_1 = conv_bn(x=pool2, kernal=(3, 3, 64, 64), phase=phase, scope='layer3_1')
    layer3_relu = tf.nn.leaky_relu(layer3_1)
    layer3_2 = conv_bn(x=layer3_relu, kernal=(3, 3, 64, 64), phase=phase, scope='layer3_2')
    layer3 = resnet_ADD(x1=layer3_1, x2=layer3_2)
    # layer3 = resnet_ADD(x1=layer3, x2=pool2)
    layer3 = layer3 + pool2
    layer3_relu = tf.nn.leaky_relu(layer3)
    layer3_out = crop_and_concat(layer3_relu, pool2)

    pool3 = max_pooling_2x2(layer3_out)


    # layer4 -> convolution
    layer4_1 = conv_bn(x=pool3, kernal=(3, 3, 128, 128), phase=phase, scope='layer4_1')
    layer4_relu = tf.nn.leaky_relu(layer4_1)
    layer4_2 = conv_bn(x=layer4_relu, kernal=(3, 3, 128, 128), phase=phase, scope='layer4_2')
    layer4 = resnet_ADD(x1=layer4_1, x2=layer4_2)
    # layer4 = resnet_ADD(x1=layer4, x2=pool3)
    layer4 = layer4 + pool3
    layer4_relu = tf.nn.leaky_relu(layer4)
    layer4_out = crop_and_concat(layer4_relu, pool3)

    pool4 = max_pooling_2x2(layer4_out)

    # layer5 -> convolution
    layer5_1 = conv_bn(x=pool4, kernal=(3, 3, 256, 256), phase=phase, scope='layer5_1')
    layer5_relu = tf.nn.leaky_relu(layer5_1)
    layer5_2 = conv_bn(x=layer5_relu, kernal=(3, 3, 256, 256), phase=phase, scope='layer5_2')
    layer5 = resnet_ADD(x1=layer5_1, x2=layer5_2)
    # layer5 = resnet_ADD(x1=layer5, x2=pool4)
    layer5 = layer5 + pool4
    layer5_relu = tf.nn.leaky_relu(layer5)
    layer5_out = crop_and_concat(layer5_relu, pool4)  #layer5_relu = (?, 32, 32, 256) pool4 = (?, 32, 32, 256) layer5_out = (?, 32, 32, 512)


    # deconvolution1
    deconv1 = deconv_relu(x=layer5_out, kernal=(3, 3, 256, 512), scope='deconv1')  #deconv1 = (?, 64, 64, 256)


    layer6 = conv_bn(x=deconv1, kernal=(3, 3, 256, 128), phase=phase, scope='layer6_0')
    layer6 = crop_and_concat(layer4_2, layer6)    #layer4_2 = (?, 64, 64, 128), layer6 = (?, 64, 64, 128), layer6 = (?, 64, 64, 256)
    layer6_1 = conv_bn(x=layer6, kernal=(3, 3, 256, 128), phase=phase, scope='layer6_1')
    layer6_relu = tf.nn.leaky_relu(layer6_1)
    layer6_2 = conv_bn(x=layer6_relu, kernal=(3, 3, 128, 128), phase=phase, scope='layer6_2')
    layer6 = resnet_ADD(x1=layer6_1, x2=layer6_2)
    layer6_conv1X1 = conv_bn(x=deconv1, kernal=(1, 1, 256, 128), phase=phase, scope='layer6_conv1X1')
    #layer6=(?, 64, 64, 128) layer6_conv1X1 = (?, 32, 32, 128)

    layer6 = resnet_ADD(x1=layer6, x2=layer6_conv1X1)
    layer6_out = tf.nn.leaky_relu(layer6)  #layer6_out = 128

    # deconvolution2
    deconv2 = deconv_relu(x=layer6_out, kernal=(3, 3, 64, 128), scope='deconv2')

    layer7 = crop_and_concat(layer3_2, deconv2)
    layer7_1 = conv_bn(x=layer7, kernal=(3, 3, 128, 64), phase=phase, scope='layer7_1')
    layer7_relu = tf.nn.leaky_relu(layer7_1)
    layer7_2 = conv_bn(x=layer7_relu, kernal=(3, 3, 64, 64), phase=phase, scope='layer7_2')
    layer7 = resnet_ADD(x1=layer7_1, x2=layer7_2)
    layer7_conv1X1 = conv_bn(x=deconv2, kernal=(1, 1, 64, 64), phase=phase, scope='layer7_conv1X1')
    layer7 = resnet_ADD(x1=layer7, x2=layer7_conv1X1)
    layer7_out = tf.nn.leaky_relu(layer7)

    # deconvolution3
    deconv3 = deconv_relu(x=layer7_out, kernal=(3, 3, 32, 64), scope='deconv3')

    layer8 = crop_and_concat(layer2_2, deconv3)
    layer8_1 = conv_bn(x=layer8, kernal=(3, 3, 64, 32), phase=phase, scope='layer8_1')
    layer8_relu = tf.nn.leaky_relu(layer8_1)
    layer8_2 = conv_bn(x=layer8_relu, kernal=(3, 3, 32, 32), phase=phase, scope='layer8_2')
    layer8 = resnet_ADD(x1=layer8_1, x2=layer8_2)
    layer8_conv1X1 = conv_bn(x=deconv3, kernal=(1, 1, 32, 32), phase=phase, scope='layer8_conv1X1')
    layer8 = resnet_ADD(x1=layer8, x2=layer8_conv1X1)
    layer8_out = tf.nn.leaky_relu(layer8)

    # deconvolution4
    deconv4 = deconv_relu(x=layer8_out, kernal=(3, 3, 16, 32), scope='deconv4')

    layer9 = crop_and_concat(layer1_2, deconv4)
    layer9_1 = conv_bn(x=layer9, kernal=(3, 3, 32, 16), phase=phase, scope='layer9_1')
    layer9_relu = tf.nn.leaky_relu(layer9_1)
    layer9_2 = conv_bn(x=layer9_relu, kernal=(3, 3, 16, 16), phase=phase, scope='layer9_2')
    layer9 = resnet_ADD(x1=layer9_1, x2=layer9_2)
    layer9_conv1X1 = conv_bn(x=deconv4, kernal=(1, 1, 16, 16), phase=phase, scope='layer9_conv1X1')
    layer9 = resnet_ADD(x1=layer9, x2=layer9_conv1X1)
    layer9_out = tf.nn.leaky_relu(layer9)


    output = conv_bn(x=layer9_out, kernal=(1, 1, 16, 1), phase=phase, scope='output_1')
    output_map = tf.nn.sigmoid(output, name='output')
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
        self.phase = tf.placeholder(tf.bool, name="Phase")   #是否训练
        self.drop_conv = tf.placeholder('float', name="DropOut")

        self.Y_pred = _create_conv_net(self.X, self.phase, self.drop_conv)

        self.cost = self.__get_cost(costname)  #损失计算
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

                # dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化
                # print("begin: %s" % dt_ms)
                pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
                                                        self.Y_gt: batch_ys,
                                                        self.phase: 1,
                                                        self.drop_conv: 1})

                # dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化
                # print("end: %s" % dt_ms)
                result = np.reshape(pred[0], (512, 512))
                result = result.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')
                cv2.imwrite("result.bmp", result)
                print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop_conv: dropout_conv})
            summary_writer.add_summary(summary, i)

            if (i > 50000 and i % 10000 == 0) or (i + 1) == train_epochs:
                save_path = saver.save(sess, model_path + str(i))
                print("Model saved in file:", save_path)
        summary_writer.close()

        # save_path = saver.save(sess, model_path)
        # print("Model saved in file:", save_path)

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
