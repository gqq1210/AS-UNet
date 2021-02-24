'''

'''
from uNet.layer import (concat, conv2d, deconv2d, max_pooling_2x2, average_pooling_2x2, crop_and_concat, weight_xavier_init, bias_variable, conv_sigmoid, conv_bn_relu_drop, deconv_relu, conv_sigmoid, upsampling)
import tensorflow as tf
import numpy as np
import os
import cv2
from keras import backend as K
from scipy.ndimage import distance_transform_edt as distance
import PIL.Image


def full_connected_relu(x, kernal, activefuncation='relu', scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        FC = tf.matmul(x, W) + B
        if activefuncation == 'relu':
            FC = tf.nn.relu(FC)
        elif activefuncation == 'softmax':
            FC = tf.nn.softmax(FC)
        elif activefuncation == 'sigmoid':
            FC = tf.nn.sigmoid(FC)
        return FC

def channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=None):
    with tf.name_scope(scope):
        squeeze = conv_sigmoid(x, kernal=(1, 1, out_dim, 1), scope=scope + 'spatial_squeeze')
        scale = x * squeeze
        return scale

def squeeze_excitation_model(x, out_dim, name='ssce', ratio=4, scope=None):
    with tf.name_scope(scope):
        if name == 'ssce':
            recalibrate = spatial_squeeze_channel_excitation_layer(x, out_dim, ratio, scope=scope + 'ssce')
            return recalibrate
        if name == 'csse':
            recalibrate = channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=scope + 'csse')
            return recalibrate


def spatial_squeeze_channel_excitation_layer(x, out_dim, ratio=4, scope=None):
    with tf.name_scope(scope):
        # Global_Average_Pooling, channel_squeeze
        squeeze = tf.reduce_mean(x, axis=(1, 2), name=scope + 'channel_squeeze')  # 这里有可能不对
        # full_connect
        exciation = full_connected_relu(squeeze, kernal=(out_dim, out_dim // ratio), activefuncation='relu',
                                        scope=scope + '_fully_connected1')
        exciation = full_connected_relu(exciation, kernal=(out_dim // ratio, out_dim), activefuncation='sigmoid',
                                        scope=scope + '_fully_connected2')
        # scale the x
        exciation = tf.reshape(exciation, [-1, 1, 1, out_dim])
        scale = x * exciation
        return scale



def attention_layer(x, ratio=4, scope=None):
    with tf.name_scope(scope):
        _, width, height, channel = x.get_shape().as_list()
        x_shape = x.get_shape().as_list()
        recalibrate1 = conv_sigmoid(x, kernal=(1, 1, channel, 1), scope=scope + 'spatial_squeeze')

        squeeze = tf.reduce_mean(x, axis=(1, 2), name=scope + 'channel_squeeze')
        exciation = full_connected_relu(squeeze, kernal=(channel, channel // ratio), activefuncation='relu',
                                        scope=scope + '_fully_connected1')
        exciation = full_connected_relu(exciation, kernal=(channel // ratio, channel), activefuncation='sigmoid',
                                        scope=scope + '_fully_connected2')
        recalibrate2 = tf.reshape(exciation, [-1, 1, 1, channel])

        recalibrate4 = tf.multiply(recalibrate1, recalibrate2)
        out = tf.multiply(recalibrate4, x)
        return out






def _create_conv_net(X, image_width, image_height, image_channel, phase, drop_conv, n_class=1):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # UNet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x = inputX, kernal = [3, 3, image_channel, 32], phase= phase, drop = drop_conv, scope = 'layer0')
    layer1 = conv_bn_relu_drop(x = layer0, kernal= [3, 3, 32, 32], phase = phase, drop=drop_conv, scope = 'layer1')
    # print(layer1.get_shape().as_list())
    layer1 = squeeze_excitation_model(layer1, out_dim=32, scope='sem1')
    pool1 = max_pooling_2x2(layer1)
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x = pool1, kernal = [3, 3, 32, 64], phase= phase, drop = drop_conv, scope= 'layer2_1')
    layer2 = conv_bn_relu_drop(x = layer2, kernal=[3, 3, 64, 64], phase = phase, drop = drop_conv, scope = 'layer2_2')
    layer2 = squeeze_excitation_model(layer2, out_dim=64, scope='sem2')
    pool2 = max_pooling_2x2(layer2)

    # layer3->convolution
    layer3 = conv_bn_relu_drop(x = pool2, kernal= [3, 3, 64, 128], phase = phase, drop=drop_conv, scope = 'layer3_1')
    layer3 = conv_bn_relu_drop(x = layer3, kernal= [3, 3, 128, 128], phase = phase, drop=drop_conv, scope = 'layer3_2')
    layer3 = squeeze_excitation_model(layer3, out_dim=128, scope='sem3')
    pool3 = max_pooling_2x2(layer3)

    # layer4->convolution
    layer4 = conv_bn_relu_drop(x = pool3, kernal= [3, 3, 128, 256], phase=phase, drop=drop_conv, scope = 'layer4_1')
    layer4 = conv_bn_relu_drop(x = layer4, kernal=[3, 3, 256, 256], phase=phase, drop=drop_conv, scope= 'layer4_2')
    layer4 = squeeze_excitation_model(layer4, out_dim=256, scope='sem4')
    pool4 = max_pooling_2x2(layer4)

    # layer5->convolution
    layer5 = conv_bn_relu_drop(x = pool4, kernal= [3, 3, 256, 512], phase=phase, drop=drop_conv, scope='layer5_1')
    layer5 = conv_bn_relu_drop(x = layer5, kernal=[3, 3, 512, 512], phase=phase, drop = drop_conv, scope='layer5_2')

    # layer6->deconvolution

    deconv1 = deconv_relu(x = layer5, kernal=[3, 3, 256, 512], scope = 'deconv1')

    layer6 = crop_and_concat(layer4, deconv1)


    # layer7->convolution
    layer6 = conv_bn_relu_drop(x = layer6, kernal= [3, 3, 512, 256], phase=phase, drop=drop_conv, scope = 'layer6_1')
    layer6 = conv_bn_relu_drop(x = layer6, kernal= [3, 3, 256, 256], phase= phase, drop=drop_conv, scope = 'layer6_2')
    layer6 = squeeze_excitation_model(layer6, out_dim=256, scope='sem5')
    deconv2 = deconv_relu(layer6, kernal=[3, 3, 128, 256], scope = 'deconv2')

    layer7 = crop_and_concat(layer3, deconv2)

    # layer9->convolution
    layer7 = conv_bn_relu_drop(x = layer7, kernal= [3, 3, 256, 128], phase=phase, drop=drop_conv, scope = 'layer7_1')
    layer7 = conv_bn_relu_drop(x = layer7, kernal = [3, 3, 128, 128], phase=phase, drop=drop_conv, scope='layer7_2')

    layer7 = squeeze_excitation_model(layer7, out_dim=128, scope='sem6')
    # layer10->deconvolution
    deconv3 = deconv_relu(x = layer7, kernal= [3, 3, 64, 128], scope = 'deconv3')

    layer8 = crop_and_concat(layer2, deconv3)

    # layer11->convolution
    layer8 = conv_bn_relu_drop(x = layer8, kernal= [3, 3, 128, 64], phase=phase, drop=drop_conv, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x = layer8, kernal= [3, 3, 64, 64], phase=phase, drop=drop_conv, scope='layer8_2')
    layer8 = squeeze_excitation_model(layer8, out_dim=64, scope='sem7')
    deconv4 = deconv_relu(layer8, kernal=[3, 3, 32, 64], scope='deconv4')

    layer9 = crop_and_concat(layer1, deconv4)

    # layer 13->convolution
    layer9 = conv_bn_relu_drop(x = layer9, kernal=[3, 3, 64, 32], phase=phase, drop=drop_conv, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x = layer9, kernal=[3, 3, 32, 32], phase=phase, drop=drop_conv, scope='layer9_2')
    layer9 = squeeze_excitation_model(layer9, out_dim=32, scope='sem8')
    output = conv_sigmoid(x=layer9, kernal=[1, 1, 32, n_class], scope='output')


    return layer7, layer8, layer9, output



def subtract_refine(subtract, layer7, layer8, layer9, phase, drop_conv, n_class=1):
    _, y_width, y_height, y_channel = subtract.get_shape().as_list()  # 拿出 输入 tensor 的 最后一维:也就是通道数
    seed = tf.reshape(subtract, [-1, y_width, y_height, y_channel])  # 将图片转换成tf识别格式


    refine1 = conv_bn_relu_drop(x=layer7, kernal=[1, 1, 128, 64], phase=phase, drop=drop_conv, scope='refine1_1')
    seed1 = tf.image.resize_images(images=seed, size=[int(y_width / 4), int(y_height / 4)], method=0)
    refine1 = crop_and_concat(refine1, seed1)
    refine1 = conv_bn_relu_drop(x=refine1, kernal=[3, 3, 65, 65], phase=phase, drop=drop_conv, scope='refine1_2')
    # refine1 = squeeze_excitation_model(refine1, out_dim=65, scope='sem2_1')
    refine1 = attention_layer(refine1, scope='attention1')
    refine1 = deconv_relu(x=refine1, kernal=[3, 3, 32, 65], scope='refine1_deconv')


    refine2 = conv_bn_relu_drop(x=layer8, kernal=[1, 1, 64, 32], phase=phase, drop=drop_conv, scope='refine2_1')
    seed2 = tf.image.resize_images(images=seed, size=[int(y_width / 2), int(y_height / 2)], method=0)
    refine2 = crop_and_concat(refine2, seed2)
    refine2 = conv_bn_relu_drop(x=refine2, kernal=[3, 3, 33, 32], phase=phase, drop=drop_conv, scope='refine2_2')
    refine2 = crop_and_concat(refine1, refine2)
    refine2 = conv_bn_relu_drop(x=refine2, kernal=[3, 3, 64, 64], phase=phase, drop=drop_conv, scope='refine1_3')
    # refine2 = squeeze_excitation_model(refine2, out_dim=64, scope='sem2_2')
    refine2 = attention_layer(refine2, scope='attention2')
    refine2 = deconv_relu(x=refine2, kernal=[3, 3, 32, 64], scope='refine2_deconv')



    refine3 = conv_bn_relu_drop(x=layer9, kernal=[1, 1, 32, 1], phase=phase, drop=drop_conv, scope='refine3_1')
    seed3 = tf.image.resize_images(images=seed, size=[int(y_width), int(y_height)], method=0)
    refine3 = crop_and_concat(refine3, seed3)
    refine3 = conv_bn_relu_drop(x=refine3, kernal=[3, 3, 2, 1], phase=phase, drop=drop_conv, scope='refine3_2')
    refine3 = crop_and_concat(refine2, refine3)
    refine3 = conv_bn_relu_drop(x=refine3, kernal=[3, 3, 33, 33], phase=phase, drop=drop_conv, scope='refine3_3')
    # refine3 = squeeze_excitation_model(refine3, out_dim=33, scope='sem2_3')
    refine3 = attention_layer(refine3, scope='attention3')



    out1 = conv_sigmoid(x=refine3, kernal=[1, 1, 33, n_class], scope='out1')

    return out1


def cul_edge(img):
    subtract = np.zeros(shape=(1, 512, 512, 1))
    for i in range(512):
        for j in range(512):
            subtract[0][i][j][0] = 255
            if i == 0 or j == 0 or i == 511 or j == 511:
                if img[0][i][j][0] == 255:
                    subtract[0][i][j][0] = 0
            elif img[0][i][j][0] == 255:
                flag = 0
                if (img[0][i - 1][j][0] == 255 or img[0][i + 1][j][0] == 255 or img[0][i][j - 1][0] == 255 or img[0][i][j + 1][
                    0] == 255 or img[0][i - 1][j - 1][0] == 255 or img[0][i - 1][j + 1][0] == 255 or img[0][i + 1][j + 1][
                    0] == 255 or img[0][i + 1][j - 1][0] == 255):
                    flag = flag + 1
                if (img[0][i - 1][j][0] == 0 or img[0][i + 1][j][0] == 0 or img[0][i][j - 1][0] == 0 or img[0][i][j + 1][
                    0] == 0 or img[0][i - 1][j - 1][0] == 0 or img[0][i - 1][j + 1][0] == 0 or img[0][i + 1][j + 1][
                    0] == 0 or img[0][i + 1][j - 1][0] == 0):
                    flag = flag + 1
                if flag == 2:
                    subtract[0][i][j][0] = 0
                else:
                    subtract[0][i][j][0] = 255
    return subtract


def _next_batch(train_images, train_labels, train_edge, batch_size, index_in_epoch):
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
        train_edge = train_edge[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], train_edge[start:end], index_in_epoch


def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b






def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def boundary_loss(y_true, y_pred):
    y_true_dist_map = tf.py_func(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def dice(y_true, y_pred):
    H, W, C = y_true.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(y_pred, [-1, H * W * C])
    true_flat = tf.reshape(y_true, [-1, H * W * C])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = -tf.reduce_mean(intersection / denominator)
    return loss

class unet2dModule(object):
    """
    A unet2d implementation

    :param image_height: number of height in the input image
    :param image_width: number of width in the input image
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param costname: name of the cost function.Default is "cross_entropy"
    """

    def __init__(self, image_height, image_width, channels=1, costname="boundary_dice"):
        self.image_with = image_width
        self.image_height = image_height
        self.channels = channels

        self.X = tf.placeholder("float", shape=[None, image_height, image_width, channels], name="Input")
        self.Y_gt = tf.placeholder("float", shape=[None, image_height, image_width, 1], name="Output_GT")
        self.edge = tf.placeholder("float", shape=[None, image_height, image_width, 1], name="edge_train")
        self.lr = tf.placeholder('float', name="Learning_rate")
        self.phase = tf.placeholder(tf.bool, name="Phase")
        self.drop_conv = tf.placeholder('float', name="DropOut")
        # self.alpha = tf.placeholder('float', name = "alpha")
        # self.beta = tf.placeholder('float', name = "beta")
        self.layer7 = tf.placeholder("float", shape=[None, int(image_width / 4), int(image_height / 4), 128], name="layer7_out")
        self.layer8 = tf.placeholder("float", shape=[None, int(image_width / 2), int(image_height / 2), 64],
                                     name="layer8_out")
        self.layer9 = tf.placeholder("float", shape=[None, image_height, image_width, 32],
                                     name="layer9_out")
        self.output = tf.placeholder("float", shape=[None, image_height, image_width, 1],
                                     name="train_output")



        self.layer7, self.layer8, self.layer9, self.output = _create_conv_net(self.X, image_width, image_height, channels, self.phase, self.drop_conv)
        self.Y_pred = subtract_refine(self.edge, self.layer7, self.layer8, self.layer9, self.phase, self.drop_conv)


        self.cost = self.__get_cost(costname)
        # self.accuracy = -self.__get_cost(costname)
        self.accuracy = -self.__get_accuracy()

    def dice(self, y_true, y_pred):
        H, W, C = y_true.get_shape().as_list()[1:]
        smooth = 1e-5
        pred_flat = tf.reshape(y_pred, [-1, H * W * C])
        true_flat = tf.reshape(y_true, [-1, H * W * C])

        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
        loss = -tf.reduce_mean(intersection / denominator)
        return loss

    def calc_dist_map(self, seg):
        res = np.zeros_like(seg)
        posmask = seg.astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

        return res

    def calc_dist_map_batch(self, y_true):
        # y_true_numpy = y_true.numpy()
        return np.array([self.calc_dist_map(y)
                         for y in y_true]).astype(np.float32)

    def boundary_loss(self, y_true, y_pred):
        y_true_dist_map = tf.py_func(func=self.calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
        multipled = y_pred * y_true_dist_map
        loss = K.mean(multipled)
        return loss

    def __get_cost(self, cost_name):
        H, W, C = self.Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(self.Y_pred, [-1, H * W * C])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C])


            out_flag = tf.reshape(self.output, [-1, H * W * C])
            intersection_flag = 2 * tf.reduce_sum(out_flag * true_flat, axis=1) + smooth
            denominator_flag = tf.reduce_sum(out_flag, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss_flag = -tf.reduce_mean(intersection_flag / denominator_flag)

            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)

            loss = (loss + loss_flag) / 2
        if cost_name == "pixelwise_cross entroy":
            assert (C == 1)
            flat_logit = tf.reshape(self.Y_pred, [-1])
            flat_label = tf.reshape(self.Y_gt, [-1])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit, labels=flat_label))
        if cost_name == "focal":
            y_pred = tf.clip_by_value(self.Y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            output = tf.clip_by_value(self.output, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            logits = tf.log(y_pred / (1 - y_pred))
            alpha = 0.25
            gamma = 2
            loss1 = focal_loss_with_logits(logits=logits, targets=self.Y_gt, alpha=alpha, gamma=gamma, y_pred=y_pred)
            loss1 = tf.reduce_mean(loss1)
            loss2 = focal_loss_with_logits(logits=logits, targets=self.output, alpha=alpha, gamma=gamma, y_pred=output)
            loss2 = tf.reduce_mean(loss2)
            loss = (loss1 + loss2) / 2
        if cost_name == "boundary":
            loss1 = self.boundary_loss(self.Y_gt, self.output)
            loss2 = self.boundary_loss(self.Y_gt, self.Y_pred)
            loss = (loss1 + loss2) / 2
        if cost_name == "boundary_dice":
            alpha = tf.constant(1.0)
            alpha = tf.Variable(alpha)
            beta = tf.constant(1.0)
            beta = tf.Variable(beta)

            # alpha_before_boundary = tf.constant(1.0)
            # alpha_before_boundary = tf.Variable(alpha_before_boundary)
            # alpha_after_boundary = tf.constant(1.0)
            # alpha_after_boundary = tf.Variable(alpha_after_boundary)
            # beta_before_dice = tf.constant(1.0)
            # beta_before_dice = tf.Variable(beta_before_dice)
            # beta_after_dice = tf.constant(1.0)
            # beta_after_dice = tf.Variable(beta_after_dice)
            loss1_1 = alpha * self.boundary_loss(self.Y_gt, self.Y_pred)
            loss1_2 = alpha * self.dice(self.Y_gt, self.Y_pred)
            loss2_1 = beta * self.boundary_loss(self.Y_gt, self.output)
            loss2_2 = beta * self.dice(self.Y_gt, self.output)
            loss = (loss1_1 + loss1_2 + loss2_1 + loss2_2)
        return loss


    def __get_accuracy(self):
        H, W, C = self.Y_gt.get_shape().as_list()[1:]
        smooth = 1e-5
        pred_flat = tf.reshape(self.Y_pred, [-1, H * W * C])
        true_flat = tf.reshape(self.Y_gt, [-1, H * W * C])

        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
        loss = -tf.reduce_mean(intersection / denominator)
        return loss




    def train(self, train_images, train_labels, train_edges, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=1000, batch_size=2):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=8)

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
            batch_xs_path, batch_ys_path, batch_edge_path, index_in_epoch = _next_batch(train_images, train_labels, train_edges, batch_size,
                                                                       index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_height, self.image_with, self.channels))
            batch_ys = np.empty((len(batch_ys_path), self.image_height, self.image_with, 1))
            batch_edge = np.empty((len(batch_edge_path), self.image_height, self.image_with, 1))

            for num in range(len(batch_xs_path)):
                # image = PIL.Image.open(batch_xs_path[num][0])
                # image = np.asarray(image)
                # label = PIL.Image.open(batch_ys_path[num][0])
                # label = np.asarray(label)
                # edges = PIL.Image.open(batch_edge_path[num][0])
                # edges = np.asarray(edges)

                image = cv2.imread(batch_xs_path[num][0], cv2.IMREAD_COLOR)
                cv2.imwrite('image_src.bmp', image)
                label = cv2.imread(batch_ys_path[num][0], cv2.IMREAD_GRAYSCALE)
                edges = cv2.imread(batch_edge_path[num][0], cv2.IMREAD_GRAYSCALE)

                batch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_with, self.channels))
                batch_ys[num, :, :, :] = np.reshape(label, (self.image_height, self.image_with, 1))
                batch_edge[num, :, :, :] = np.reshape(edges, (self.image_height, self.image_with, 1))


               # print("ffffffffffffffffffff")
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            batch_edge = batch_edge.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
            batch_edge = np.multiply(batch_edge, 1.0 / 255.0)



            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:


                all, train_accuracy = sess.run([self.cost, self.accuracy], feed_dict={self.X: batch_xs,
                                                                                             self.Y_gt: batch_ys,
                                                                                             self.edge: batch_edge,
                                                                                             self.lr: learning_rate,
                                                                                             self.phase: 1,
                                                                                             self.drop_conv: dropout_conv})
                train_loss = all
                # print("before_boundary=", all[1])
                # print("after_boundary=", all[2])
                # print("before_dice=", all[3])
                # print("after_dice=", all[4])
                train_layer7, train_layer8, train_layer9, train_out = sess.run([self.layer7, self.layer8, self.layer9, self.output], feed_dict={self.X: batch_xs,
                                                                                             self.Y_gt: batch_ys,
                                                                                             self.edge: batch_edge,
                                                                                             self.phase: 1,
                                                                                             self.drop_conv: dropout_conv})

                pred = sess.run(self.Y_pred, feed_dict={self.edge: batch_edge,
                                                        self.layer7: train_layer7,
                                                        self.layer8: train_layer8,
                                                        self.layer9: train_layer9,
                                                        self.phase: 1,
                                                        self.drop_conv: 1})


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
                                                                            self.edge: batch_edge,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop_conv: dropout_conv})
            summary_writer.add_summary(summary, i)


            if (i % 10000 == 0) or (i + 1) == train_epochs:
                save_path = saver.save(sess, model_path + str(i))
                print("Model saved in file:", save_path)
        summary_writer.close()

    def prediction(self, model_path, test_images):

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)


        test_img = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], self.channels))
        # subtract_images = np.reshape(subtract_images, (1, test_images.shape[0], test_images.shape[1], 1))

        # test_label = cv2.imread("D:\Data\GlandCeil\Test\Mask\\train_37_anno.bmp", 0)
        # test_label = np.multiply(test_label, 1.0 / 255.0)
        # test_label = np.reshape(test_label, (1, test_label.shape[0], test_label.shape[1], 1))
        test_layer7, test_layer8, test_layer9, test_output = sess.run([self.layer7, self.layer8, self.layer9, self.output],
                                                               feed_dict={self.X: test_img,
                                                                          self.phase: 1,
                                                                          self.drop_conv: 1})


        # subtract_images = cul_edge(test_output)
        # pred = sess.run(self.Y_pred, feed_dict={self.edge: test_output,
        #                                               self.layer7: test_layer7,
        #                                               self.layer8: test_layer8,
        #                                               self.layer9: test_layer9,
        #                                               self.phase: 1,
        #                                               self.drop_conv: 1})
        # print(test_output)
        result = np.reshape(test_output, (test_img.shape[1], test_img.shape[2]))
        result = result.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        tf.reset_default_graph()

        return result