import tensorflow as tf
import numpy as np


def weight_xavier_init(shape,n_inputs,n_outputs,activefunction = 'sigmoid',uniform = True,variable_name = None):
    if activefunction == 'sigmoid':
        if uniform:  #编码
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name = variable_name,initializer=initial, trainable = True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs))
            initial = tf.truncated_normal(shape,mean = 0.0,stddev = stddev)
            return tf.get_variable(name = variable_name,initializer=initial,trainable=True)
    elif activefunction == 'relu':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name = variable_name,initializer=initial,trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_outputs + n_inputs)) * np.sqrt(2)
            initial = tf.truncated_normal(shape,mean = 0.0,stddev = stddev)
            return tf.get_variable(name = variable_name,initializer=initial,trainable=True)


def bias_variable(shape, variable_name=None):
    initial = tf.constant(0.1, shape=shape)  # constant定义shape形状的数组
    return tf.Variable(initial, name=variable_name)


def conv2d(x, W, strides=1):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    return conv_2d


def deconv2d(x, W, strides=2):  # 上采样
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * strides, x_shape[2] * strides, x_shape[3] // strides])  # 张量拼接函数
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding="SAME")  # 反卷积


# x_shape 四维 样本数，图像高度，图像宽度，图像通道数（减半）
# tf.nn.conv2d_transpose(value,filter,output_shape,stride,padding = "SAME",data_format = "NHWC",name = None)
# value:tensor 张量
# filter:tensor 张量,[filter_height,filter_width,out_channels,in_channels]  在卷积中的卷积核是先in，再out
# output_shape:卷积中没有的

def max_pooling_2x2(x):
    pool2d = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return pool2d

def average_pooling_2x2(x):
    pool2d = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    return pool2d

def max_pool_2x2(x):
    pool2d = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return pool2d

def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets = [0, a , b , 0]
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    # size = [-1, c, d, -1]
    size = [-1, x2_shape[1], x2_shape[1], -1]
    # 从（a,b）位置开始减大小为（c,d）的矩形
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)  # 3表示通道上的叠加



def concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets = [0, a , b , 0]
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    # size = [-1, c, d, -1]
    size = [-1, x2_shape[1], x2_shape[1], -1]
    # 从（a,b）位置开始减大小为（c,d）的矩形
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 0)  # 3表示通道上的叠加

def conv_softmax(x,kernal,scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal,n_inputs = kernal[0] * kernal[1] *kernal[2],
                               n_outputs = kernal[-1],activefunction = 'sigmoid',variable_name=scope + 'W')
        B = bias_variable([kernal[-1]],variable_name=scope + 'B')
        conv = conv2d(x,W) + B
        conv = tf.nn.softmax(conv)
        return conv


def down_sampling(x,kernal,phase,drop,scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal,n_inputs = kernal[0] * kernal[1] * kernal[2],
                               n_outputs = kernal[-1],activefunction='relu',variable_name = scope + 'W')
        B = bias_variable([kernal[-1]],variable_name = scope + 'B')
        conv = conv2d(x,W,2) + B #步长改为2
        conv = normalizationlayer(conv,is_train = phase,
                                  norm_type = 'batch',scope = scope)
        conv = tf.nn.dropout(tf.nn.leaky_relu(conv),drop)
        return conv


def deconv_relu(x, kernal, scope = None):
    """
    deconv + relu
    """
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal,n_inputs=kernal[0] * kernal[1] * kernal[-1],
                               n_outputs=kernal[-2],activefunction = 'relu',variable_name=scope + 'W')
        B = bias_variable([kernal[-2]],variable_name = scope + 'B')
        deconv = deconv2d(x, W) + B
        deconv = tf.nn.leaky_relu(deconv)
        return deconv



def resnet_ADD(x1,x2):
    if x1.get_shape().as_list()[3] != x2.get_shape().as_list()[3]:    #如果两图大小不一样
        residual_connection = x2 + tf.pad(x1,[[0,0],[0,0],[0,0],
                                              [0,x2.get_shape().as_list()[3] - x1.get_shape().as_list()[3]]])
    else:
        residual_connection = x2 + x1
    return residual_connection


def normalizationlayer(x,is_train,norm_type = None,scope = None):
    """
         x --- input data with shape of [batch, depth, height, width, channel]
         is_training --- flag of normalization, True is training, False is Testing
         height --- in some condition, the data height is in RunTime datamined, such as through deconv layer and conv2d
         width --- as last
         norm_type --- normalization type: support -- "batch" and "group" "None"
         G --- in group normalization, channel is seperated with group number(G)
         esp --- Prevent divisop from being zero
         scope --- normalizationlayer scope
    """
    with tf.name_scope(scope + norm_type):  #with的意思就是接下来的操作都做一遍
        if norm_type == None:
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(x,center = True,scale = True,is_training = is_train)
            #center --> 如果True,有beta偏移量，反之亦然
            #scale --> 如果为True，则乘以gama，反之亦然；当下一层是线性时，由于缩放可以下一层完成，所以可以禁用，即center和scale都设为False
            #is_training --> 图层是否处于训练模式。在训练模式下，它将累积转入的统计量moving_mean和moving_variance，使用给定的指数移动平均值delay。
            #当它不是训练模式时，那么它将使用的数值moving_mean和moving_variance
            #训练时，需要更新moving_mean和moving_variance（均值和方差）
        return output


def conv_bn_relu_drop(x, kernal, phase, drop, scope = None):
    with tf.name_scope(scope): #scope表示一个操作范围，在这个命名下的文件都是以这个开头的
        W = weight_xavier_init(shape = kernal,n_inputs = kernal[0] * kernal[1] * kernal[2], n_outputs=kernal[-1],
                               activefunction = 'relu',variable_name = scope + 'conv_W')  #n_inputs就是卷积核的大小，n_outputs就是通道数
        B = bias_variable([kernal[-1]],variable_name = scope + 'conv_B')
        conv = conv2d(x,W) + B
        conv = normalizationlayer(conv,is_train = phase,norm_type = 'batch',scope = scope + 'normalization')
        conv = tf.nn.dropout(tf.nn.leaky_relu(conv),drop, name = scope + 'conv_dropout')
        return conv

def conv_maxpooling_bn_relu(x, kernel, phase, scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2], n_outputs=kernel[-1],
                               activefunction='relu', variable_name=scope + 'conv_W')  # n_inputs就是卷积核的大小，n_outputs就是通道数
        B = bias_variable([kernel[-1]], variable_name=scope + 'conv_B')
        conv = conv2d(x, W) + B
        pool2d = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        out = normalizationlayer(pool2d, is_train=phase, norm_type='batch', scope=scope + 'normalization')
        out = tf.nn.leaky_relu(out)
        return out

def upsampling(x):
    _, width, height, channel = x.get_shape().as_list()
    out = tf.image.resize_images(images=x, size=[1, width * 2, height * 2, 1])  #默认为双线性插值
    return out

def conv_unsampling_bn_relu(x, kernel, phase, scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2], n_outputs=kernel[-1],
                               activefunction='relu', variable_name=scope + 'conv_W')  # n_inputs就是卷积核的大小，n_outputs就是通道数
        B = bias_variable([kernel[-1]], variable_name=scope + 'conv_B')
        conv = conv2d(x, W) + B
        unsamp = upsampling(conv)
        out = normalizationlayer(unsamp, is_train=phase, norm_type='batch', scope=scope + 'normalization')
        out = tf.nn.leaky_relu(out)
        return out





def conv_sigmoid(x,kernal,scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal,n_inputs = kernal[0] * kernal[1] *kernal[2],
                               n_outputs = kernal[-1],activefunction = 'sigmoid',variable_name=scope + 'W')
        B = bias_variable([kernal[-1]],variable_name=scope + 'B')
        conv = conv2d(x,W) + B
        conv = tf.nn.sigmoid(conv)
        return conv

def conv_relu(x,kernal,scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape = kernal,n_inputs = kernal[0] * kernal[1] *kernal[2],
                               n_outputs = kernal[-1],activefunction = 'sigmoid',variable_name=scope + 'W')
        B = bias_variable([kernal[-1]],variable_name=scope + 'B')
        conv = conv2d(x,W) + B
        conv = tf.nn.relu(conv)
        return conv


