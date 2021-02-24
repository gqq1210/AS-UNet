import tensorflow as tf
import numpy as np


import time
import datetime
Y_M_D_H_M_S = '%Y-%m-%d %H:%M:%S'
Y_M_D = '%Y-%m-%d'
H_M_S = '%H:%M:%S'

# 获取系统当前时间并转换请求数据所需要的格式
def get_time(format_str):
    # 获取当前时间的时间戳
    now = int(time.time())
    # 转化为其他日期格式， 如 “%Y-%m-%d %H:%M:%S”
    timeStruct = time.localtime(now)
    strTime = time.strftime(format_str, timeStruct)
    return strTime

# 获取时分秒时间格式字符串
def get_hms_time():
    return get_time(H_M_S)




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


def max_pooling_with_argmax(x):
    """
    :param x: 特征图x
    :return: 做完maxpool的特征图pool，对应值位置信息xindex，yindex，大小和pool一样，用来表示pool所提取值在原图中的位置
    """

    # dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化
    # print("begin: %s" % dt_ms)
    _, row, col, channel = x.get_shape().as_list()
    # 函数只支持GPU操作,indices的值是将整个数组flat后的索引，并保持与池化结果一致的shape
    pool, pool_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME')

    row_col_channel = row * col * channel
    row_col = row * col
    # print(xindex.shape)
    xindex = tf.to_float(pool_indices)
    xindex = xindex - (xindex // row_col_channel) * row_col_channel   #转换到一个batch上
    xindex = xindex - (xindex // row_col) * row_col   #转换到一个通道上
    a_x = xindex // row
    a_y = xindex - (xindex//row) * row
    a_x = (a_x % 2) * 2 - 1
    a_y = (a_y % 2) * 2 - 1
    # a_x = tf.to_float(a_x)
    # a_y = tf.to_float(a_y)
    # print(a_x.shape)
    # print(a_y)
    # dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化
    # print("end: %s" % dt_ms)
    return pool, a_x, a_y


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


def conv_bn_relu(x, kernel, phase, strides = 1, scope = None):
    with tf.name_scope(scope): #scope表示一个操作范围，在这个命名下的文件都是以这个开头的
        W = weight_xavier_init(shape = kernel,n_inputs = kernel[0] * kernel[1] * kernel[2], n_outputs=kernel[-1],
                               activefunction = 'relu',variable_name = scope + 'conv_W')  #n_inputs就是卷积核的大小，n_outputs就是通道数
        B = bias_variable([kernel[-1]],variable_name = scope + 'conv_B')
        conv = conv2d(x, W, strides) + B
        conv = normalizationlayer(conv,is_train = phase,norm_type = 'batch',scope = scope + 'normalization')
        conv = tf.nn.leaky_relu(conv)
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

def unsampling(x, size = 2):
    _, width, height, channel = x.get_shape().as_list()
    out = tf.image.resize_images(images=x, size=[int(width * size), int(height * size)], method=0)  #默认为双线性插值
    return out

def conv_unsampling_bn_relu(x, kernel, size, phase, scope = None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2], n_outputs=kernel[-1],
                               activefunction='relu', variable_name=scope + 'conv_W')  # n_inputs就是卷积核的大小，n_outputs就是通道数
        B = bias_variable([kernel[-1]], variable_name=scope + 'conv_B')
        conv = conv2d(x, W) + B
        unsamp = unsampling(conv, size)
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




def dilated_conv2D_relu(inputs,kernel,rate, scope = None):
    with  tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2],
                               n_outputs=kernel[-1], activefunction='sigmoid', variable_name=scope + 'W')
        B = bias_variable([kernel[-1]], variable_name=scope + 'B')
        inputs = tf.nn.atrous_conv2d(inputs, W, rate=rate, padding='SAME') + B
        conv = tf.nn.relu(inputs)
        return conv



class CoordConv2D:
    #coord_conv = CoordConv2D(1, 32, 1, activation=tf.nn.leaky_relu)
    #output = coord_conv(x)
    #对x做卷积核大小1*1，输出通道数为32，stride为2的CoordConv
    def __init__(self, k_size, filters,
                 strides=1, padding='same',
                 with_r=False, activation=None,
                 kernel_initializer=None, name=None):

        self.with_r = with_r

        self.conv_kwargs = {
            'filters': filters,
            'kernel_size': k_size,
            'strides': strides,
            'padding': padding,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'name': name
        }

    def __call__(self, in_tensor):
        with tf.name_scope('coord_conv'):
            batch_size = tf.shape(in_tensor)[0]
            x_dim = tf.shape(in_tensor)[1]
            y_dim  = tf.shape(in_tensor)[2]

            xx_indices = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.range(x_dim), 0), 0),
                [batch_size, y_dim, 1])
            xx_indices = tf.expand_dims(xx_indices, -1)

            yy_indices = tf.tile(
                tf.expand_dims(tf.reshape(tf.range(y_dim), (y_dim, 1)), 0),
                [batch_size, 1, x_dim])
            yy_indices = tf.expand_dims(yy_indices, -1)

            xx_indices = tf.divide(xx_indices, x_dim - 1)
            yy_indices = tf.divide(yy_indices, y_dim - 1)

            xx_indices = tf.cast(tf.subtract(tf.multiply(xx_indices, 2.), 1.),
                                 dtype=in_tensor.dtype)
            yy_indices = tf.cast(tf.subtract(tf.multiply(yy_indices, 2.), 1.),
                                 dtype=in_tensor.dtype)

            processed_tensor = tf.concat([in_tensor, xx_indices, yy_indices], axis=-1)

            if self.with_r:
                rr = tf.sqrt(tf.add(tf.square(xx_indices - 0.5),
                                    tf.square(yy_indices - 0.5)))
                processed_tensor = tf.concat([processed_tensor, rr], axis=-1)

            return tf.layers.conv2d(processed_tensor, **self.conv_kwargs)


def atrous_spatial_pyramid_pooling(inputs, filters=256, regularizer=None):  # ASPP层
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''
    pool_height = tf.shape(inputs)[1]
    pool_width = tf.shape(inputs)[2]

    resize_height = pool_height
    resize_width = pool_width

    # Atrous Spatial Pyramid Pooling
    # Atrous 1x1
    aspp1x1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1, 1),
                               padding='same', kernel_regularizer=regularizer,
                               name='aspp1x1')
    # Atrous 3x3, rate = 6
    aspp3x3_1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),
                                 padding='same', dilation_rate=(12, 12), kernel_regularizer=regularizer,
                                 name='aspp3x3_1')
    # Atrous 3x3, rate = 12
    aspp3x3_2 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),
                                 padding='same', dilation_rate=(24, 24), kernel_regularizer=regularizer,
                                 name='aspp3x3_2')
    # Atrous 3x3, rate = 18
    aspp3x3_3 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),
                                 padding='same', dilation_rate=(36, 36), kernel_regularizer=regularizer,
                                 name='aspp3x3_3')
    # Image Level Pooling
    image_feature = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    image_feature = tf.layers.conv2d(inputs=image_feature, filters=filters, kernel_size=(1, 1),
                                     padding='same')
    image_feature = tf.image.resize_bilinear(images=image_feature,
                                             size=[resize_height, resize_width],
                                             align_corners=True, name='image_pool_feature')
    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature],
                        axis=3, name='aspp_pools')
    outputs = tf.layers.conv2d(inputs=outputs, filters=filters, kernel_size=(1, 1),
                               padding='same', kernel_regularizer=regularizer, name='aspp_outputs')

    return outputs

