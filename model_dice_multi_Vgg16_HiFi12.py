# coding=utf-8
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets_tensorflow_HiFi import vgg
# from focal_loss import *

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def FusionCh(conv1, conv2, channels ):
    '''
    Hierarchical feature integration inspired by HiFi.
    顺序concat和 交叉 concat 应该是不一样的.
    :param conv1: 
    :param conv2: 
    :param channels: the number of channels which is required to transform by 1*1 convolutional layer.
    :return: 
    '''
    conv1_Fu = slim.max_pool2d(conv1[0], [2, 2])
    for ch in conv1[1:]:
        ch = slim.max_pool2d(ch, [2, 2] )
        conv1_Fu = tf.concat([conv1_Fu,ch], axis=-1)

    conv2_Fu = slim.conv2d(conv2[0], channels, 1)
    for ch in conv2[1:]:
        ch = slim.conv2d(ch, channels, 1)
        conv2_Fu = tf.concat([conv2_Fu,ch], axis=-1)

    conv12_Fu = tf.concat([conv1_Fu, conv2_Fu], axis=-1)

    conv12_Fu = slim.conv2d(conv12_Fu, channels, 3)

    return conv12_Fu


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
        logits, end_points= vgg.vgg_16(images, is_training=is_training, scope='vgg_16')

    # with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
    #     logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            #First stage of HiFi
            conv1 = [end_points["conv1_1"], end_points["conv1_2"]]
            conv2 = [end_points["conv2_1"], end_points["conv2_2"]]
            conv3 = [end_points["conv3_1"], end_points["conv3_2"], end_points["conv3_3"]]
            conv4 = [end_points["conv4_1"], end_points["conv4_2"], end_points["conv4_3"]]
            conv5 = [end_points["conv5_1"], end_points["conv5_2"], end_points["conv5_3"]]

            Hi_12 = FusionCh(conv1, conv2, 64)
            Hi_23 = FusionCh(conv2, conv3, 128)
            Hi_34 = FusionCh(conv3, conv4, 256)
            Hi_45 = FusionCh(conv4, conv5, 512)

            # Second stage of HiFi
            Hi_123 = slim.conv2d(tf.concat([slim.max_pool2d(Hi_12, [2, 2]), Hi_23], axis=-1), 64, 3)
            Hi_234 = slim.conv2d(tf.concat([slim.max_pool2d(Hi_23, [2, 2]), Hi_34], axis=-1), 128, 3)
            Hi_345 = slim.conv2d(tf.concat([slim.max_pool2d(Hi_34, [2, 2]), Hi_45], axis=-1), 256, 3)

            # Arrange from deep to shallow
            HiFi_2 = [Hi_345, Hi_234, Hi_123]

            # f = [end_points['pool5'], end_points['pool4'],
            #      end_points['pool3'], end_points['pool2']]
            f = HiFi_2

            for i in range(3):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None]
            h = [None, None, None]
            num_outputs = [128, 64, 32]
            for i in range(3):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 1:
                    g[i] = unpool(h[i])
                # FIXME Cai: this "else" lead to the failure of load other models from tensorflow of github,e.g.,resnet,vgg.
                # else:
                #     g[i] = slim.conv2d(h[i], num_outputs[i], 3) # This step is not effective for the structure.
                # print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            F_score, F_geometry = [], []
            # # multi scale
            for scale in range(3):
                # with tf.variable_scope('out_'+str(scale)):
                m = slim.conv2d(h[scale], 32, 3)
                # m = slim.conv2d(h[scale], num_outputs[3], 3)
                score_map = slim.conv2d(m, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(m, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
                angle_map = (slim.conv2d(m, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                geo = tf.concat([geo_map, angle_map], axis=-1)
                F_score.append(score_map)
                F_geometry.append(geo)

    return F_score, F_geometry


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)

    return loss


def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''

    # classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask, resize_factor)
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
