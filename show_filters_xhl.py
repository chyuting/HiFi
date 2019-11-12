#encoding:utf-8

import tensorflow as tf
import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'east_HiFi2/', '')
tf.app.flags.DEFINE_string('test_data_path', 'data/icdar2015/test/img_2.jpg', '')

import model_HiFi12_show as model


FLAGS = tf.app.flags.FLAGS

# def get_images():
#     '''
#     find image files in test data path
#     :return: list of files found
#     '''
#     files = []
#     exts = ['jpg', 'png', 'jpeg']
#     for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
#         for filename in filenames:
#             for ext in exts:
#                 if filename.endswith(ext):
#                     files.append(os.path.join(parent, filename))
#                     break
#     print('Find {} images'.format(len(files)))
#     return files


def resize_image(im, max_side_len=1024):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio    2400
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    # limit the min side
    min_side_len = 640
    if min(resize_h, resize_w) < min_side_len:
        ratio = float(min_side_len) / max(resize_h, 32) if resize_h < resize_w else float(min_side_len) / max(resize_w,
                                                                                                              32)
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)


    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))



    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry, feature_maps, weight_list = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            #im_fn_list = get_images()
            #for im_fn in im_fn_list:S
            im_fn = FLAGS.test_data_path
            im = cv2.imread(im_fn)[:, :, ::-1]  # ?
            im_resized, (ratio_h, ratio_w) = resize_image(im)

            score, geometry, feature_maps_output, weight_list_output = sess.run([f_score, f_geometry, feature_maps, weight_list], feed_dict={input_images: [im_resized]})

            weight_array = []
            weight_array_sort = []  # index
            for scale in range(3):
                temp = []
                temp_abs = []
                for i in range(32):
                    temp.append(weight_list_output[scale][0][0][i][0])
                    #temp_abs.append(abs(weight_list_output[scale][0][0][i][0]))
                weight_array.append(temp)
                weight_array_sort.append(np.argsort(np.array(temp)))

            #show together
            for scale in range(3):
                fig, ax = plt.subplots(nrows=8, ncols=4)
                for i in range(8):
                    for j in range(4):
                        temp = feature_maps_output[scale][0, :, :, weight_array_sort[scale][31 - (i * 4 + j)]]
                        # if weight_array[scale][weight_array_sort[scale][31-(i*4+j)]] < 0:
                        #     temp = temp*(-1)
                        # temp = np.maximum(temp, 0)  # relu, do not need, feature map values are all positive
                        ax[i, j].imshow(temp, cmap=plt.cm.gray)  # tensor [batch, channels, row, column]
                        ax[i, j].set_title(str(weight_array[scale][weight_array_sort[scale][31-(i*4+j)]]))
                        #ax[i, j].set_title(str(weight_array[scale][31 - (i * 4 + j)]))


                plt.show()

if __name__ == '__main__':
    tf.app.run()


