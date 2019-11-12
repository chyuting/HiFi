import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# gpu_options=tf.GPUOptions(allow_growth=True)
gpu_options = tf.GPUOptions(allow_growth=True)


tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 8, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '7', '')

# ICDAR2015 with clear text.
tf.app.flags.DEFINE_string('checkpoint_path', 'east_HiFi2', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 1000, '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'model_vgg_16/vgg_16.ckpt', '')

import model_dice_multi_Vgg16_HiFi12 as model
import icdar_HiFi12 as icdar


FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))


def tower_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)

    # multi-scale
    model_loss = model.loss(score_maps, f_score[2],
                            geo_maps, f_geometry[2],
                            training_masks)
    for scale in range(2):
        resize_factor = 2 ** (2-scale)
        model_loss += model.loss(score_maps[:,::resize_factor,::resize_factor,:], f_score[scale],
                                geo_maps[:,::resize_factor,::resize_factor,:], f_geometry[scale],
                                training_masks[:,::resize_factor,::resize_factor,:])

    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    # if reuse_variables is None:
    # 	tf.summary.image('input', images)
    # 	tf.summary.image('score_map', score_maps)
    # 	tf.summary.image('score_map_pred', f_score[2] * 255)
    # 	tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
    # 	tf.summary.image('geo_map_0_pred', f_geometry[2][:, :, :, 0:1])
    # 	tf.summary.image('training_masks', training_masks)
    # 	tf.summary.scalar('model_loss', model_loss)
    # 	tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os

    fp_loss = open("loss_record.txt", "w+")
    fp_loss.truncate()

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    # else:
    #	  if not FLAGS.restore:
    #		  tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
    #		  tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss = tower_loss(iis, isms, igms, itms, reuse_variables)  # CAI Note: calculate loss.
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        # data_generator = icdar.generator(input_size=FLAGS.input_size,
        #                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_score_maps: data[2],
                                                                                input_geo_maps: data[3],
                                                                                input_training_masks: data[4]})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus)) / (time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, ml, tl, avg_time_per_step, avg_examples_per_second))

                fp_loss.write('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second\n'.format(
                step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_path, 'model.ckpt'), global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4]})
                summary_writer.add_summary(summary_str, global_step=step)
        fp_loss.closs()


if __name__ == '__main__':
    tf.app.run()
