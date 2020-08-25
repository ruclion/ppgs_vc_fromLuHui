import tensorflow as tf
import argparse
from datetime import datetime
import time
import os
import sys

from jessa_dataset import get_iterator
from models import ConversionModelV3

# some super parameters
BATCH_SIZE = 16
STEPS = int(1e5)
LEARNING_RATE = 1e-3
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_RATE = 16000
MAX_TO_SAVE = 10
CKPT_EVERY = 500
LC_DIM = 347
# STFT_DIM = 513
NUM_MEL = 80
DROP_RATE = 0.5


def get_arguments():
    parser = argparse.ArgumentParser(description="WaveNet training script")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--drop_rate', type=float, default=DROP_RATE)
    parser.add_argument('--output-model-path', dest='output_model_path',
                        required=True, type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help='Philly model output path.')
    parser.add_argument('--log-dir', dest='log_dir', required=True, type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help='Philly log dir.')
    parser.add_argument('--restore_from', type=str, default=None)
    parser.add_argument('--max_ckpts', type=int, default=MAX_TO_SAVE)
    parser.add_argument('--ckpt_every', type=int, default=CKPT_EVERY)
    return parser.parse_args()


def save_model(saver, sess, logdir, step):
    model_name = 'vc.ckpt'
    ckpt_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, ckpt_path, global_step=step)
    print('Done')


def load_model(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


# def stft2wav(spec, mean_f=STFT_MEAN_F, std_f=STFT_STD_F):
# from audio import log_power_denormalize_tf, db2power_tf, grinffin_lim_tf
#     # get inputs
#     mean = np.load(mean_f)
#     std = np.load(std_f)
#     spec = spec * std + mean
#     # build graph
#     denormalized = log_power_denormalize_tf(spec)
#     mag_spec = tf.pow(db2power_tf(denormalized), 0.5)
#     wav = grinffin_lim_tf(mag_spec)
#     return wav


def main():
    args = get_arguments()

    model_dir = args.output_model_path
    logdir = args.log_dir
    train_dir = os.path.join(logdir, STARTED_DATESTRING, 'train')
    dev_dir = os.path.join(logdir, STARTED_DATESTRING, 'dev')
    os.makedirs(train_dir)
    os.makedirs(dev_dir)
    restore_dir = args.restore_from

    # load data
    train_set = tf.data.Dataset.from_generator(
        get_iterator('train'),
        output_types=(tf.float32, tf.float32, tf.int32),
        output_shapes=([None, LC_DIM], [None, NUM_MEL], []))
    train_set = train_set.padded_batch(args.batch_size,
                                       ([None, LC_DIM],
                                        [None, NUM_MEL], [])
                                       ).repeat()
    train_ds_iter = train_set.make_initializable_iterator()

    dev_set = tf.data.Dataset.from_generator(
        get_iterator('dev'),
        output_types=(tf.float32, tf.float32, tf.int32),
        output_shapes=([None, LC_DIM], [None, NUM_MEL], []))
    dev_set = dev_set.padded_batch(args.batch_size,
                                   ([None, LC_DIM],
                                    [None, NUM_MEL], [])
                                   ).repeat()
    dev_ds_iter = dev_set.make_initializable_iterator()
    dataset_handle = tf.placeholder(tf.string, shape=[])
    dataset_iter = tf.data.Iterator.from_string_handle(
        dataset_handle,
        train_set.output_types,
        train_set.output_shapes
    )
    batch_data = dataset_iter.get_next()

    # create network
    # conversion_net = ConversionModelV5(lstm_hidden=512, proj_dim=256, out_dim=NUM_MEL, drop_rate=DROP_RATE)
    conversion_net = ConversionModelV3(out_dim=NUM_MEL, drop_rate=DROP_RATE, is_train=True)
    out_dict = conversion_net(inputs=batch_data[0],
                              lengths=batch_data[2],
                              targets=batch_data[1])
    loss = out_dict['loss']
    predicts = out_dict['out']
    # wav_predict = stft2wav(predicts)
    # wav_gt = stft2wav(batch_data[1])
    # tf.summary.audio('generated', wav_predict, sample_rate=SAMPLE_RATE)
    # tf.summary.audio('resyn_ground_truth', wav_gt, sample_rate=SAMPLE_RATE)
    tf.summary.scalar('loss', loss)
    tf.summary.image(
        'predictions',
        tf.transpose(tf.expand_dims(predicts, axis=-1), [0, 2, 1, 3]),
        max_outputs=1)
    tf.summary.image(
        'ground_truth',
        tf.transpose(tf.expand_dims(batch_data[1], axis=-1), [0, 2, 1, 3]),
        max_outputs=1)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4)
    optim = optimizer.minimize(loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optim = tf.group([optim, update_ops])

    # Set up logging for TensorBoard.
    train_writer = tf.summary.FileWriter(train_dir)
    train_writer.add_graph(tf.get_default_graph())
    dev_writer = tf.summary.FileWriter(dev_dir)
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=args.max_ckpts)

    # set up session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run([train_ds_iter.initializer, dev_ds_iter.initializer])
    train_handle, dev_handle = sess.run([train_ds_iter.string_handle(),
                                         dev_ds_iter.string_handle()])
    # try to load saved model
    try:
        if restore_dir is None:
            saved_global_step = -1
        else:
            saved_global_step = load_model(saver, sess, restore_dir)
            if saved_global_step is None:
                saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    last_saved_step = saved_global_step
    step = None
    try:
        for step in range(saved_global_step + 1, args.steps):
            start_time = time.time()
            if step % args.ckpt_every == 0 and step != 0:
                summary, loss_value = sess.run([summaries, loss],
                                               feed_dict={dataset_handle: dev_handle})
                dev_writer.add_summary(summary, step)
                duration = time.time() - start_time
                print('step {:d} - eval loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
                save_model(saver, sess, model_dir, step)
                last_saved_step = step
            else:
                summary, loss_value, _ = sess.run([summaries, loss, optim],
                                                  feed_dict={dataset_handle: train_handle})
                train_writer.add_summary(summary, step)
                duration = time.time() - start_time
                print('step {:d} - training loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save_model(saver, sess, model_dir, step)
    sess.close()


if __name__ == '__main__':
    main()
