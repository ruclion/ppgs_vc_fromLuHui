import tensorflow as tf
import argparse
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm

from data_reader import all_data_generator
from models import ConversionModelV1
from audio import log_power_denormalize, db2power

# some super parameters
BATCH_SIZE = 16
RESTORE_FROM = ''
SAVE_ROOT = './predicted_stft-513'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
STFT_MEAN_F = '../share/zhiling/statistics/stft-513/mean.npy'
STFT_STD_F = '../share/zhiling/statistics/stft-513/std.npy'
SAMPLE_RATE = 16000
LC_DIM = 132
STFT_DIM = 513
DROP_RATE = 0.0


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description="WaveNet training script")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--drop_rate', type=float, default=DROP_RATE)
    parser.add_argument('--restore_from', type=str, default=RESTORE_FROM)
    parser.add_argument('--overwrite', type=_str_to_bool, default=True,
                        help='Whether to overwrite the old model ckpt,'
                             'valid when restore_from is not None')
    return parser.parse_args()


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


def save_stft_batch(spec, save_root, save_names, lengths, mean_f=STFT_MEAN_F, std_f=STFT_STD_F):
    """
    :param spec: [batch, time, dims]
    :param save_names: a list of filename, [batch, ]
    :param lengths: a list of time lengths, [batch, ]
    :param mean_f: stft mean file
    :param std_f: stft std file
    :return:
    """
    assert len(spec.shape) == 3
    batch_size = spec.shape[0]
    mean = np.load(mean_f)    # [dims, ]
    std = np.load(std_f)      # [dims, ]
    spec = spec * std + mean  # [batch, time, dims]
    spec = log_power_denormalize(spec)  # [batch, time, dims]
    power_specs = db2power(spec)  # [batch, time, dims]
    mag_specs = power_specs ** 0.5
    for i in range(batch_size):
        mag = mag_specs[i, : lengths[i], :]
        np.save(os.path.join(save_root, save_names[i]), mag)
    return


def main():
    args = get_arguments()

    # load data
    data_set = tf.data.Dataset.from_generator(
        all_data_generator,
        output_types=(tf.float32, tf.float32, tf.int32, tf.string),
        output_shapes=([None, LC_DIM], [None, STFT_DIM], [], []))
    data_set = data_set.padded_batch(args.batch_size,
                                     ([None, LC_DIM], [None, STFT_DIM],
                                      [], []))

    # create network
    ppgs_pl = tf.placeholder(dtype=tf.float32, shape=[None, None, LC_DIM],
                             name='ppgs_pl')
    stft_pl = tf.placeholder(dtype=tf.float32, shape=[None, None, STFT_DIM],
                             name='stft_pl')
    lengths_pl = tf.placeholder(dtype=tf.float32, shape=[None], name='lengths_pl')
    conversion_net = ConversionModelV1(out_dim=STFT_DIM,
                                       drop_rate=args.drop_rate,
                                       is_train=True)
    out_dict = conversion_net(inputs=ppgs_pl,
                              lengths=lengths_pl,
                              targets=stft_pl)
    predicts = out_dict['out']

    # Set up
    loader = tf.train.Saver(max_to_keep=args.max_ckpts)

    # set up session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # try to load saved model
    print('Restoring model from {}'.format(args.ckpt))
    loader.restore(sess, args.ckpt)
    for batch_data in tqdm(data_set):
        preds = sess.run(predicts,
                         feed_dict={ppgs_pl: batch_data[0],
                                    lengths_pl: batch_data[2],
                                    stft_pl: batch_data[1]})
        save_stft_batch(preds, save_root=SAVE_ROOT,
                        save_names=batch_data[3],
                        lengths=batch_data[2])
    sess.close()


if __name__ == '__main__':
    main()
