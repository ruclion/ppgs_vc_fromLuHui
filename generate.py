import tensorflow as tf
import numpy as np
import argparse
import time
import os

from models import ConversionModelV3
from audio_ import inv_mel_spectrogram, save_wav, inv_preemphasize
from jessa_dataset import make_local_cond, softmax, make_local_cond_min_max_norm_f0

DATA_PATH = '/notebooks/projects/share/zhiling'
TARGET = '00001172'
SAVE_DIR = 'test_results'
CKPT = './saved_models/vc.ckpt-57000'
SAVE_NAME = os.path.join(SAVE_DIR, TARGET + CKPT[22:] + '.wav')
STFT_MEAN_F = '/notebooks/projects/share/zhiling/statistics/stft-513/mean.npy'
STFT_STD_F = '/notebooks/projects/share/zhiling/statistics/stft-513/std.npy'
LC_DIM = 347
# STFT_DIM = 513
SAMPLE_RATE = 16000
WIN_LEN = 400
HOP_LEN = 80
N_FFT = 512
NUM_MELS = 80
# for 250
# SRC_LF0_MEAN = 5.184465944788293
# SRC_LF0_STD = 0.2898044753197573
# for 201
# SRC_LF0_MEAN = 4.715685441351366
# SRC_LF0_STD = 0.26241882898012947
# for use voice
# SRC_LF0_MEAN = 5.0501071571800775
# SRC_LF0_STD = 0.1971468485298694
# for jessa
# SRC_LF0_MEAN = 5.304005892546428
# SRC_LF0_STD = 0.22764796667864182

SRC_LF0_MEAN = None
SRC_LF0_STD = None

SRC_LF0_MIN = 3.592402458190918
SRC_LF0_MAX = 6.201971530914307

def get_inputs(ppgs_path, lf0s_path, lf0_mean=None, lf0_std=None):
    ppgs = np.load(ppgs_path)
    lf0s = np.fromfile(lf0s_path, dtype=np.float32)
    inputs = make_local_cond(ppgs, lf0s, lf0_mean=lf0_mean, lf0_std=lf0_std)
    return inputs


def get_ppgs(ppgs_path):
    ppgs = np.load(ppgs_path)
    ppgs = softmax(ppgs)
    return ppgs


def get_inputs_min_max_norm_lf0(ppgs_path, lf0s_path, lf0_min=None, lf0_max=None):
    ppgs = np.load(ppgs_path)
    lf0s = np.fromfile(lf0s_path, dtype=np.float32)
    inputs = make_local_cond_min_max_norm_f0(ppgs, lf0s, lf0_min=lf0_min, lf0_max=lf0_max)
    return inputs

# def stft2wav(spec, save_name, mean_f=STFT_MEAN_F,
#              std_f=STFT_STD_F, sr=SAMPLE_RATE):
#     mean = np.load(mean_f)
#     std = np.load(std_f)
#     spec = spec * std + mean
#     spec = log_power_denormalize(spec)
#     power_spec = db2power(spec)
#     mag_spec = power_spec ** 0.5
#     y = griffin_lim(mag_spec)
#     y = deemphasis(y)
#     write_wav(save_name, y, sr)
#     return y


def get_arguments():
    parser = argparse.ArgumentParser(description="PPGs_VC generation script")
    parser.add_argument('--ppgs', type=str,
                        default=os.path.join(DATA_PATH, 'ppgs_luhui', TARGET + '.npy'))
    parser.add_argument('--f0s', type=str,
                        default=os.path.join(DATA_PATH, 'lf0_raw', TARGET + '.lf0'))
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE)
    parser.add_argument('--save_name', type=str,
                        default=SAVE_NAME)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    return parser.parse_args()


def main():
    args = get_arguments()

    # Set up network
    inputs_pl = tf.placeholder(tf.float32,
                               [None, None, LC_DIM],
                               'inputs')

    # create network
    # conversion_net = ConversionModelV5(lstm_hidden=512, proj_dim=256, out_dim=NUM_MELS, drop_rate=0.0)
    conversion_net = ConversionModelV3(out_dim=NUM_MELS, drop_rate=0., is_train=False)
    outputs = conversion_net(inputs=inputs_pl,
                             lengths=None,
                             targets=None)['out']

    # read data
    # inputs = get_inputs(args.ppgs, args.f0s, SRC_LF0_MEAN, SRC_LF0_STD)
    inputs = get_inputs_min_max_norm_lf0(args.ppgs, args.f0s, lf0_min=SRC_LF0_MIN, lf0_max=SRC_LF0_MAX)
    # inputs = get_ppgs(args.ppgs)

    # set up a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    start_time = time.time()

    # load saved model
    saver = tf.train.Saver(tf.trainable_variables())
    # sess.run(tf.global_variables_initializer())
    print('Restoring model from {}'.format(args.ckpt))
    saver.restore(sess, args.ckpt)
    predicted_logmel = sess.run(outputs,
                                feed_dict={
                                    inputs_pl: np.expand_dims(inputs, axis=0)})
    np.save(args.save_name.replace('.wav', '.npy'), np.squeeze(predicted_logmel))
    wav_arr = inv_mel_spectrogram(np.squeeze(predicted_logmel).T)
    wav_arr = inv_preemphasize(wav_arr)
    save_wav(wav_arr, args.save_name)
    generate_len = len(wav_arr)
    duration = time.time() - start_time
    print("Wav file generated in {:.3f} seconds".format(duration))
    print("Generation speed is {:.3f}sec/sec".format(
        duration / float(generate_len / args.sample_rate)))
    sess.close()


if __name__ == '__main__':
    main()
