import tensorflow as tf
import numpy as np
import pickle
from random import sample
from sklearn import preprocessing
import os

from audio import load_wav, wav2melspec, mel2log_mel, preempahsis

DATA_DIR = '/notebooks/projects/share/zhiling'
LF0_MEAN_F = '/notebooks/projects/share/zhiling/statistics/lf0/mean.npy'
LF0_STD_F = '/notebooks/projects/share/zhiling/statistics/lf0/std.npy'
LC_DIM = 132
NUM_MEL = 80
# STFT_DIM = 513


def sorted_file_list(data_dir, pattern='.lf0'):
    # default use lf0 data directory
    file_list = [(f.split('.')[0], os.path.getsize(os.path.join(data_dir, f)))
                 for f in os.listdir(data_dir) if f.endswith(pattern)]
    file_list.sort(key=lambda s: s[1])
    sorted_id_list = [f for f, _ in file_list]
    return sorted_id_list


def split_dataset(id_list, dev_size=50):
    # split dataset into train and dev set
    dev_set = sample(id_list, dev_size)
    for item in dev_set:
        id_list.remove(item)
    datasplit = {'train': id_list, 'dev': dev_set}
    with open('data_split.pkl', 'wb') as f:
        pickle.dump(datasplit, f, protocol=pickle.HIGHEST_PROTOCOL)
    return datasplit


def get_files_list(data_dir, mode='train'):
    # mode: 'train' --> get training data;
    #       'dev'   --> get dev data;
    #       'all'   --> get all ( both train and dev) data, used for generation.
    assert mode in ['train', 'dev', 'all'] and 'mode should be one of aforementioned'
    ppgs_dir = os.path.join(data_dir, 'ppgs')
    lf0_dir = os.path.join(data_dir, 'lf0')
    stft_dir = os.path.join(data_dir, 'stft-513')
    # read or create data split dictionary
    data_split_f = 'data_split.pkl'
    if os.path.isfile(data_split_f):
        fid_list = sorted_file_list(lf0_dir, pattern='.lf0')
        data_dict = split_dataset(fid_list, dev_size=50)
    else:
        with open(data_split_f, 'rb') as f:
            data_dict = pickle.load(f)
    if mode == 'all':
        ppgs_files = ([os.path.join(ppgs_dir, f + '.npy') for f in data_dict['train']] +
                      [os.path.join(ppgs_dir, f + '.npy') for f in data_dict['dev']])
        lf0_files = ([os.path.join(lf0_dir, f + '.lf0') for f in data_dict['train']] +
                     [os.path.join(lf0_dir, f + '.lf0') for f in data_dict['dev']])
        stft_files = ([os.path.join(stft_dir, f + '.npy') for f in data_dict['train']] +
                      [os.path.join(stft_dir, f + '.npy') for f in data_dict['dev']])
    else:
        ppgs_files = [os.path.join(ppgs_dir, f + '.npy') for f in data_dict[mode]]
        lf0_files = [os.path.join(lf0_dir, f + '.lf0') for f in data_dict[mode]]
        stft_files = [os.path.join(stft_dir, f + '.npy') for f in data_dict[mode]]
    return ppgs_files, lf0_files, stft_files


def all_data_generator():
    ppg_files, lf0_files, stft_files = get_files_list(DATA_DIR, mode='all')
    standard_scalar = preprocessing.StandardScaler()
    for ppg_f, lf0_f, stft_f in zip(ppg_files, lf0_files, stft_files):
        fname = stft_f.split('/')[-1]
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        stft = np.load(stft_f)
        ppg = standard_scalar.fit_transform(ppg)
        lf0[lf0 == -1.0E+10] = 0.0
        assert ppg.shape[0] == len(lf0) and stft.shape[0] == ppg.shape[0]
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, stft, lc.shape[0], fname


def train_generator():
    ppg_files, lf0_files, stft_files = get_files_list(DATA_DIR, mode='train')
    standard_scalar = preprocessing.StandardScaler()
    for ppg_f, lf0_f, stft_f in zip(ppg_files, lf0_files, stft_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        stft = np.load(stft_f)
        ppg = standard_scalar.fit_transform(ppg)
        lf0[lf0 == -1.0E+10] = 0.0
        assert ppg.shape[0] == len(lf0) and stft.shape[0] == ppg.shape[0]
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, stft, lc.shape[0]


def dev_generator():
    ppg_files, lf0_files, stft_files = get_files_list(DATA_DIR, mode='dev')
    standard_scalar = preprocessing.StandardScaler()
    for ppg_f, lf0_f, stft_f in zip(ppg_files, lf0_files, stft_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        stft = np.load(stft_f)
        ppg = standard_scalar.fit_transform(ppg)
        lf0[lf0 == -1.0E+10] = 0.0
        assert ppg.shape[0] == len(lf0) and stft.shape[0] == ppg.shape[0]
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, stft, lc.shape[0]


def get_files_list_fly(data_dir, mode='train'):
    # mode: 'train' --> get training data;
    #       'dev'   --> get dev data;
    #       'all'   --> get all ( both train and dev) data, used for generation.
    assert mode in ['train', 'dev', 'all'] and 'mode should be one of aforementioned'
    ppgs_dir = os.path.join(data_dir, 'ppgs_luhui')
    lf0_dir = os.path.join(data_dir, 'lf0_raw')
    wav_dir = os.path.join(data_dir, 'wavs')
    # read or create data split dictionary
    data_split_f = 'data_split.pkl'
    if os.path.isfile(data_split_f):
        fid_list = sorted_file_list(lf0_dir, pattern='.lf0')
        data_dict = split_dataset(fid_list, dev_size=50)
    else:
        with open(data_split_f, 'rb') as f:
            data_dict = pickle.load(f)
    if mode == 'all':
        ppgs_files = ([os.path.join(ppgs_dir, f + '.npy') for f in data_dict['train']] +
                      [os.path.join(ppgs_dir, f + '.npy') for f in data_dict['dev']])
        lf0_files = ([os.path.join(lf0_dir, f + '.lf0') for f in data_dict['train']] +
                     [os.path.join(lf0_dir, f + '.lf0') for f in data_dict['dev']])
        wav_files = ([os.path.join(wav_dir, f + '.wav') for f in data_dict['train']] +
                     [os.path.join(wav_dir, f + '.wav') for f in data_dict['dev']])
    else:
        ppgs_files = [os.path.join(ppgs_dir, f + '.npy') for f in data_dict[mode]]
        lf0_files = [os.path.join(lf0_dir, f + '.lf0') for f in data_dict[mode]]
        wav_files = [os.path.join(wav_dir, f + '.wav') for f in data_dict[mode]]
    return ppgs_files, lf0_files, wav_files


def lf0_normailze(lf0, mean_f=None, std_f=None):
    mean = np.load(mean_f) if mean_f is not None else np.mean(lf0[lf0 > 0])
    std = np.load(std_f) if std_f is not None else np.std(lf0[lf0 > 0])
    normalized = np.copy(lf0)
    normalized[normalized > 0] = (lf0[lf0 > 0] - mean) / std
    normalized[normalized == -1e10] = 0.0
    return normalized


def possible_pad(ppg, logmel, lf0):
    assert ppg.shape[0] == logmel.shape[0]
    if ppg.shape[0] > len(lf0):
        pad_n = ppg.shape[0] - len(lf0)
        lf0 = np.pad(lf0, (0, pad_n), mode='edge')
    elif ppg.shape[0] < len(lf0):
        pad_n = len(lf0) - ppg.shape[0]
        ppg = np.pad(ppg, ((0, pad_n), (0, 0)), mode='edge')
        logmel = np.pad(logmel, ((0, pad_n), (0, 0)), mode='edge')
    return ppg, logmel, lf0


def train_fly_generator():
    ppg_files, lf0_files, wav_files = get_files_list_fly(DATA_DIR, mode='train')
    standard_scalar = preprocessing.StandardScaler()
    for ppg_f, lf0_f, wav_f in zip(ppg_files, lf0_files, wav_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        lf0 = lf0_normailze(lf0, mean_f=LF0_MEAN_F, std_f=LF0_STD_F)
        wav_arr = load_wav(wav_f)
        preemphasized = preempahsis(wav_arr)
        melspectrogeam = wav2melspec(preemphasized)
        logmel = mel2log_mel(melspectrogeam)
        ppg, logmel, lf0 = possible_pad(ppg, logmel, lf0)
        ppg = standard_scalar.fit_transform(ppg)
        assert ppg.shape[0] == len(lf0) and logmel.shape[0] == ppg.shape[0]
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, logmel, lc.shape[0]


def dev_fly_generator():
    ppg_files, lf0_files, wav_files = get_files_list_fly(DATA_DIR, mode='dev')
    standard_scalar = preprocessing.StandardScaler()
    for ppg_f, lf0_f, wav_f in zip(ppg_files, lf0_files, wav_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        lf0 = lf0_normailze(lf0)  # dev set have no access to global mean and std
        wav_arr = load_wav(wav_f)
        preemphasized = preempahsis(wav_arr)
        melspectrogeam = wav2melspec(preemphasized)
        logmel = mel2log_mel(melspectrogeam)
        ppg, logmel, lf0 = possible_pad(ppg, logmel, lf0)
        ppg = standard_scalar.fit_transform(ppg)
        assert ppg.shape[0] == len(lf0) and logmel.shape[0] == ppg.shape[0]
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, logmel, lc.shape[0]


def dataset_test():
    batch_size = 8
    train_set = tf.data.Dataset.from_generator(
        train_fly_generator,
        output_types=(tf.float32, tf.float32, tf.int32),
        output_shapes=([None, LC_DIM], [None, NUM_MEL], []))
    train_set = train_set.padded_batch(batch_size,
                                       ([None, LC_DIM],
                                        [None, NUM_MEL], [])
                                       ).repeat()
    train_itertor = train_set.make_initializable_iterator()
    train_data = train_itertor.get_next()
    dev_set = tf.data.Dataset.from_generator(
        dev_fly_generator,
        output_types=(tf.float32, tf.float32, tf.int32),
        output_shapes=([None, LC_DIM], [None, NUM_MEL], []))
    dev_set = dev_set.padded_batch(batch_size,
                                   ([None, LC_DIM],
                                    [None, NUM_MEL], [])
                                   ).repeat()
    dev_iterator = dev_set.make_initializable_iterator()
    dev_data = dev_iterator.get_next()
    sess = tf.Session()
    sess.run([train_itertor.initializer, dev_iterator.initializer])
    for i in range(10):
        a, b = sess.run([train_data, dev_data])
        print('train:', i, a[0].shape, a[1].shape, a[2])
        print('dev:', i, b[0].shape, b[1].shape, b[2])
    sess.close()


def fly_generator_test():
    ppg_files, lf0_files, wav_files = get_files_list_fly(DATA_DIR, mode='all')
    standard_scalar = preprocessing.StandardScaler()
    for ppg_f, lf0_f, wav_f in zip(ppg_files, lf0_files, wav_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        lf0 = lf0_normailze(lf0, mean_f=LF0_MEAN_F, std_f=LF0_STD_F)
        wav_arr = load_wav(wav_f)
        wav_len = len(wav_arr)
        preemphasized = preempahsis(wav_arr)
        melspectrogeam = wav2melspec(preemphasized)
        logmel = mel2log_mel(melspectrogeam)
        ppg, logmel, lf0 = possible_pad(ppg, logmel, lf0)
        ppg = standard_scalar.fit_transform(ppg)
        if ppg.shape[0] == len(lf0) and logmel.shape[0] == ppg.shape[0]:
            continue
        else:
            print(ppg.shape, logmel.shape, lf0.shape, wav_len, wav_f)


# if __name__ == '__main__':
#     fly_generator_test()
#     dataset_test()

