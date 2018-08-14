#from https://github.com/imironhead/ml_dcgan_lsun/blob/master/lsun.py
import lmdb
import numpy
import os
import pickle
import scipy.misc
from io import BytesIO


class Lsun(object):
    """
    A utility to read LSUN data set.

    http://lsun.cs.princeton.edu/2016/
    """
    def load_keys(path_lsun_dir):
        """
        Load keys from a pickle file. All key of the database will be dumped
        into a pickle file if there is none.
        """
        path_keys = os.path.join(path_lsun_dir, 'keys.pkl')

        if os.path.isfile(path_keys):
            with open(path_keys, 'rb') as kf:
                return pickle.Unpickler(kf).load()

        print('generating keys of lmdb: ' + path_lsun_dir)

        keys = []

        with lmdb.open(path_lsun_dir) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    keys_iter = cursor.iternext_nodup(keys=True, values=False)

                    keys_count = env.stat()['entries']

                    for idx, key in enumerate(keys_iter):
                        keys.append(key)

                        if idx % 1000 == 0:
                            print('found keys: {} / {}'.format(idx, keys_count))
        
        with open(path_keys, 'wb') as kf:
            pickle.Pickler(kf).dump(keys)

        return keys

    def __init__(self, path_lsun_dir):

        self._lmdb_path = path_lsun_dir
        self._lmdb_keys = Lsun.load_keys(path_lsun_dir)
        self._key_indice = numpy.arange(len(self._lmdb_keys))
        self._key_position = 0

        numpy.random.shuffle(self._key_indice)
        
    def load_data(self,len_x_train):
        begin = 0
        end = len_x_train

        images = []

        with lmdb.open(self._lmdb_path) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    for i in range(begin, end):
                        val = cursor.get(self._lmdb_keys[self._key_indice[i]])
                        sio = BytesIO(val)

                        img = scipy.misc.imread(sio)
                        img = scipy.misc.imresize(img,(64,64))
                        img = img.astype(numpy.float32)

                        img /= 255.

                        images.append(img)

        return images
    
    def next_batch(self, batch_size):
        """
        Get next batch_size images from the database.
        All images are resized to 25% (either 64x? or ?x64).
        All pixels are remapped to range between -1.0 ~ +1.0.
        """
        begin = self._key_position

        self._key_position += batch_size

        if self._key_position > len(self._key_indice):
            numpy.random.shuffle(self._key_indice)

            begin = 0

            self._key_position = batch_size

            assert batch_size <= len(self._key_indice)

        end = self._key_position

        images = []

        with lmdb.open(self._lmdb_path) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    for i in range(begin, end):
                        val = cursor.get(self._lmdb_keys[self._key_indice[i]])
                        sio = BytesIO(val)

                        img = scipy.misc.imread(sio)
                        img = scipy.misc.imresize(img,(64,64))
                        img = img.astype(numpy.float32)

                        img /= 255.

                        images.append(img)

        return images