import os
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import faiss
from tqdm import tqdm
import pickle
import lmdb
from lz4.frame import compress, decompress
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


class VliLmdb(object):
    def __init__(self, db_dir, read_only=True, local_world_size=1):
        self.readonly = read_only
        self.path = db_dir
        if not os.path.isdir(db_dir):
            os.mkdir(db_dir)
        if read_only:
            readahead = not self._check_distributed(local_world_size)
            self.env = lmdb.open(db_dir, readonly=True, create=False, lock=False, readahead=readahead)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            self.env = lmdb.open(db_dir, readonly=False, create=True, map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        if not self.txn.get('n_examples'.encode('utf-8')):
            self['n_examples'] = 0

    def _check_distributed(self, local_world_size):
        try:
            dist = local_world_size != 1
        except ValueError:
            # not using horovod
            dist = False
        return dist

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        value = self.txn.get(str(key).encode('utf-8'))
        if value is None:
            raise KeyError(key)
        return msgpack.loads(decompress(value), raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'), compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret
    
    def preprocess_example(self, ex_id, ex_data, faiss_indices, faiss_data):
        txt_len = ex_data['txt_len']
        # image tokens
        if ex_data['img_coords'] is not None and len(ex_data['img_coords']) > 0:
            ex_data['img_grid_size'] = np.max(ex_data['img_coords'], axis=0) + 1
            img_coords = [tuple(coords) for coords in ex_data['img_coords']]
            img_coords = np.array(img_coords, np.dtype([('x', int), ('y', int)]))
            img_sort = np.argsort(img_coords, order=('y', 'x'))
            ex_data['img_coords'] = np.take_along_axis(img_coords, img_sort, axis=0).tolist()
            img_tokens = np.array(ex_data['tokens'][txt_len:])
            ex_data['tokens'][txt_len:] = np.take_along_axis(img_tokens, img_sort, axis=0)
            ex_data['attention'][:,:,txt_len:] = np.take(ex_data['attention'][:,:,txt_len:], img_sort, axis=2)
            ex_data['attention'][:,:,:,txt_len:] = np.take(ex_data['attention'][:,:,:,txt_len:], img_sort, axis=3)
        else:
            ex_data['img_grid_size'] = (0, 0)
            print(f'Warning: Image coordinates are missing for example #{ex_id}.')
        # t-SNE
        tsne = [TSNE(n_components=2, random_state=None, n_jobs=-1).fit_transform(ex_data['hidden_states'][i])
                for i in range(len(ex_data['hidden_states']))]
        if len(faiss_indices) == 0:
            faiss_indices = {(layer, mod): faiss.IndexFlatL2(2) \
                             for layer in range(len(tsne)) for mod in ('txt', 'img')}
        ex_data['tsne'] = tsne
        self[str(ex_id)] = pickle.dumps(ex_data, protocol=pickle.HIGHEST_PROTOCOL)
        self['n_examples'] += 1
        # faiss
        for layer in range(len(tsne)):
            faiss_all_tokens = [(ex_id, i) for i in range(len(ex_data['tokens']))]
            faiss_data[(layer, 'txt')] += faiss_all_tokens[:txt_len]
            faiss_data[(layer, 'img')] += faiss_all_tokens[txt_len:]
            tsne_txt = tsne[layer][:txt_len]
            tsne_img = tsne[layer][txt_len:]
            faiss_indices[(layer, 'txt')].add(tsne_txt)
            faiss_indices[(layer, 'img')].add(tsne_img)
        return faiss_indices, faiss_data

    def preprocess(self):
        print('Preprocessing database...')
        faiss_indices = {}
        faiss_data = defaultdict(list)
        n_layers, n_heads = 0, 0
        with self.txn.cursor() as cursor:
            for key, value in tqdm(cursor):
                ex_id = str(key, 'utf8')
                if not ex_id.isdigit():
                    continue
                ex_data = pickle.loads(msgpack.loads(decompress(value), raw=False))
                faiss_indices, faiss_data = self.preprocess_example(ex_id, ex_data, faiss_indices, faiss_data)
        
        n_layers, n_heads, _, _ = ex_data['attention'].shape
        self['n_layers'] = n_layers
        self['n_heads'] = n_heads

        print('Generating faiss indices...')
        faiss_path = os.path.join(self.path, 'faiss')
        if not os.path.exists(faiss_path):
            os.makedirs(faiss_path)
        for layer in range(n_layers + 1):
            faiss.write_index(faiss_indices[(layer, 'txt')], os.path.join(faiss_path, f'txt_indices_{layer}'))
            faiss.write_index(faiss_indices[(layer, 'img')], os.path.join(faiss_path, f'img_indices_{layer}'))
        self['faiss'] = pickle.dumps(faiss_data, protocol=pickle.HIGHEST_PROTOCOL)
        print('Preprocessing done.')
