import os
import numpy as np
from scipy.stats import zscore
import faiss
import pickle
from app.database.database import VliLmdb

ATTN_MAP_SCALE_FACTOR = 64

class VliDataBaseAnalyzer:
    def __init__(self, db_dir, read_only=True):
        self.db_dir = db_dir
        self.db = VliLmdb(db_dir, read_only=read_only)
        faiss_folder = os.path.join(db_dir, 'faiss')
        self.faiss_indices = {(layer, mod): faiss.read_index(f'{faiss_folder}/{mod}_indices_{layer}')
                              for layer in range(self['n_layers'] + 1) for mod in ('txt', 'img')}
        self.faiss_data = self['faiss']
        self.updated = False

    def __getitem__(self, item):
        if isinstance(self.db[item], bytes):
            return pickle.loads(self.db[item])
        return self.db[item]

    def __del__(self):
        if self.updated:
            print('Writing new examples to database...')
            self.db['faiss'] = pickle.dumps(self.faiss_data, protocol=pickle.HIGHEST_PROTOCOL)
            faiss_path = os.path.join(self.db.path, 'faiss')
            for layer in range(self['n_layers'] + 1):
                faiss.write_index(self.faiss_indices[(layer, 'txt')], os.path.join(faiss_path, f'txt_indices_{layer}'))
                faiss.write_index(self.faiss_indices[(layer, 'img')], os.path.join(faiss_path, f'img_indices_{layer}'))
            print('Done.')

    def add_example(self, ex_id, ex_data):
        self.faiss_indices, self.faiss_data = self.db.preprocess_example(ex_id, ex_data, self.faiss_indices, self.faiss_data)
        self.updated = True

    def get_ex_attn(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        ex_attn = self[ex_id]['attention']
        ex_tokens = self[ex_id]['tokens']
        if exclude_tokens:
            exclude_indices = [i for i, token in enumerate(ex_tokens) if token in exclude_tokens]
            ex_attn = np.delete(ex_attn, exclude_indices, axis=2)
            ex_attn = np.delete(ex_attn, exclude_indices, axis=3)
        return ex_attn

    def get_attn_means(self, attn, normalize=True):
        head_avg = attn.mean(axis=(2, 3))
        if normalize:
            head_avg = zscore(head_avg)
        layer_avg = np.mean(head_avg, axis=1)
        return head_avg, layer_avg

    def get_attn_components_means(self, components):
        avg_attn = np.mean(np.array(components), axis=0)
        avg_attn = zscore(avg_attn)  # normalize
        layer_avg = np.mean(avg_attn, axis=1)
        return avg_attn, layer_avg

    def img2txt_mean_attn(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        ex_attn = self.get_ex_attn(ex_id, exclude_tokens)
        txt_len = self[ex_id]['txt_len']
        img2txt_attn = ex_attn[:, :, :txt_len, txt_len:]
        return self.get_attn_means(img2txt_attn)

    def txt2img_mean_attn(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        ex_attn = self.get_ex_attn(ex_id, exclude_tokens)
        txt_len = self[ex_id]['txt_len']
        txt2img_attn = ex_attn[:, :, txt_len:, :txt_len]
        return self.get_attn_means(txt2img_attn)

    def txt2txt_mean_attn(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        ex_attn = self.get_ex_attn(ex_id, exclude_tokens)
        txt_len = self[ex_id]['txt_len']
        txt2txt_attn = ex_attn[:, :, :txt_len, :txt_len]
        return self.get_attn_means(txt2txt_attn)

    def img2img_mean_attn(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        ex_attn = self.get_ex_attn(ex_id, exclude_tokens)
        txt_len = self[ex_id]['txt_len']
        txt2img_attn = ex_attn[:, :, txt_len:, txt_len:]
        return self.get_attn_means(txt2img_attn)

    def img2img_mean_attn_without_self(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        ex_attn = self.get_ex_attn(ex_id, exclude_tokens)
        txt_len = self[ex_id]['txt_len']
        txt2img_attn = ex_attn[:, :, txt_len:, txt_len:]
        n_layer, n_head, attn_len, attn_len = txt2img_attn.shape
        for i in range(attn_len):
            for j in range(attn_len):
                if i == j or i == j+1 or j == i+1:
                    txt2img_attn[:, :, i, j ] = 0
        return self.get_attn_means(txt2img_attn)

    def get_all_attn_stats(self, ex_id, exclude_tokens=['[SEP]', '[CLS]']):
        attn_stats = []
        for func in (self.img2txt_mean_attn,
                     self.txt2img_mean_attn,
                     self.img2img_mean_attn,
                     self.img2img_mean_attn_without_self,
                     self.txt2txt_mean_attn):
            attn, layer_avg = func(ex_id, exclude_tokens)
            attn_stats.append((attn, layer_avg))
        # crossmodal
        attn_stats.append(self.get_attn_components_means([attn_stats[0][0], attn_stats[1][0]]))
        # intramodal
        attn_stats.append(self.get_attn_components_means([attn_stats[2][0], attn_stats[3][0]]))
        for i, (stats, layer_avg) in enumerate(attn_stats):
            attn_stats[i] = np.hstack((stats, layer_avg.reshape(layer_avg.shape[0], 1)))
        return attn_stats

    def get_custom_metrics(self, ex_id):
        if 'custom_metrics' in self[ex_id]:
            cm = self[ex_id]['custom_metrics']
            labels, stats = [], []
            for metrics in cm:
                data = cm[metrics]
                mean = np.mean(data, axis=1)
                mean = mean.reshape(mean.shape[0], 1)
                data = np.hstack((data, mean))
                labels.append(metrics)
                stats.append(data)
            return labels, stats

    def find_closest_token(self, tsne, layer, mod):
        _, index = self.faiss_indices[(layer, mod)].search(tsne.reshape(1, -1), 1)  # returns distance, index
        index = index.item(0)
        ex_id, token_id = self.faiss_data[(layer, mod)][index]
        return ex_id, token_id

    def get_txt_token_index(self, ex_id, text_tokens_id):
        # text_tokens_id: {'sentence': 0, 'word': 0}
        if not text_tokens_id:
            return
        sep_idx = [-1] + [i for i, x in enumerate(self[ex_id]['tokens']) if x == '[SEP]']
        sentence, word = text_tokens_id['sentence'], text_tokens_id['word']
        return sep_idx[sentence] + 1 + word

    def get_img_token_index(self, ex_id, img_coords):
        txt_len = self[ex_id]['txt_len']
        return self[ex_id]['img_coords'].index(tuple(img_coords)) + txt_len

    def get_img_unit_len(self, ex_id):
        token_grid_size = self[ex_id]['img_grid_size']
        image_size = self[ex_id]['image'].shape
        return image_size[1]//token_grid_size[0], image_size[0]//token_grid_size[1]
