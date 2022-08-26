'''
The following script runs a forward passes with the KD-VLP model for
each given image+text pair, and produces the corresponding attention
and hidden states that can be visualized with VL-InterpreT.

The KD-VLP model has not been made publicly available in this repo.
Please create your own model class by inheriting from VL_Model.

Note that to run your own model with VL-InterpreT, you are only
required to implement the data_setup function. Most of the code in
this file are specific to our KD-VLP model, and they are provided just
for your reference.
'''


import numpy as np
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

try:
    from app.database.models.vl_model import VL_Model
except ModuleNotFoundError:
    from vl_model import VL_Model

import sys
MODEL_DIR = '/workdisk/ccr_vislang/benchmarks/vision_language/vcr/VILLA/'
sys.path.append(MODEL_DIR)

from model.vcr import UniterForVisualCommonsenseReasoning
from utils.const import IMG_DIM
from utils.misc import remove_prefix
from data import get_transform


class Kdvlp(VL_Model):
    '''
    Running KD-VLP with VL-Interpret:
    python run_app.py -p 6006  -d example_database2  \
                      -m kdvlp /data1/users/shaoyent/e2e-vcr-kgmvm-pgm-run49-2/ckpt/model_step_8500.pt
    '''
    def __init__(self, ckpt_file, device='cuda'):
        self.device = torch.device(device, 0)
        if device == 'cuda':
            torch.cuda.set_device(0)
        self.model, self.tokenizer = self.build_model(ckpt_file)


    def build_model(self, ckpt_file):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        tokenizer._add_tokens([f'[PERSON_{i}]' for i in range(81)])

        ckpt = torch.load(ckpt_file)
        checkpoint = {remove_prefix(k.replace('bert', 'uniter'), 'module.') : v for k, v in ckpt.items()}
        model = UniterForVisualCommonsenseReasoning.from_pretrained(
                        f'{MODEL_DIR}/config/uniter-base.json', state_dict={},
                        img_dim=IMG_DIM)
        model.init_type_embedding()
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        model = model.to(self.device)
        return model, tokenizer


    def move_to_device(self, x):
        if isinstance(x, list):
            return [self.move_to_device(y) for y in x]
        elif isinstance(x, dict):
            new_dict = {}
            for k, v in x.items():
                new_dict[k] = self.move_to_device(x[k])
            return new_dict
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:
            return x


    def build_batch(self, input_text, image, answer=None, person_info=None):
        if not input_text:
            input_text = ''
        if answer is None:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # all*input_ids
            txt_type_ids = torch.zeros_like(input_ids)
        else:
            input_ids_q = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_text))
            input_ids_c = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))
            input_ids = [torch.tensor(self.tokenizer.build_inputs_with_special_tokens(input_ids_q, input_ids_c))]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # all*input_ids

            txt_type_ids = torch.tensor((len(input_ids_q) + 2 )* [0] + (len(input_ids_c) + 1) * [2])

        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
        num_sents = [input_ids.size(0)]       
        txt_lens = [i.size(0) for i in input_ids]

        if image is None:
            images_batch = None
        else:
            images_batch = torch.as_tensor(image.copy(), dtype=torch.float32)
            images_batch = get_transform(images_batch.permute(2, 0, 1))

        batch = {'input_ids': input_ids, 'txt_type_ids': txt_type_ids, 'position_ids': position_ids, 'images': images_batch,
                "txt_lens": txt_lens, "num_sents": num_sents, 'person_info': person_info}
        batch = self.move_to_device(batch)

        return batch


    def data_setup(self, ex_id, image_location, input_text):
        image = self.fetch_image(image_location) if image_location else None

        batch = self.build_batch(input_text, image, answer=None, person_info=None)
        scores, hidden_states, attentions = self.model(batch,
                                                       compute_loss=False,
                                                       output_attentions=True,
                                                       output_hidden_states=True)

        attentions = torch.stack(attentions).transpose(1,0).detach().cpu()[0]

        if batch['images'] is None:
            img, img_coords = np.array([]), []
            len_img = 0
        else:
            image1, mask1 = self.model.preprocess_image(batch['images'].to(self.device))
            image1 = (image1 * self.model.pixel_std + self.model.pixel_mean) * mask1
            img = image1.cpu().numpy().astype(int).squeeze().transpose(1,2,0)

            h, w, _ = img.shape
            h0, w0 = h//64, w//64
            len_img = w0 * h0
            img_coords = np.fliplr(list(np.ndindex(h0, w0)))

        input_ids = batch['input_ids'].cpu()
        len_text = input_ids.size(1)
        txt_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :len_text])

        len_tokens = len_text + len_img
        attentions = attentions[:, :, :len_tokens, :len_tokens]
        hidden_states = [hs[0].detach().cpu().numpy()[:len_tokens] for hs in hidden_states]

        return {
            'ex_id': ex_id,
            'image': img,
            'tokens': txt_tokens + [f'IMG_{i}' for i in range(len_img)],
            'txt_len': len(txt_tokens),
            'attention': attentions.detach().cpu().numpy(),
            'img_coords': img_coords,
            'hidden_states': hidden_states
        }


def create_example_db():
    '''
    This function creates example_database2.
    '''
    images = [f'{MODEL_DIR}/visualization/ex.jpg']
    texts = ['Horses are pulling a carriage, while someone is standing on the top of a golden ball.']
    kdvlp = Kdvlp('/data1/users/shaoyent/e2e-vcr-kgmvm-pgm-run49-2/ckpt/model_step_8500.pt')
    data = [kdvlp.data_setup(i, img, txt) for i, (img, txt) in enumerate(zip(images, texts))]

    db = VliLmdb(db_dir='/workdisk/VL-InterpreT/example_database2', read_only=False)
    for i, dat in enumerate(data):
        db[str(i)] = pickle.dumps(dat, protocol=pickle.HIGHEST_PROTOCOL)
    db.preprocess()


if __name__ == '__main__':
    sys.path.append('..')
    from database import VliLmdb
    create_example_db()
