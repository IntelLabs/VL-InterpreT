import pickle
import numpy as np
from database import VliLmdb

if __name__ == '__main__':

    # The following example contains randomly generated data to
    # illustrate what the data should look like before preprocessing
    example_data = []
    for ex_id in range(3):
        example_data.append(
            # each example should contain the following information:
            {
                # Example ID (integers starting from 0)
                'ex_id': ex_id,

                # The original input image (RGB)
                # If your model preprocesses the image (e.g., resizing, padding), you may want to
                # use the preprocessed image instead of the original
                'image': np.ones((450, 800, 3), dtype=int) * 100 * ex_id + 30,

                # Input tokens (text tokens followed by image tokens)
                'tokens': ['[CLS]', 'text', 'input', 'for', 'example', str(ex_id), '.', '[SEP]',
                        'IMG_0', 'IMG_1', 'IMG_2', 'IMG_3', 'IMG_4', 'IMG_5'],

                # The number of text tokens
                'txt_len': 8,

                # The (x, y) coordinates of each image token on the original image,
                # assuming the *top left* corner of an image is (0, 0)
                # The order of coordinates should correspond to how image tokens are ordered in 'tokens'
                'img_coords': [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],

                # (Optional) Whether model predicted correctly for this example
                'accuracy': ex_id % 2 == 0,  # either True or False

                # Attention weights for all attention heads in all layers
                # Shape: (n_layers, n_attention_heads_per_layer, n_tokens, n_tokens)
                # n_layers and n_attention_heads_per_layer should be the same accross example
                # The order of columns and rows of the attention weight matrix for each head should
                # correspond to how tokens are ordered in 'tokens'
                'attention': np.random.rand(12, 12, 14, 14),

                # The hidden representations for each token in the model,
                # both before the first layer and after each layer
                # Shape: (n_layers + 1, n_tokens, hidden_state_vector_size)
                # Note that in our demo app, hidden representations of stop words were removed
                # to reduce the number of displayed datapoints
                'hidden_states': np.random.rand(13, 14, 768),

                # (Optional) Custom statistics for attention heads in all layers
                # Shape: (n_layers, n_attention_heads_per_layer)
                # The order should follow how attention heads are ordered in 'attention' matrices
                'custom_metrics': {'Example Custom Metrics': np.random.rand(12, 12)}
            }
        )

    # Create database
    print('Creating database...')
    db = VliLmdb(db_dir='example_database1', read_only=False)
    for ex_id, ex_data in enumerate(example_data):
        # Keys must be strings
        db[str(ex_id)] = pickle.dumps(ex_data, protocol=pickle.HIGHEST_PROTOCOL)

    # Preprocess the database
    db.preprocess()

