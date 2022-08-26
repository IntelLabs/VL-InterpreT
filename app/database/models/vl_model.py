'''
An abstract base class for live models that can run together with VL-InterpreT.

To run your own model with VL-InterpreT, create another file your_model.py in this
folder that contains a class Your_Model (use title case for the class name), which
inherits from the VL_Model class and implements the data_setup method. The data_setup
method should take the ID, image and text of a given example, run a forward pass for
this example with your model, and return the corresponding attention, hidden states
and other required data that can be visualized with VL-InterpreT.
'''


from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import urllib.request


class VL_Model(ABC):
    '''
    To run a live transformer with VL-InterpreT, define your own model class by inheriting
    from this class and implementing the data_setup method.

    Please follow these naming patterns to make sure your model runs easily with VL-InterpreT:
        - Create a new python script in this folder for your class, and name it in all lower
          case (e.g., yourmodelname.py)
        - Name your model class in title case, e.g., Yourmodelname. This class name should be
          the result of calling 'yourmodelname'.title(), where 'yourmodelname.py' is the name
          of your python script.

    Then you can run VL-InterpreT with your model:
        python run_app.py -p 6006 -d example_database2 -m yourmodelname your_model_parameters
    '''

    @abstractmethod
    def data_setup(self, example_id: int, image_location: str, input_text: str) -> dict:
        '''
        This method should run a forward pass with your model given the input image and
        text, and return the required data. See app/database/db_example.py for specifications
        of the return data format, and see the implementation in kdvlp.py for an example.
        '''
        return {
            'ex_id': example_id,
            'image': np.array(),
            'tokens': [],
            'txt_len': 0,
            'img_coords': [],
            'attention': np.array(),
            'hidden_states': np.array()
        }


    def fetch_image(self, image_location: str):
        '''
        This helper function takes the path to an image (either an URL or a local path) and
        returns the image as an numpy array.
        '''
        if image_location.startswith('http'):
            urllib.request.urlretrieve(image_location, 'temp.jpg')
            image_location = 'temp.jpg'

        img = Image.open(image_location).convert('RGB')
        img = np.array(img)
        return img
