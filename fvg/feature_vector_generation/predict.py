# predict.py - predict (infer) inputs (single/batch).
# Code from https://github.com/christiansafka/img2vec.git used

import PIL
import numpy as np

from typing import Iterable, Dict, Callable

import torch
from torch import nn

from feature_vector_generation.models import Img2Vec

class Predictor(Img2Vec):

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        super().__init__(cuda=cuda, model=model, layer=layer, layer_output_size=layer_output_size)
        self._activation = {}

    def save_output_hook(self, layer_id : str) -> Callable:
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self._activation[layer_id] = output.detach()
        return hook

    def _get_feature_vectors(self) -> np.ndarray:
        return self._activation['ext_layer'].squeeze().numpy()
        
    def get_vec_np(self, imgs: np.ndarray, tensor:bool=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        
        if not len(imgs.shape) == 4 and not imgs.shape[3] == 3:
            raise ValueError("Input nd array should have 4 dimensions and (HxWx3) images")
        
        images = torch.stack([self.normalize(self.scaler(self.to_tensor(i))) for i in imgs]).to(self.device)

        h = self.extraction_layer.register_forward_hook(self.save_output_hook('fvec'))
        with torch.no_grad():
            h_x = self.model(images)
        h.remove()
        
        return self._get_feature_vectors()

    def predict(self, imgs_np:np.ndarray) -> np.ndarray:
        if not len(imgs_np.shape) == 4 and not imgs_np.shape[3] == 3:
            raise ValueError("Input nd array should have 4 dimensions and (HxWx3) images")
        
        images = torch.stack([self.normalize(self.scaler(self.to_tensor(i))) for i in imgs_np]).to(self.device)

        h = self.extraction_layer.register_forward_hook(self.save_output_hook('fvec'))
        with torch.no_grad():
            h_x = self.model(images)
        h.remove()
        
        return self._get_feature_vectors()

        pass

    def get_my_embedding_ph(self, num_imgs:int) -> torch.Tensor:

        if self.model_name in ['alexnet', 'vgg']:
            my_embedding = torch.zeros(num_imgs, self.layer_output_size)
        elif self.model_name == 'densenet':
            my_embedding = torch.zeros(num_imgs, self.layer_output_size, 7, 7)
        else:
            my_embedding = torch.zeros(num_imgs, self.layer_output_size, 1, 1)
            
        return my_embedding


    def get_vec_pil(self, img, tensor:bool=False) -> np.ndarray:
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            
            my_embedding = self.get_my_embedding_ph(len(img))

            def copy_data(model, input, output):
                my_embedding.copy_(output.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[:, :]
                elif self.model_name == 'densenet':
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            my_embedding = self.get_my_embedding_ph(1)

            def copy_data(model, input, output):
                my_embedding.copy_(output.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[0, :]
                elif self.model_name == 'densenet':
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                   return my_embedding.numpy()[0, :, 0, 0]
 