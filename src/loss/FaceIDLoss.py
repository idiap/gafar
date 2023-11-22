# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute
# <hatef.otroshi@idiap.ch>
# 
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# 
# SPDX-License-Identifier: BSD-3-Clause

import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
import imp
import os
from bob.extension.download import get_file # installation: pip install bob.extension


def Crop_and_resize(img):
    pad = 10
    img  =  torch.nn.functional.pad(img, pad=(pad, pad, pad, pad))

    FFHQ_REYE_POS = (190 + pad, 190 + pad) #(480, 380) 
    FFHQ_LEYE_POS = (190 + pad, 325 + pad) #(480, 650) 
    
    CROPPED_IMAGE_SIZE=(112, 112)
    fixed_positions={'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}

    cropped_positions = {
                        "leye": (51.6, 73.5318),
                        "reye": (51.6, 38.2946)
                         }
    """
    Steps:
        1) find rescale ratio

        2) find corresponding pixel in 512 image which will be mapped to                          
        the coordinate (0,0) at the croped_and_resized image
        
        3) find corresponding pixel in 512 image which will be mapped to                          
        the coordinate (112,112) at the croped_and_resized image
        
        4) crop image in 512
        
        5) resize the cropped image
    """
    # step1: find rescale ratio
    alpha = ( cropped_positions['leye'][1] - cropped_positions['reye'][1] )  /  ( fixed_positions['leye'][1]- fixed_positions['reye'][1] ) 
    
    # step2: find corresponding pixel in 512 image for (0,0) at the croped_and_resized image
    coord_0_0_at_512 = np.array(fixed_positions['reye']) - 1/alpha* np.array(cropped_positions['reye'])
    
    # step3: find corresponding pixel in 512 image for (112,112) at the croped_and_resized image
    coord_112_112_at_512 = coord_0_0_at_512 + np.array(CROPPED_IMAGE_SIZE) / alpha
    
    # step4: crop image in 512
    # cropped_img_512 = img[int(coord_0_0_at_512[0])     : int(coord_0_0_at_512[1]),
    #                        int(coord_112_112_at_512[0]) : int(coord_112_112_at_512[1]),
    #                        :]
    cropped_img_512 = img[:,
                           :,
                           int(coord_0_0_at_512[0]) : int(coord_112_112_at_512[0]),
                           int(coord_0_0_at_512[1]) : int(coord_112_112_at_512[1])
                           ]
    
    # step5: resize the cropped image
    # resized_and_croped_image = cv2.resize(cropped_img_512, CROPPED_IMAGE_SIZE) 
    resized_and_croped_image = torch.nn.functional.interpolate(cropped_img_512, mode='bilinear', size=CROPPED_IMAGE_SIZE, align_corners=False)

    return resized_and_croped_image


class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        device='cpu',
        image_dim = 112,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor_ = preprocessor
        self.device = device
        self.image_dim=  image_dim

    def preprocessor(self, X):
        X = self.preprocessor_(X)
        if X.size(2) == 512:
            X = Crop_and_resize(X)
        if X.size(2) != self.image_dim:
            X = torch.nn.functional.interpolate(X, mode='bilinear', size=(self.image_dim, self.image_dim), align_corners=False)
        return X
        
    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        if self.model is None:
            self._load_model()
            
            self.model.eval()
            
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad=False
                
        # X = check_array(X, allow_nd=True)
        # X = torch.Tensor(X)
        X = self.preprocessor(X)

        return self.model(X)#.detach().numpy()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
    
    def to(self,device):
        self.device=device
        
        if self.model !=None:            
            self.model.to(self.device)


def _get_iresnet_file():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    ]

    return get_file(
        "iresnet-91a5de61.tar.gz",
        urls,
        cache_subdir="data/pytorch/iresnet-91a5de61/",
        file_hash="3976c0a539811d888ef5b6217e5de425",
        extract=True,
    )

class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self,  
                preprocessor=lambda x: (x - 127.5) / 128.0, 
                device='cpu'
                ):

        self.device = device
        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(
            checkpoint_path, config, device=device
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model



class IResnet100Elastic(PyTorchModel):
    """
    ElasticFace model
    """

    def __init__(self,  
                preprocessor=lambda x: (x - 127.5) / 128.0, 
                device='cpu'
                ):

        self.device = device
        
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet100-elastic.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet100-elastic.tar.gz",
        ]

        filename= get_file(
            "iresnet100-elastic.tar.gz",
            urls,
            cache_subdir="data/pytorch/iresnet100-elastic/",
            file_hash="0ac36db3f0f94930993afdb27faa4f02",
            extract=True,
        )

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-elastic.pt")

        super(IResnet100Elastic, self).__init__(
            checkpoint_path, config, device=device,  preprocessor=preprocessor, 
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model
        

def get_FaceRecognition_transformer(FR_system='ArcFace', device='cpu'):
    if FR_system== 'ArcFace':
        FaceRecognition_transformer = IResnet100(device=device)
    elif FR_system== 'ElasticFace':
        FaceRecognition_transformer = IResnet100Elastic(device=device)
    else:
        print(f"[FaceIDLoss] {FR_system} is not defined!")
    return FaceRecognition_transformer 

class ID_Loss:
    def __init__(self,  FR_system='ArcFace', FR_loss='ArcFace', device='cpu' ):
        self.FaceRecognition_transformer = get_FaceRecognition_transformer(FR_system=FR_system, device=device)
        self.FaceRecognition_transformer_db = get_FaceRecognition_transformer(FR_system=FR_loss,device=device)
        
    def get_embedding(self,img):
        """
        img: generated     range: (-1,+1) +- delta
        """
        img = torch.clamp(img, min=-1, max=1)
        img = (img + 1) / 2.0 # range: (0,1)
        embedding = self.FaceRecognition_transformer.transform(img*255) # Note: input img should be in (0,255)
        return embedding

    def get_embedding_db(self,img):
        """
        img: generated     range: (-1,+1) +- delta
        """
        img = torch.clamp(img, min=-1, max=1)
        img = (img + 1) / 2.0 # range: (0,1)
        embedding = self.FaceRecognition_transformer_db.transform(img*255) # Note: input img should be in (0,255)
        return embedding

    def __call__(self, embedding1,embedding2):
        return torch.nn.MSELoss()(embedding1,embedding2)