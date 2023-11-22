# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute
# <hatef.otroshi@idiap.ch>
# 
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# 
# SPDX-License-Identifier: BSD-3-Clause

import torch
from   torch.utils.data import Dataset
from .loss.FaceIDLoss import get_FaceRecognition_transformer

import glob
import random
import numpy as np
import cv2

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def Crop_512_Synthesize(im):
    pad = 150
    img  =  np.zeros([im.shape[0]+int(2*pad), im.shape[1]+int(2*pad), 3])
    img[pad:-pad,pad:-pad,:] = im

    FFHQ_REYE_POS = (480 + pad, 380 + pad) #(480, 380) 
    FFHQ_LEYE_POS = (480 + pad, 650 + pad) #(480, 650) 
    
    CROPPED_IMAGE_SIZE=(512, 512)
    fixed_positions={'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}

    cropped_positions = {
                        "leye": (190, 325),
                        "reye": (190, 190)
                         }
    """
    Steps:
        1) find rescale ratio

        2) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (0,0) at the croped_and_resized image
        
        3) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (112,112) at the croped_and_resized image
        
        4) crop image in 1024
        
        5) resize the cropped image
    """
    # step1: find rescale ratio
    alpha = ( cropped_positions['leye'][1] - cropped_positions['reye'][1] )  /  ( fixed_positions['leye'][1]- fixed_positions['reye'][1] ) 
    
    # step2: find corresponding pixel in 1024 image for (0,0) at the croped_and_resized image
    coord_0_0_at_1024 = np.array(fixed_positions['reye']) - 1/alpha* np.array(cropped_positions['reye'])
    
    # step3: find corresponding pixel in 1024 image for (112,112) at the croped_and_resized image
    coord_112_112_at_1024 = coord_0_0_at_1024 + np.array(CROPPED_IMAGE_SIZE) / alpha
    
    # step4: crop image in 1024
    cropped_img_1024 = img[int(coord_0_0_at_1024[0]) : int(coord_112_112_at_1024[0]),
                           int(coord_0_0_at_1024[1]) : int(coord_112_112_at_1024[1]),
                           :]
    
    # step5: resize the cropped image
    resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE) 

    return resized_and_croped_image

def Crop_112_FR(img):
    """
    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale 
    Output:
        - new_img: RGB or BGR image in 0-1 or 0-255 scale 
    """

    FFHQ_REYE_POS = (480, 380) 
    FFHQ_LEYE_POS = (480, 650) 
    
    CROPPED_IMAGE_SIZE=(112, 112)
    fixed_positions={'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}

    cropped_positions = {
                        "leye": (51.6, 73.5318),
                        "reye": (51.6, 38.2946)
                         }
    """
    Steps:
        1) find rescale ratio

        2) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (0,0) at the croped_and_resized image
        
        3) find corresponding pixel in 1024 image which will be mapped to                          
        the coordinate (112,112) at the croped_and_resized image
        
        4) crop image in 1024
        
        5) resize the cropped image
    """
    # step1: find rescale ratio
    alpha = ( cropped_positions['leye'][1] - cropped_positions['reye'][1] )  /  ( fixed_positions['leye'][1]- fixed_positions['reye'][1] ) 
    
    # step2: find corresponding pixel in 1024 image for (0,0) at the croped_and_resized image
    coord_0_0_at_1024 = np.array(fixed_positions['reye']) - 1/alpha* np.array(cropped_positions['reye'])
    
    # step3: find corresponding pixel in 1024 image for (112,112) at the croped_and_resized image
    coord_112_112_at_1024 = coord_0_0_at_1024 + np.array(CROPPED_IMAGE_SIZE) / alpha

    # step4: crop image in 1024
    cropped_img_1024 = img[int(coord_0_0_at_1024[0]) : int(coord_112_112_at_1024[0]),
                           int(coord_0_0_at_1024[1]) : int(coord_112_112_at_1024[1]),
                           :]
    
    # step5: resize the cropped image
    resized_and_croped_image = cv2.resize(cropped_img_1024, CROPPED_IMAGE_SIZE) 

    return resized_and_croped_image
    
class MyDataset(Dataset):
    def __init__(self, dataset_dir = './Flickr-Faces-HQ/images1024x1024',
                       FR_system= 'ArcFace',
                       train=True,
                       device='cpu',
                       mixID_TrainTest=True,
                       train_test_split = 0.9,
                       random_seed=2021
                ):
        self.dataset_dir = dataset_dir
        self.device = device
        self.train  = train

        self.dir_all_images = []

        all_folders = glob.glob(dataset_dir+'/*')
        all_folders.sort()
        for folder in all_folders:
            all_imgs = glob.glob(folder+'/*.png')
            all_imgs.sort()
            for img in all_imgs:
                self.dir_all_images.append(img)
                                
        if mixID_TrainTest:
            random.seed(random_seed)
            random.shuffle(self.dir_all_images)
            
        if self.train:
            self.dir_all_images = self.dir_all_images[:int(train_test_split*len(self.dir_all_images))]
        else:
            self.dir_all_images = self.dir_all_images[int(train_test_split*len(self.dir_all_images)):]


        self.Face_Recognition_Network = get_FaceRecognition_transformer(FR_system=FR_system, device=self.device)

        self.FFHQ_align_mask = Crop_512_Synthesize(np.ones([1024,1024,3]).astype('uint8')) 
        self.FFHQ_align_mask = torch.tensor(self.FFHQ_align_mask).to(device) 
        self.FFHQ_align_mask = torch.transpose(self.FFHQ_align_mask,0,2)
    
    def __len__(self):
        return len(self.dir_all_images)

    def get_batch(self, batch_idx, batch_size):
        all_embedding = []  
        all_image = []
        all_image_HQ = []
        for idx in range(batch_size):
            embedding, image, image_HQ = self.__getitem__(batch_idx*batch_size+ idx) 
            all_embedding.append(embedding)
            all_image.append(image)
            all_image_HQ.append(image_HQ)
        return torch.stack(all_embedding).to(self.device), torch.stack(all_image).to(self.device), torch.stack(all_image_HQ ).to(self.device)

    def __getitem__(self, idx):

        image_1024 = cv2.imread(self.dir_all_images[idx]) # (1024, 1024, 3)

        image_HQ = cv2.cvtColor(image_1024, cv2.COLOR_BGR2RGB)
        image = Crop_112_FR(image_HQ) # (112, 112, 3)
        image_HQ = Crop_512_Synthesize(image_HQ) 

        image_HQ = image_HQ/255.
        image    = image/255.

        image = image.transpose(2,0,1)  # (3, 112, 112)
        image = np.expand_dims(image, axis=0) # (1, 3, 112, 112)
        
        img = torch.Tensor( (image*255.).astype('uint8') ).type(torch.FloatTensor)
        embedding = self.Face_Recognition_Network.transform(img.to(self.device) )
        image = image[0] # range (0,1) and shape (3, 112, 112)

        image = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        

        image_HQ = image_HQ.transpose(2,0,1)  # (3, 256, 256)
        image_HQ = torch.Tensor( image_HQ ).type(torch.FloatTensor).to(self.device)
        
        return embedding, image, image_HQ
    
    def transform_image(self,image):
        image = torch.Tensor(image).to(self.device)
        return image
    
    def transform_embedding(self, embedding):
        embedding = embedding.view(-1).to(self.device)
        return embedding