"""
SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute
<hatef.otroshi@idiap.ch>

SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>

SPDX-License-Identifier: BSD-3-Clause

--------------
Training code for GaFaR (Geometry-aware Face Reconstruction)
--------------
Papers (citation):
    [TPAMI] Hatef Otroshi Shahreza and Sébastien Marcel, "Comprehensive Vulnerability Evaluation of Face Recognition Systems 
    to Template Inversion Attacks Via 3D Face Reconstruction", IEEE Transactions on Pattern Analysis and Machine 
    Intelligence, 2023.
    [ICCV] Hatef Otroshi Shahreza and Sébastien Marcel, "Template Inversion Attack against Face Recognition Systems using 3D 
    Face Reconstruction", IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
"""
import argparse
parser = argparse.ArgumentParser(description='Train face reconstruction network - GaFaR')
parser.add_argument('--path_eg3d_repo', metavar='<path_eg3d_repo>', type= str, default='./eg3d',
                    help='./eg3d')
parser.add_argument('--path_eg3d_checkpoint', metavar='<path_eg3d_checkpoint>', type= str, default='./ffhq512-128.pkl',
                    help='./ffhq512-128.pkl')
parser.add_argument('--path_ffhq_dataset', metavar='<path_ffhq_dataset>', type= str, default='./Flickr-Faces-HQ/images1024x1024',
                    help='FFHQ directory')
parser.add_argument('--FR_system', metavar='<FR_system>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace (FR system from whose database the templates are leaked)')
parser.add_argument('--FR_loss', metavar='<FR_loss>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace (same model as FR_loss in whitebox and a different proxy model in blackbox attacks)')
args = parser.parse_args()



import os,sys
sys.path.append(os.getcwd()) # import src
sys.path.append(f"{args.path_eg3d_repo}/eg3d") # import eg3d files
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

import pickle
import torch
import torch_utils

import random
import numpy as np
import cv2
from tqdm import tqdm

seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

#=================== import Network =====================
path_EG3D = args.path_eg3d_checkpoint

with open(path_EG3D, 'rb') as f:
    EG3D = pickle.load(f)['G_ema']
    EG3D.to(device)
    EG3D.eval()
    EG3D_synthesis = EG3D.synthesis
    EG3D_mapping   = EG3D.mapping


from src.Network import Discriminator, MappingNetwork 
model_Discriminator = Discriminator()
model_Discriminator.to(device)

new_mapping = MappingNetwork(z_dim = 16,                      # Input latent (Z) dimensionality.
                             c_dim = 512,                       # Conditioning label (C) dimensionality, 0 = no labels.
                             w_dim = 512,                      # Intermediate latent (W) dimensionality.
                             num_ws = 14,                      # Number of intermediate latents to output.
                             num_layers = 2,                   # Number of mapping layers.
                            )
new_mapping.to(device)
z_dim_new_mapping = new_mapping.z_dim
z_dim_EG3D        = EG3D.z_dim
z_dim_EG3D = 512
#========================================================

#=================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(dataset_dir=args.path_ffhq_dataset, FR_system= args.FR_system, train=True,  device=device)
testing_dataset  = MyDataset(dataset_dir=args.path_ffhq_dataset, FR_system= args.FR_system, train=False, device=device)

train_dataloader = training_dataset
test_dataloader  = DataLoader(testing_dataset,  batch_size=18, shuffle=False)
#========================================================


#=================== Optimizers =========================
# ***** optimizer_Generator
for param in new_mapping.parameters():
    param.requires_grad = True

# ***** optimizer_Generator
optimizer1_Generator    = torch.optim.Adam(new_mapping.parameters(), lr=1e-1)
scheduler1_Generator    = torch.optim.lr_scheduler.StepLR(optimizer1_Generator, step_size=3, gamma=0.5)

optimizer2_Generator    = torch.optim.Adam(new_mapping.parameters(), lr=1e-1)
scheduler2_Generator    = torch.optim.lr_scheduler.StepLR(optimizer2_Generator, step_size=3, gamma=0.5)

optimizer3_Generator    = torch.optim.Adam(new_mapping.parameters(), lr=1e-1)
scheduler3_Generator    = torch.optim.lr_scheduler.StepLR(optimizer3_Generator, step_size=3, gamma=0.5)
# ***** optimizer_Discriminator
optimizer_Discriminator = torch.optim.Adam(model_Discriminator.parameters(), lr=1e-1)
scheduler_Discriminator = torch.optim.lr_scheduler.StepLR(optimizer_Discriminator, step_size=3, gamma=0.5)
#========================================================



#=================== import Loss ========================
# ***** ID_loss
from src.loss.FaceIDLoss import ID_Loss
ID_loss = ID_Loss(FR_system= args.FR_system, FR_loss= args.FR_loss, device=device)

# ***** Other losses
Pixel_loss = torch.nn.MSELoss()
w_loss = torch.nn.MSELoss()
#========================================================


#=================== Save models and logs ===============
import os
os.makedirs('training_files',exist_ok=True)
os.makedirs('training_files/models',exist_ok=True)
os.makedirs('training_files/Reconstructed_images',exist_ok=True)
os.makedirs('training_files/logs_train',exist_ok=True)

with open('training_files/logs_train/generator.csv','w') as f:
    f.write("epoch,Pixel_loss_Gen,W_loss_Gen,ID_loss_Gen,total_loss\n")

with open('training_files/logs_train/log.txt','w') as f:
    pass

saved_original_figures = False

#=================== Train ==============================
num_epochs=18
iterations_per_epoch_train=4500
iterations_per_test=150
batch_size = 6

FFHQ_align_mask = train_dataloader.FFHQ_align_mask.repeat(batch_size,1,1,1)
for epoch in range(num_epochs):  
    print(f'epoch: {epoch}, \t learning rate: {optimizer1_Generator.param_groups[0]["lr"]}')
    
    torch.random.manual_seed(epoch)
    for iteration in tqdm(range(iterations_per_epoch_train)):

        # =========================================== Teacher-Force using pretrained EG3D ===========================================   
        # generate images using EG3D 
        fov_deg = 18.837
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        
        z = torch.randn([batch_size, z_dim_EG3D]).to(device)    # latent codes
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
        camera_params = camera_params.repeat(batch_size,1)
        w = EG3D_mapping(z, camera_params)
        img = EG3D_synthesis(w, camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation
        
        # calculate embeddings of images
        embedding_db = ID_loss.get_embedding_db(img)
        embedding = ID_loss.get_embedding(img)

        # ===> now we have (embedding, w,  and img)



        # Reconstruct image from embedding with same camera params
        new_mapping.train()

        fov_deg = 18.837
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
        camera_params = camera_params.repeat(batch_size,1)

        z = torch.randn([batch_size, z_dim_new_mapping]).to(device)    # latent codes
        w_reconstructed = new_mapping(z, embedding_db)
        img_reconstructed = EG3D_synthesis(w_reconstructed, camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation
        
        # calculate embeddings of images
        embedding_reconstructed = ID_loss.get_embedding(img_reconstructed)

        ### =============== Calculate Loss ============

        ID  = ID_loss(embedding_reconstructed, embedding)
        Pixel = Pixel_loss(img_reconstructed, img)
        W = w_loss(w_reconstructed,w)

        loss_train_new_mapping = Pixel + ID + W
        
        # ================== backward =================
        optimizer1_Generator.zero_grad()
        loss_train_new_mapping.backward()
        optimizer1_Generator.step()

        # =========================================================================================================================== 

        # =========================================== Trainin using FFHQ dataset ====================================================
        # 
        fov_deg = 18.837 # https://github.com/NVlabs/eg3d/blob/870300f29f4058b8c5088ca79e926762745e40b8/docs/visualizer_guide.md#fov
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
        camera_params = camera_params.repeat(batch_size,1)

        embedding_db, real_image, real_image_HQ  = train_dataloader.get_batch(batch_idx=iteration, batch_size=batch_size)
        
        if iteration % 4 == 0:   
            """
            ******************* GAN: Update Discriminator *******************
            """
            new_mapping.eval()
            model_Discriminator.train()

            # Generate batch of latent vectors
            z = torch.randn([batch_size, z_dim_new_mapping]).to(device)    # latent codes
            w_fake = new_mapping(z=z, c=embedding_db).detach()

            noise = torch.randn(embedding_db.size(0), z_dim_EG3D, device=device)
            w_real = EG3D_mapping(z=noise, c=camera_params).detach()
            # ==================forward==================
            # disc should give lower score for real and high for gnerated (fake)
            output_discriminator_real = model_Discriminator(w_real)
            errD_real  = output_discriminator_real

            output_discriminator_fake  = model_Discriminator(w_fake)
            errD_fake  = (-1) * output_discriminator_fake

            loss_GAN_Discriminator = (errD_fake + errD_real).mean()
            # ==================backward=================
            optimizer_Discriminator.zero_grad()
            loss_GAN_Discriminator.backward()
            optimizer_Discriminator.step()

            for param in model_Discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)
        
        
        if iteration % 2 == 0:   
            new_mapping.train()
            model_Discriminator.eval()           
            """
            ******************* GAN: Update Generator *******************
            """
            # Generate batch of latent vectors
            z = torch.randn([batch_size, z_dim_new_mapping]).to(device)    # latent codes
            w_fake = new_mapping(z=z, c=embedding_db)
            # ==================forward==================
            output_discriminator_fake  = model_Discriminator(w_fake)
            
            loss_GAN_Generator  = output_discriminator_fake.mean()
            # ==================backward=================
            optimizer2_Generator.zero_grad()
            loss_GAN_Generator.backward()
            optimizer2_Generator.step()
    
        # if iteration % 2 == 0:   
        new_mapping.train()
        """
        ******************* Train Generator *******************
        """
        # ==================forward==================
        z = torch.randn([batch_size, z_dim_new_mapping]).to(device)    # latent codes
        w = new_mapping(z=z, c=embedding_db)
        img_reconstructed = EG3D_synthesis(w, c=camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation
        
        # calculate embeddings of images
        embedding_reconstructed = ID_loss.get_embedding(img_reconstructed)
        embedding               = ID_loss.get_embedding(real_image_HQ)

        ID  = ID_loss(embedding_reconstructed, embedding)

        Pixel = Pixel_loss( ( torch.clamp(img_reconstructed*FFHQ_align_mask, min=-1, max=1) + 1) / 2.0 ,real_image_HQ*FFHQ_align_mask)
        
        loss_train_Generator = Pixel + ID 
        
        # ==================backward=================
        optimizer3_Generator.zero_grad()
        loss_train_Generator.backward()#(retain_graph=True)
        optimizer3_Generator.step()

    
        # =========================================================================================================================== 
        
        # ================== log ======================
        iteration +=1
        if iteration % 200 == 0:
            with open('training_files/logs_train/log.txt','a') as f:
                f.write(f'epoch:{epoch+1}, \t iteration: {iteration}, \t loss_train_new_mapping:{loss_train_new_mapping.data.item()}\n')
            pass

    # ====================== Evaluation ===============
    new_mapping.eval()
    ID_loss_Gen_test = Pixel_loss_Gen_test = W_loss_Gen_test = total_loss_Gen_test = 0
    torch.random.manual_seed(1000)
    for iteration in range(iterations_per_test):
        # ==================forward==================
        with torch.no_grad():
            # generate images using EG3D 
            fov_deg = 18.837
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            intrinsics = FOV_to_intrinsics(fov_deg, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
            camera_params = camera_params.repeat(batch_size,1)

            z = torch.randn([batch_size, z_dim_EG3D]).to(device)    # latent codes
            w = EG3D_mapping(z, camera_params)
            img = EG3D_synthesis(w, camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation
            
            # calculate embeddings of images
            embedding_db = ID_loss.get_embedding_db(img)
            embedding = ID_loss.get_embedding(img)

            # Reconstruct image from embedding with same camera params
            z = torch.randn([batch_size, z_dim_new_mapping]).to(device)    # latent codes
            w_reconstructed = new_mapping(z, embedding_db)
            img_reconstructed = EG3D_synthesis(w_reconstructed, camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation
            
            embedding_reconstructed = ID_loss.get_embedding(img_reconstructed)

            ID  = ID_loss(embedding_reconstructed, embedding)
            # Pixel = Pixel_loss(img_reconstructed, img)
            Pixel = Pixel_loss( ( torch.clamp(img_reconstructed*FFHQ_align_mask, min=-1, max=1) + 1) / 2.0 ,img*FFHQ_align_mask)
            W = w_loss(w_reconstructed,w)

            total_loss_Generator = Pixel + ID + W
            #### 
            ID_loss_Gen_test  += ID.item()
            Pixel_loss_Gen_test += Pixel.item()
            W_loss_Gen_test += W.item()
            total_loss_Gen_test += total_loss_Generator.item()

    with open('training_files/logs_train/generator.csv','a') as f:
        f.write(f"{epoch+1}, {Pixel_loss_Gen_test/iteration}, {W_loss_Gen_test/iteration}, {ID_loss_Gen_test/iteration}, {total_loss_Gen_test/iteration}\n")
        



    # generate images using EG3D 
    fov_deg = 18.837
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
    camera_params = camera_params.repeat(batch_size,1)

    z = torch.randn([batch_size, z_dim_EG3D]).to(device)    # latent codes
    img = EG3D(z, camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation
    
    # calculate embeddings of images
    embedding_db = ID_loss.get_embedding_db(img)

    # Reconstruct image from embedding with same camera params
    z = torch.randn([batch_size, z_dim_new_mapping]).to(device)    # latent codes
    w = new_mapping(z=z, c=embedding_db)
    img_reconstructed = EG3D_synthesis(w, camera_params)['image']    # NCHW, float32, dynamic range [-1, +1], no truncation

    img_reconstructed = img_reconstructed.detach()



    if not saved_original_figures:
        saved_original_figures = True
        for i in range(img_reconstructed.size(0)):
            im = img[i].squeeze()
            im =  (torch.clamp(im, min=-1, max=1) + 1) / 2.0
            im = (im.cpu().numpy().transpose(1,2,0))
            im = (im * 255).astype(int)
            os.makedirs(f'training_files/Reconstructed_images/{i}',exist_ok=True)
            cv2.imwrite(f'training_files/Reconstructed_images/{i}/original.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
        
    for i in range(img_reconstructed.size(0)):
        img = img_reconstructed[i].squeeze()
        img =  (torch.clamp(img, min=-1, max=1) + 1) / 2.0
        im = (img.cpu().numpy().transpose(1,2,0))
        im = (im * 255).astype(int)
        cv2.imwrite(f'training_files/Reconstructed_images/{i}/epoch_{epoch+1}.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
    # *******************************************************
    

    # Save models
    torch.save(new_mapping.state_dict(), 'training_files/models/new_mapping_{}.pth'.format(epoch+1))
    # torch.save(model_Discriminator.state_dict(), 'training_files/models/Discriminator_{}.pth'.format(epoch+1))
    
    # Update schedulers
    scheduler1_Generator.step()
    scheduler2_Generator.step()
    scheduler3_Generator.step()
    scheduler_Discriminator.step()
#========================================================
 