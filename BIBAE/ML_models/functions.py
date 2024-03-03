# imports for COMET ML logging and plotting
from comet_ml import Experiment
from API_keys import api_key
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import autograd
import os
import time
tf_logscale_rev = lambda x:(torch.exp(x)-1.0)
tf_logscale_rev_np = lambda x:(np.exp(x)-1.0)
tf_logscale_ML = lambda x:(np.log((skimage.measure.block_reduce(x, (6,1,1), np.sum)+1.0)))

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    isApex = True
except:
    isApex = False
    
import math    
    
import subprocess
import time

from torch.autograd.variable import Variable


# create CometML experiment to see how training is progressing in real time
experiment = Experiment(
    api_key=api_key,
    project_name="YOUR PROJECT NAME",
    workspace="YOUR WORK SPACE",
)


experiment.add_tag('BIBAE PP- test')



# Training cut and calibration
def tf_Photons_thresh_calib_tensor(x):
    x[x < 0.0001] = 0.0 

    # Calibration factor in ILD config is 2.03
    x_calib = torch.empty_like(x).copy_(x)    
    
    for i in range(30):
        if i>=20:
            x_calib[:,i,:,:] = 2.03*x[:,i,:,:]    

    return x_calib

# MIP cut for spectrum
def tf_MIP_thresh(x):
    x[x < 0.1] = 0.0
    return x


###########################################################################
######## BIB-AE Energy and Angular Conditioning Gradient Penalty ##########
###########################################################################

def calc_gradient_penalty_EandA_F_DIRENR(netD, real_data, fake_data, device, batchsize, E_true, theta_true, phi_true, LAMBDA = 10.0):
    alpha = torch.rand(batchsize, 1, device=device)
    alpha = alpha.expand(batchsize, 
                         int(real_data.nelement()/batchsize)).contiguous().view(batchsize, 
                         real_data.size(1), real_data.size(2), real_data.size(3), real_data.size(4))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    
    interpolates = interpolates.clone().detach().requires_grad_(True)
    disc_interpolates = netD(interpolates, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA  

    return gradient_penalty


def calc_gradient_penalty_AAE(netD, real_data, fake_data, device, batchsize, LAMBDA = 10.0):
    alpha = torch.rand(batchsize, 1, device=device)
    alpha = alpha.expand(batchsize,
                         int(real_data.nelement()/batchsize)).contiguous().view(batchsize,
                         real_data.size(1))
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.clone().detach().requires_grad_(True) 
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size(),device=device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

###########################################################################
######## BIB-AE Training Energy and Two Angle Conditioning Reset ##########
###########################################################################
    
    
def train_BiBAE_F_linear_Eand2A_Reset(model, netD, netD_L, epoch, train_loader, device, args, netD_Reset,
                optimizer, optimizerD, optimizerD_L, L_Losses, L_D, L_D_L, optimizerD_Reset,
                         scheduler=None, schedulerD=None, schedulerD_L=None):
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)

    E_true_trans = args['E_true_trans']
        
    model.train()
    train_loss = 0

    
    
    for batch_idx, (data, energy, theta, phi) in enumerate(train_loader):
        data = data.to(device)
        input_size_1 = data.size(-3)
        input_size_2 = data.size(-2)
        input_size_3 = data.size(-1)
        
        E_true = E_true_trans(energy).to(device)
        theta_true = theta
        theta_true = theta_true.to(device)
        phi_true = phi
        phi_true = phi_true.to(device)
        for p in netD.parameters():
            p.requires_grad = True  # to avoid computation
        for p in netD_Reset.parameters():
            p.requires_grad = True  # to avoid computation
        for p in netD_L.parameters():
            p.requires_grad = True  # to avoid computation
  
        
        for i in range(5):
            
            ####################################
            ######## Critic Continuous #########
            ####################################
            recon_batch, mu, logvar, z = model(data, E_true, theta_true, phi_true)

            optimizerD.zero_grad()            
            
            
            real_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                                   data.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
            
            D_real = netD(real_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
            D_real = D_real.mean()
            LossD = D_real*(-1.0)
            

            fake_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                            recon_batch.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
            fake_data = fake_data.clone().detach()

            D_fake = netD(fake_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
            D_fake = D_fake.mean()
            LossD += D_fake*(1.0)
            
            gradient_penalty = calc_gradient_penalty_EandA_F_DIRENR(netD, real_data.data, fake_data.data, device, data.size(0), 
                                                    E_true=E_true, theta_true=theta_true, phi_true=phi_true, LAMBDA = 10.0)  
            LossD += gradient_penalty

            LossD.backward()
            optimizerD.step()
            
            
            ####################################
            ########## Critic Reset ############
            ####################################
            optimizerD_Reset.zero_grad()            
            
            
            real_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                                   data.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
            
            D_Reset_real = netD_Reset(real_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
            D_Reset_real = D_Reset_real.mean()
            LossD_Reset = D_Reset_real*(-1.0)
            

            fake_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                            recon_batch.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
            fake_data = fake_data.clone().detach()

            D_Reset_fake = netD_Reset(fake_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
            D_Reset_fake = D_Reset_fake.mean()
            LossD_Reset += D_Reset_fake*(1.0)
            
            gradient_penalty = calc_gradient_penalty_EandA_F_DIRENR(netD_Reset, real_data.data, fake_data.data, device, data.size(0), 
                                                    E_true=E_true, theta_true=theta_true, phi_true=phi_true, LAMBDA = 10.0)  
            LossD_Reset += gradient_penalty

            LossD_Reset.backward()
            optimizerD_Reset.step()

            
            ####################################
            ########## Critic Latent ###########
            ####################################
            optimizerD_L.zero_grad()
    
            zv = z.clone().detach()
         
            realv = torch.randn_like(zv)
        
            D_L_real = netD_L(realv)
            D_L_real = D_L_real.mean()
            lossDL = D_L_real*(-1.0)

            D_L_fake = netD_L(zv)
            D_L_fake = D_L_fake.mean()
            lossDL += D_L_fake*(1.0)

            gradient_penalty_L = calc_gradient_penalty_AAE(netD_L, realv.data, zv.data, device, realv.size(0),
                                                    LAMBDA = 10.0)                                                 
            lossDL += gradient_penalty_L
            
            lossDL.backward()
            optimizerD_L.step()

            
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netD_Reset.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netD_L.parameters():
            p.requires_grad = False  # to avoid computation
               
        recon_batch, mu, logvar, z = model(data, E_true, theta_true, phi_true)

        optimizer.zero_grad()
 
        sum_data = torch.sum((data.view(-1, input_size_1*input_size_2*input_size_3)), dim=1)*1e-4 #GeV to TeV   
        sum_recon = torch.sum((recon_batch.view(-1, input_size_1*input_size_2*input_size_3)), dim=1)*1e-4 #GeV to TeV

        fake_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                        recon_batch.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
        lossD = netD(fake_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
        lossD = lossD.mean()

        lossD_Reset = netD_Reset(fake_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
        lossD_Reset = lossD_Reset.mean()
        
        lossD_L = netD_L(z)
        lossD_L = lossD_L.mean()
        
        lossFix,weighted,unwe,name = loss_function_VAE_ENR(recon_x=recon_batch, x=data, mu=mu, logvar=logvar, 
                                                           E_pred=E_true, E_true=E_true, E_sum_pred=sum_recon, E_sum_true=sum_data, 
                                                           device=device, L_Losses = L_Losses, z=z, args=args)   
        
        loss = L_D*lossD + L_D*lossD_Reset + L_D_L*lossD_L - lossFix
        
        train_loss += loss.item()     
        
        loss.backward(mone)
        
        ###########################################################
        # Log to COMET ML
        ###########################################################

        experiment.log_metric('lossD', L_D*lossD.item(), step=epoch*batch_idx, epoch=epoch)
        experiment.log_metric('lossD_L', L_D_L*lossD_L.item(), step=epoch*batch_idx, epoch=epoch)
        experiment.log_metric('lossFix', lossFix.item(), step=epoch*batch_idx, epoch=epoch)

        ### plot example image
        if batch_idx % args["log_interval"] == 0 or batch_idx == 0:
            image = recon_batch.view(-1, 30, 30, 49).cpu().detach().numpy()
            cmap = mpl.cm.viridis
            cmap.set_bad('white',1.)
            figExIm = plt.figure(figsize=(6,6))
            axExIm1 = figExIm.add_subplot(1,1,1)
            image1 = np.sum(image[0], axis=0)
            masked_array1 = np.ma.array(image1, mask=(image1==0.0))
            im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100, norm=mpl.colors.LogNorm(), origin='lower')
            figExIm.patch.set_facecolor('white')
            axExIm1.set_xlabel('y [cells]', family='serif')
            axExIm1.set_ylabel('x [cells]', family='serif')
            figExIm.colorbar(im1)
            experiment.log_figure(figure=plt, figure_name="x-y")
            plt.close(figExIm)


            figExIm = plt.figure(figsize=(6,6))
            axExIm2 = figExIm.add_subplot(1,1,1)
            image2 = np.sum(image[0], axis=1)
            masked_array2 = np.ma.array(image2, mask=(image2==0.0))
            im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100, norm=mpl.colors.LogNorm(), origin='lower') 
            figExIm.patch.set_facecolor('white')
            axExIm2.set_xlabel('y [cells]', family='serif')
            axExIm2.set_ylabel('z [layers]', family='serif')
            figExIm.colorbar(im2)

            experiment.log_figure(figure=plt, figure_name="y-z")
            plt.close(figExIm)

            figExIm = plt.figure(figsize=(6,6))
            axExIm3 = figExIm.add_subplot(1,1,1)
            image3 = np.sum(image[0], axis=2)
            masked_array3 = np.ma.array(image3, mask=(image3==0.0))
            im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,  norm=mpl.colors.LogNorm(), origin='lower')
            figExIm.patch.set_facecolor('white')
            axExIm3.set_xlabel('x [cells]', family='serif')
            axExIm3.set_ylabel('z [layers]', family='serif')
            figExIm.colorbar(im3)
            experiment.log_figure(figure=plt, figure_name="x-z")

            plt.close(figExIm)
        
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\lossD: {:.6f} lossD_L: {:.6f} Fix {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                L_D*lossD.item(),L_D_L*lossD_L.item(), lossFix.item()
            ))
            line=''
            
        if not args["SafeAfterEpochOnly"]:
            if (epoch%args["save_interval"] == 0 or epoch == 1) and (batch_idx % args["save_iter_interval"] == 0):
                print('Saving to ' + args["output_path"] + "check_" +
                    args["model"] + args["suffix"] + '_' + str(epoch) + '_' + str(batch_idx) + '.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_L_state_dict': netD_L.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'optimizerD_L_state_dict': optimizerD_L.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'schedulerD_state_dict': schedulerD.state_dict(),
                    'schedulerD_L_state_dict': schedulerD_L.state_dict()
                    }, 
                    args["output_path"] + "check_" +
                    args["model"] + args["suffix"] + '_' + str(epoch) + '_' + str(batch_idx) + '.pth'
                )
            
            
    for (w, u, n) in zip(weighted, unwe, name):
        line = line + n + ' \t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))


#### functions to get energy sum and number of hits in shower
# Total visible energy
def getTotE(data, xbins=30, ybins=49, layers=30):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr

# Occupancy
def getOcc(data, xbins=30, ybins=49, layers=30):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    #occ_arr = np.count_nonzero(data, axis=(1))
    #print(data.min())
    occ_arr = (data > 0.0).sum(axis=(1))
    return occ_arr  

#############################################################################################
######## BIB-AE Training Energy and Two Angle Conditioning Reset: Post Processing ##########
#############################################################################################
##### THIS MODEL ONLY CONDITIONS THE PP ON THE ENERGY SUM (NO ESUM conditioning for BIBAE base or Discriminator)!!!!
    
    
def train_BiBAE_F_linear_Eand2A_Reset_withA_cond_PP_ESum(model, modelP, netD, netD_L, epoch, train_loader, device, args,
                optimizer, optimizerP, optimizerD, optimizerD_L, L_Losses, L_Losses_P, L_D, L_D_L, L_D_P, rank=0, scheduler=None, schedulerD=None, schedulerD_L=None, record_network=False): 
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)

    E_true_trans = args['E_true_trans']  
        
    model.train()
    train_loss = 0

    
    
    for batch_idx, (data, energy, theta, phi) in enumerate(train_loader):
        data = data.to(device)
        input_size_1 = data.size(-3)
        input_size_2 = data.size(-2)
        input_size_3 = data.size(-1)
        
        #### data sum with MIP cut
        data_MIP_cut = tf_MIP_thresh(data.cpu().numpy())
        
        
        Esum = getTotE(data_MIP_cut, xbins=30, ybins=49, layers=30)
        Esum = torch.from_numpy(np.float32(Esum.reshape(-1, 1))).float()
        
        E_true = E_true_trans(energy).to(device)
        theta_true = theta
        theta_true = theta_true.to(device)
        phi_true = phi
        phi_true = phi_true.to(device)
        batch_size_n = data.size(0)
        Esum = Esum.to(device)
        
        if args['BIBAE_Train']:
            for p in netD.parameters():
                p.requires_grad = True  # to avoid computation
            for p in netD_L.parameters():
                p.requires_grad = True  # to avoid computation


            for i in range(5):

                ####################################
                ######## Critic Continuous #########
                ####################################
                recon_batch, mu, logvar, z = model(data, E_true, theta_true, phi_true)

                optimizerD.zero_grad()            


                real_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                                       data.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)

                D_real = netD(real_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
                D_real = D_real.mean()
                LossD = D_real*(-1.0)


                fake_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                                recon_batch.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
                fake_data = fake_data.clone().detach()

                D_fake = netD(fake_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
                D_fake = D_fake.mean()
                LossD += D_fake*(1.0)

                gradient_penalty = calc_gradient_penalty_EandA_F_DIRENR(netD, real_data.data, fake_data.data, device, data.size(0), 
                                                        E_true=E_true, theta_true=theta_true, phi_true=phi_true, LAMBDA = 10.0)  
                LossD += gradient_penalty

                LossD.backward()
                optimizerD.step()


                ####################################
                ########## Critic Latent ###########
                ####################################
                optimizerD_L.zero_grad()

                zv = z.clone().detach()

                realv = torch.randn_like(zv)

                D_L_real = netD_L(realv)
                D_L_real = D_L_real.mean()
                lossDL = D_L_real*(-1.0)

                D_L_fake = netD_L(zv)
                D_L_fake = D_L_fake.mean()
                lossDL += D_L_fake*(1.0)

                gradient_penalty_L = calc_gradient_penalty_AAE(netD_L, realv.data, zv.data, device, realv.size(0),
                                                        LAMBDA = 10.0)                                                 
                lossDL += gradient_penalty_L

                lossDL.backward()
                optimizerD_L.step()


            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netD_L.parameters():
                p.requires_grad = False  # to avoid computation
               
        recon_batch, mu, logvar, z = model(data, E_true, theta_true, phi_true)

        optimizer.zero_grad()
 
        sum_data = torch.sum((data.view(-1, input_size_1*input_size_2*input_size_3)), dim=1)*1e-4 #GeV to TeV   
        sum_recon = torch.sum((recon_batch.view(-1, input_size_1*input_size_2*input_size_3)), dim=1)*1e-4 #GeV to TeV

        fake_data = torch.cat((data.view(-1,1,input_size_1,input_size_2,input_size_3), 
                        recon_batch.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
        lossD = netD(fake_data, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
        lossD = lossD.mean()

        
        lossD_L = netD_L(z)
        lossD_L = lossD_L.mean()
        
        lossFix,weighted,unwe,name = loss_function_VAE_ENR(recon_x=recon_batch, x=data, mu=mu, logvar=logvar, 
                                                           E_pred=E_true, E_true=E_true, E_sum_pred=sum_recon, E_sum_true=sum_data, 
                                                           device=device, L_Losses = L_Losses, z=z, args=args)   
        
        loss = L_D*lossD + L_D_L*lossD_L - lossFix
        
        train_loss += loss.item()     
        
        if args['BIBAE_Train']:
            loss.backward(mone)
            optimizer.step()
        
        
        ####### Post Process- here with angular conditioning #########################
        postprocess_batch = modelP(recon_batch.detach(), E_True=E_true, theta_True=theta_true, phi_True=phi_true, ESum=Esum)
        
        sum_postprocess = torch.sum((postprocess_batch.view(-1, input_size_1*input_size_2*input_size_3)), dim=1) #GeV to TeV
        
        optimizerP.zero_grad()
        
        if L_D_P == 0:
            lossD_P = torch.tensor(0, device=device)
        else:
            fake_data_pp = torch.cat((postprocess_batch.view(-1,1,input_size_1,input_size_2,input_size_3), 
                                      postprocess_batch.view(-1,1,input_size_1,input_size_2,input_size_3)), 1)
            lossD_P = netD(fake_data_pp, E_true=E_true, theta_true=theta_true, phi_true=phi_true)
            lossD_P = lossD_P.mean()
            
        if epoch <= args['start_PostProc_after_ep']:
            #run mse loss to get postProc network to reporoduce orig showers for the first section
            if args['PostProc_pretrain'] == 'MSE':
                lossFixP = F.mse_loss(postprocess_batch.view(batch_size_n, -1), 
                                      recon_batch.detach().view(batch_size_n, -1), reduction='mean')
            elif args['PostProc_pretrain'] == 'MAE_Rela':
                lossFixP =  torch.mean(torch.abs(
                              (recon_batch.view(batch_size_n, -1).detach()/sum_recon.view(-1, 1).detach()) -
                              (postprocess_batch.view(batch_size_n, -1)/sum_postprocess.view(-1, 1).detach())
                            ))*2000.0 
        
        else:
            lossFixP,weightedP,unweP,nameP = loss_function_VAE_ENR(recon_x=postprocess_batch, x=data, mu=mu, logvar=logvar, 
                                                           E_pred=E_true, E_true=E_true, E_sum_pred=sum_postprocess, 
                                                           E_sum_true=sum_data, 
                                                           device=device, L_Losses = L_Losses_P, z=z, args=args)   

            if args['PostProc_train'] == 'MSE':
                lossFixP += F.mse_loss(postprocess_batch.view(batch_size_n, -1), 
                                       recon_batch.detach().view(batch_size_n, -1), reduction='mean')
            elif args['PostProc_train'] == 'MAE_Rela':
                lossFixP += torch.mean(torch.abs(
                              (recon_batch.view(batch_size_n, -1).detach()/sum_recon.view(-1, 1).detach()) -
                              (postprocess_batch.view(batch_size_n, -1)/sum_postprocess.view(-1, 1).detach())
                            ))*2000.0
        
        
        lossP = L_D_P*lossD_P - lossFixP
        
        if record_network:
            network_and_time(world=rank)
            
        lossP.backward(mone)
        optimizerP.step()
        
        
        if record_network:
            network_and_time(world=rank)
            
        
        
        ###########################################################
        # Log to COMET ML
        ###########################################################
        
        experiment.log_metric('lossD', L_D*lossD.item(), step=epoch*batch_idx, epoch=epoch)
        experiment.log_metric('lossD_L', L_D_L*lossD_L.item(), step=epoch*batch_idx, epoch=epoch)
        experiment.log_metric('lossFix', lossFix.item(), step=epoch*batch_idx, epoch=epoch)
        
        experiment.log_metric('lossD_P', L_D_P*lossD_P.item(), step = epoch*batch_idx, epoch=epoch)
        experiment.log_metric('lossFix_P', lossFixP.item(), step = epoch*batch_idx, epoch=epoch)
        
        ### plot example image
        if batch_idx % args["log_interval"] == 0 or batch_idx == 0:
            image = recon_batch.view(-1, 30, 30, 49).cpu().detach().numpy()
            cmap = mpl.cm.viridis
            cmap.set_bad('white',1.)
            figExIm = plt.figure(figsize=(6,6))
            axExIm1 = figExIm.add_subplot(1,1,1)
            image1 = np.sum(image[0], axis=0)
            masked_array1 = np.ma.array(image1, mask=(image1==0.0))
            im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100, norm=mpl.colors.LogNorm(), origin='lower')
            figExIm.patch.set_facecolor('white')
            axExIm1.set_xlabel('y [cells]', family='serif')
            axExIm1.set_ylabel('x [cells]', family='serif')
            figExIm.colorbar(im1)
            experiment.log_figure(figure=plt, figure_name="x-y")
            plt.close(figExIm)

            figExIm = plt.figure(figsize=(6,6))
            axExIm2 = figExIm.add_subplot(1,1,1)
            image2 = np.sum(image[0], axis=1)
            masked_array2 = np.ma.array(image2, mask=(image2==0.0))
            im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100, norm=mpl.colors.LogNorm(), origin='lower') 
            figExIm.patch.set_facecolor('white')
            axExIm2.set_xlabel('y [cells]', family='serif')
            axExIm2.set_ylabel('z [layers]', family='serif')
            figExIm.colorbar(im2)

            experiment.log_figure(figure=plt, figure_name="y-z")
            plt.close(figExIm)
            
            figExIm = plt.figure(figsize=(6,6))
            axExIm3 = figExIm.add_subplot(1,1,1)
            image3 = np.sum(image[0], axis=2)
            masked_array3 = np.ma.array(image3, mask=(image3==0.0))
            im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,  norm=mpl.colors.LogNorm(), origin='lower')
            figExIm.patch.set_facecolor('white')
            axExIm3.set_xlabel('x [cells]', family='serif')
            axExIm3.set_ylabel('z [layers]', family='serif')
            figExIm.colorbar(im3)
            experiment.log_figure(figure=plt, figure_name="x-z")
            
            plt.close(figExIm)
        
        
        
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\lossD: {:.6f} lossD_L: {:.6f} Fix {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                L_D*lossD.item(),L_D_L*lossD_L.item(), lossFix.item()
            ))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\lossD_P: {:.6f}  FixP {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                L_D_P*lossD_P.item(), lossFixP.item()
            ))
            line=''
            
            
    for (w, u, n) in zip(weighted, unwe, name):
        line = line + n + ' \t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)
    if epoch > args['start_PostProc_after_ep']:
        line=''
        for (w, u, n) in zip(weightedP, unweP, nameP):
            line = line + n + ' \t{:.3f} \t{:.3f} \n'.format(w,u)
        print(line)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))            

def loss_function_VAE_ENR(recon_x, x, mu, logvar, E_pred, E_true, E_sum_pred, E_sum_true, reduction='mean', device='cpu',
              L_Losses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                          z=None, args=None):
    B=(logvar.size(0)) #
    calo_size = x.size(-1)
    
    
    [L_MSE,          L_MSE_PS,     L_MSE_PM, 
     L_MSE_PL,       L_KLD,        L_MMD, 
     L_ENR_PRED,     L_ENR_RECON,  L_ENR_LATEN, 
     L_VAR,          L_MAE,        L_MLE,
     L_ENR_RECON_N,  L_E_LogSum,   L_E_SumLog,
     L_MMD_E,        L_MMD_Hit,    L_MMD_Latent,
     L_MMD_HitKS,    L_SortMSE,    L_SortMAE,
     L_MMD_Hit2,     L_MMD_HitKS2, L_ENR_Cut,
     L_ENR_Cut_N,    L_KLD_sq,     L_KLD_quad,
     L_Disco,        L_batchcomp,  L_bcompLog] = L_Losses
 
    empt = 0
    
    
    x_sorted = x.view(B, -1)
    recon_x_sorted = recon_x.view(B, -1)
    
    x_sorted, _ = torch.sort(x_sorted, dim=1, descending=True)
    recon_x_sorted, _ = torch.sort(recon_x_sorted, dim=1, descending=True)

    
    if L_Disco == 0:
        Disco = 0
    else:
        Disco = pairwise_disco3(mu=mu[:,0:args["latentSmL"]], logvar=logvar[:,0:args["latentSmL"]], power=args["DiscoPower"])

    if L_MSE == 0:
        MSE = 0
    else:
        MSE = F.mse_loss(recon_x.view(B, -1), x.view(B, -1), reduction=reduction)


    if L_MSE_PL == 0:
        MSE_PL = 0
    else:
        MSE_PL = mse_pool(x.view(-1, calo_size, calo_size, calo_size), recon_x.view(-1, calo_size, calo_size, calo_size), 
                          kernel=(10,10,10), reduction=reduction, stride=5) 
    if L_MSE_PM == 0:
        MSE_PM  = 0
    else:
        MSE_PM = mse_pool(x.view(-1, calo_size, calo_size, calo_size), recon_x.view(-1, calo_size, calo_size, calo_size), 
                          kernel=(5,5,5), reduction=reduction, stride=3)   
    if L_MSE_PS == 0:
        MSE_PS = 0
    else:
        MSE_PS = mse_pool(x.view(-1, calo_size, calo_size, calo_size), recon_x.view(-1, calo_size, calo_size, calo_size), 
                          kernel=(3,3,3), reduction=reduction, stride=2)
     


    if L_MAE == 0:
        MAE = 0
    else:
        MAE = meanAbsError_loss(recon_x.view(B, -1), x.view(B, -1), reduction=reduction)
    if L_MLE == 0:
        MLE = 0
    else:
        MLE = meanLogError_loss(recon_x.view(B, -1), x.view(B, -1), reduction=reduction)
    
    
    if L_MMD == 0:
        MMD = 0
    else:
        MMD = mmd_loss_rad(recon_x.view(-1, calo_size, calo_size, calo_size), x, device=device, sym = 'true')/(B*B)

    if L_KLD == 0:
        KLD = 0
    else:
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/(B)
        
    if L_KLD_sq == 0:
        KLD_sq = 0
    else:
        KLD_sq = (0.25 * torch.sum(
                                  (1 + logvar - mu.pow(2) - logvar.exp()).pow(2)
                                  ))/(B)
        
        
    if L_KLD_quad == 0:
        KLD_quad = 0
    else:
        KLD_quad = (0.25 * 0.25 * torch.sum(
                                    (1 + logvar - mu.pow(2) - logvar.exp()).pow(4)
                                    ))/(B)



    if L_VAR == 0:
        VAR = 0
    else:
        VAR = batch_VAR(recon_x.view(B, -1))

    if L_SortMSE == 0:
        SortMSE = 0
    else:  
        SortMSE = (x_sorted - recon_x_sorted).pow(2)
        SortMSE = torch.mean(SortMSE)

    if L_SortMAE == 0:
        SortMAE = 0
    else:  
        SortMAE = torch.abs(x_sorted - recon_x_sorted)
        SortMAE = torch.mean(SortMAE)
        
    if L_ENR_PRED == 0:
        ENR_PRED = 0
    else:  
        ENR_PRED  = F.mse_loss(E_pred, E_true, reduction = reduction) 

    if L_ENR_RECON == 0:
        ENR_RECON = 0
    else:  
        ENR_RECON = (E_sum_true - E_sum_pred).pow(2)
        ENR_RECON = torch.mean(ENR_RECON)

        
    if L_ENR_Cut == 0:
        ENR_Cut = 0
    else:
        ENR_Cut = (torch.sum(F.threshold(recon_x, args['ENR_Cut_Cutoff'], 0.0).view(B, -1), 1) - 
                   torch.sum(F.threshold(x, args['ENR_Cut_Cutoff'], 0.0).view(B, -1), 1)).pow(2)
        ENR_Cut = torch.mean(ENR_Cut)
 
    if L_ENR_Cut_N == 0:
        ENR_Cut_N = 0
    else:
        ENR_Cut_N = torch.abs((torch.sum(F.threshold(recon_x, args['ENR_Cut_Cutoff'], 0.0).view(B, -1), 1) - 
                   torch.sum(F.threshold(x, args['ENR_Cut_Cutoff'], 0.0).view(B, -1), 1)))
        ENR_Cut_N = torch.mean(ENR_Cut_N)



    if L_ENR_LATEN == 0:
        ENR_LATEN = 0
    else:  
        ENR_LATEN = 0
        
    if L_ENR_RECON_N == 0:
        ENR_RECON_N = 0
    else:  
        ENR_RECON_N = torch.abs(E_sum_true - E_sum_pred)
        ENR_RECON_N = torch.mean(ENR_RECON_N)

    if L_E_LogSum == 0:
        E_LogSum = 0
    else:        
        E_LogSum_pred = torch.log(torch.sum(tf_logscale_rev(recon_x.view(B, -1)), dim=1)) 
        E_LogSum_true = torch.log(torch.sum(tf_logscale_rev(x.view(B, -1)), dim=1)) 

        E_LogSum  = F.mse_loss(E_LogSum_pred, E_LogSum_true, reduction = reduction) 

    if L_E_SumLog == 0:
        E_SumLog = 0
    else:  
        E_SumLog_pred = torch.sum(recon_x.view(B, -1), dim=1)
        E_SumLog_true = torch.sum(x.view(B, -1), dim=1) 

        E_SumLog  = F.mse_loss(E_SumLog_pred, E_SumLog_true, reduction = reduction) 

    if L_MMD_E == 0:
        MMD_E = 0
    else:  
        MMD_E  = (mmd_energy_loss(E_sum_pred*1000, E_sum_true*1000, E_true, alpha=0.1)/(B*B) +
                  mmd_energy_loss(E_sum_pred*1000, E_sum_true*1000, E_true, alpha=1.0)/(B*B)+
                  mmd_energy_loss(E_sum_pred*1000, E_sum_true*1000, E_true, alpha=100.0)/(B*B)+
                  mmd_energy_loss(E_sum_pred*1000, E_sum_true*1000, E_true, alpha=1000.0)/(B*B))

    if L_MMD_Hit == 0:
        MMD_Hit = 0
    else:
        mmd2 = mmd_hit_loss_sort_cast_mean(recon_x_sorted, x_sorted, alpha=args["HitMMD_alpha"])
        

        MMD_Hit = torch.mean(mmd2)

    if L_MMD_HitKS == 0:
        MMD_HitKS = 0
    else:
        mmd1 = mmd_hit_sortKernel(recon_x_sorted, x_sorted, kernel_size = args["HitMMDKS_Ker"], 
                                  stride = args["HitMMDKS_Str"], cutoff = args["HitMMDKS_Cut"], 
                                  alpha = args["HitMMD_alpha"])        
        MMD_HitKS = (mmd1)


    if L_MMD_Hit2 == 0:
        MMD_Hit2 = 0
    else:
        mmd3 = mmd_hit_loss_sort_cast_mean(recon_x_sorted, x_sorted, alpha=args["HitMMD2_alpha"])
        

        MMD_Hit2 = torch.mean(mmd3)

    if L_MMD_HitKS2 == 0:
        MMD_HitKS2 = 0
    else:
        mmd4 = mmd_hit_sortKernel(recon_x_sorted, x_sorted, kernel_size = args["HitMMDKS_Ker"], 
                                  stride = args["HitMMDKS_Str"], cutoff = args["HitMMDKS_Cut"], 
                                  alpha = args["HitMMD2_alpha"])        
        MMD_HitKS2 = (mmd4)

        
    if L_MMD_Latent == 0:
        MMD_Latent = 0
    else:
        Bz = z.size(0)
        z_real = torch.randn_like(z)
        MMD_Latent = (mmd_latent_loss(z, z_real, alpha=1.0)/(Bz*Bz) + 
                      mmd_latent_loss(z, z_real, alpha=0.01)/(Bz*Bz))
        
        
    if L_batchcomp == 0:
        batchcomp = 0
    else: 
        batchcomp = (torch.sum(x, 0) - torch.sum(recon_x, 0))**2
        batchcomp = torch.mean(batchcomp)

        
    if L_bcompLog == 0:
        bcompLog = 0
    else: 
        torch.log(F.threshold(recon_x, 0.0001, 0.0001))
        
        bcompLog = (torch.sum(torch.log(F.threshold(x, 0.0001, 0.0001)), 0) - 
                    torch.sum(torch.log(F.threshold(recon_x, 0.0001, 0.0001)), 0))**2
        bcompLog = torch.mean(bcompLog)

        

    loss = (L_MSE*MSE                 + L_MSE_PS*MSE_PS           + L_MSE_PM*MSE_PM         + 
            L_MSE_PL*MSE_PL           + L_KLD*KLD                 + L_MMD*MMD               + 
            L_ENR_PRED*ENR_PRED       + L_ENR_RECON*ENR_RECON     + L_ENR_LATEN*ENR_LATEN   + 
            L_VAR*VAR                 + L_MAE*MAE                 + L_MLE*MLE               +
            L_ENR_RECON_N*ENR_RECON_N + L_E_LogSum*E_LogSum       + L_E_SumLog*E_SumLog     +
            L_MMD_E*MMD_E             + L_MMD_Hit*MMD_Hit         + L_MMD_Latent*MMD_Latent +
            L_MMD_HitKS*MMD_HitKS     + L_SortMSE*SortMSE         + L_SortMAE*SortMAE       +
            L_MMD_Hit2*MMD_Hit2       + L_MMD_HitKS2*MMD_HitKS2   + L_ENR_Cut*ENR_Cut       +
            L_ENR_Cut_N*ENR_Cut_N     + L_KLD_sq*KLD_sq           + L_KLD_quad*KLD_quad     +
            L_Disco*Disco             + L_batchcomp*batchcomp     + L_bcompLog*bcompLog
           )
    return loss, \
           [L_MSE*MSE,                 L_MSE_PS*MSE_PS,             L_MSE_PM*MSE_PM,  
            L_MSE_PL*MSE_PL,           L_KLD*KLD,                   L_MMD*MMD,      
            L_ENR_PRED*ENR_PRED,       L_ENR_RECON*ENR_RECON,       L_ENR_LATEN*ENR_LATEN,
            L_VAR*VAR,                 L_MAE*MAE,                   L_MLE*MLE,    
            L_ENR_RECON_N*ENR_RECON_N, L_E_LogSum*E_LogSum,         L_E_SumLog*E_SumLog,       
            L_MMD_E*MMD_E,             L_MMD_Hit*MMD_Hit,           L_MMD_Latent*MMD_Latent,
            L_MMD_HitKS*MMD_HitKS,     L_SortMSE*SortMSE,           L_SortMAE*SortMAE,
            L_MMD_Hit2*MMD_Hit2,       L_MMD_HitKS2*MMD_HitKS2,     L_ENR_Cut*ENR_Cut,
            L_ENR_Cut_N*ENR_Cut_N,     L_KLD_sq*KLD_sq,             L_KLD_quad*KLD_quad,
            L_Disco*Disco,             L_batchcomp*batchcomp,       L_bcompLog*bcompLog],\
           [MSE,         MSE_PS,     MSE_PM, 
            MSE_PL,      KLD,        MMD, 
            ENR_PRED,    ENR_RECON,  ENR_LATEN,
            VAR,         MAE,        MLE,              
            ENR_RECON_N, E_LogSum,   E_SumLog,              
            MMD_E,       MMD_Hit,    MMD_Latent,
            MMD_HitKS,   SortMSE,    SortMAE,
            MMD_Hit2,    MMD_HitKS2, ENR_Cut,
            ENR_Cut_N,   KLD_sq,     KLD_quad,
            Disco,       batchcomp,  bcompLog],\
           ['MSE        ', 'MSE_PS     ', 'MSE_PM     ', 
            'MSE_PL     ', 'KLD        ', 'MMD        ', 
            'ENR_PRED   ', 'ENR_RECON  ', 'ENR_LATEN  ',
            'VAR        ', 'MAE        ', 'MLE        ',              
            'ENR_RECON_N', 'E_LogSum   ', 'E_SumLog   ',              
            'MMD_E      ', 'MMD_Hit    ', 'MMD_Latent ',
            'MMD_HitKS  ', 'SortMSE    ', 'SortMAE    ',
            'MMD_Hit2   ', 'MMD_HitKS2 ', 'ENR_Cut    ',
            'ENR_Cut_N  ', 'KLD_sq     ', 'KLD_quad   ',
            'Disco      ', 'batchcomp  ', 'bcompLog   ']


       

def mmd_latent_loss(recon_z, z, alpha=1.0): 
    
    B = z.size(0)
    x = z
    y = recon_z
    
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*xy))

    return  (torch.sum(K)+torch.sum(L) - 2*torch.sum(P))
    
def mmd_hit_loss(recon_x, x, alpha=0.01): 
    
    B = x.size(0)
    x = x.view(B, -1)
    y = recon_x.view(B, -1)
    
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*xy))

    return  (torch.sum(K)+torch.sum(L) - 2*torch.sum(P))

def mmd_hit_loss2(recon_x, x, alpha=0.01): 
    
    B = x.size(0)
    
    x_batch = x.view(B, -1)
    y_batch = recon_x.view(B, -1)
    out = 0
        
    for i in np.arange(0,B):

        x, _ = torch.sort(x_batch[i:i+1].t(), dim=0, descending=True)   
        y, _ = torch.sort(y_batch[i:i+1].t(), dim=0, descending=True)   
        x = x[0:1800]
        y = y[0:1800]

        xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
        L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
        P = torch.exp(- alpha * (rx.t() + ry - 2*xy))
        loss = (torch.sum(K)+torch.sum(L) - 2*torch.sum(P))
        
        out += loss
    return out



def mmd_hit_loss_cast_mean(recon_x, x, alpha=0.01): 
    
    B = x.size(0)
    
    x_batch = x.view(B, -1)
    y_batch = recon_x.view(B, -1)

    x = x_batch.view(B,1,-1)
    y = y_batch.view(B,1,-1)

    xx = torch.matmul(torch.transpose(x,1,2),x) 
    yy = torch.matmul(torch.transpose(y,1,2),y)
    xy = torch.matmul(torch.transpose(y,1,2),x)
    
    rx = (torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx))
    ry = (torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy))
    
    K = torch.exp(- alpha * (torch.transpose(rx,1,2) + rx - 2*xx))
    L = torch.exp(- alpha * (torch.transpose(ry,1,2) + ry - 2*yy))
    P = torch.exp(- alpha * (torch.transpose(ry,1,2) + rx - 2*xy))

    out = (torch.mean(K, (1,2))+torch.mean(L, (1,2)) - 2*torch.mean(P, (1,2)))
    
    return out



def mmd_hit_sortKernel(recon_x_sorted, x_sorted, kernel_size, stride, cutoff, alpha = 200):
    
    B = x_sorted.size(0)
    pixels = x_sorted.size(1)
    out = 0
    norm_out = 0
    
    for j in np.arange(0, min(cutoff, pixels), step = stride):
        distx = x_sorted[:, j:j+kernel_size]
        disty = recon_x_sorted[:, j:j+kernel_size]

        if j == 0:
            out = mmd_hit_loss_cast_mean(disty, distx, alpha=alpha)
        else:
            out += mmd_hit_loss_cast_mean(disty, distx, alpha=alpha)
        
        norm_out += 1
    return (torch.mean(out)/norm_out)
  

### FILE MANAGEMENT ###
 
# create output folder if it does not exists yet
def create_output_folder(outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print("ouput directory created in ", outpath)
