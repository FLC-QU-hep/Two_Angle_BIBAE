from comet_ml import Experiment
from API_keys import api_key

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from data_utils.data_loader import MaxwellBatchLoader_Angular

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.multiprocessing import Process

import ML_models.models as models
import ML_models.functions as functions
import time


try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    isApex = True
except:
    isApex = False
    
import importlib
importlib.reload(models)
importlib.reload(functions)

def weight_reset(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
        m.reset_parameters()

def main(kwargs):
    L_MSE,         L_P_MSE         = 1.0,    0.0
    L_MSE_PS,      L_P_MSE_PS      = 0.0,    0.0
    L_MSE_PM,      L_P_MSE_PM      = 0.0,    0.0
    L_MSE_PL,      L_P_MSE_PL      = 0.0,    0.0
    L_KLD,         L_P_KLD         = 0.0,    0.0
    L_MMD,         L_P_MMD         = 0.0,    0.0
    L_VAR,         L_P_VAR         = 0.0,    0.0
    L_ENR_PRED,    L_P_ENR_PRED    = 0.0,    0.0
    L_ENR_RECON,   L_P_ENR_RECON   = 0.0,    0.0
    L_ENR_LATEN,   L_P_ENR_LATEN   = 0.0,    0.0
    L_MAE,         L_P_MAE         = 0.0,    0.0
    L_MLE,         L_P_MLE         = 0.0,    0.0
    L_ENR_RECON_N, L_P_ENR_RECON_N = 0.0,    0.0
    L_E_LogSum,    L_P_E_LogSum    = 0.0,    0.0
    L_E_SumLog,    L_P_E_SumLog    = 0.0,    0.0
    L_MMD_E,       L_P_MMD_E       = 0.0,    0.0
    L_MMD_Hit,     L_P_MMD_Hit     = 0.0,    0.0
    L_MMD_Latent,  L_P_MMD_Latent  = 1.0,    0.0
    L_MMD_HitKS,   L_P_MMD_HitKS   = 0.0,    0.0
    L_SortMSE,     L_P_SortMSE     = 0.0,    0.0
    L_SortMAE,     L_P_SortMAE     = 0.0,    0.0
    L_MMD_Hit2,    L_P_MMD_Hit2    = 0.0,    0.0
    L_MMD_HitKS2,  L_P_MMD_HitKS2  = 0.0,    0.0
    L_ENR_Cut,     L_P_ENR_Cut     = 0.0,    0.0
    L_ENR_Cut_N,   L_P_ENR_Cut_N   = 0.0,    0.0
    L_KLD_sq,      L_P_KLD_sq      = 0.0,    0.0
    L_KLD_quad,    L_P_KLD_quad    = 0.0,    0.0
    L_Disco,       L_P_Disco       = 1.0,    0.0
    
    
    empt = 0.0
    E_true_trans = lambda x:(x)  #100GeV -> 1.0

    L_Losses =  [L_MSE,         L_MSE_PS,     L_MSE_PM, 
                 L_MSE_PL,      L_KLD,        L_MMD, 
                 L_ENR_PRED,    L_ENR_RECON,  L_ENR_LATEN, 
                 L_VAR,         L_MAE,        L_MLE,
                 L_ENR_RECON_N, L_E_LogSum,   L_E_SumLog,
                 L_MMD_E,       L_MMD_Hit,    L_MMD_Latent,
                 L_MMD_HitKS,   L_SortMSE,    L_SortMAE,
                 L_MMD_Hit2,    L_MMD_HitKS2, L_ENR_Cut,
                 L_ENR_Cut_N,   L_KLD_sq,     L_KLD_quad,
                 L_Disco,       empt,         empt]

    L_P_Losses =  [L_P_MSE,         L_P_MSE_PS,     L_P_MSE_PM, 
                   L_P_MSE_PL,      L_P_KLD,        L_P_MMD, 
                   L_P_ENR_PRED,    L_P_ENR_RECON,  L_P_ENR_LATEN, 
                   L_P_VAR,         L_P_MAE,        L_P_MLE,
                   L_P_ENR_RECON_N, L_P_E_LogSum,   L_P_E_SumLog,
                   L_P_MMD_E,       L_P_MMD_Hit,    L_P_MMD_Latent,
                   L_P_MMD_HitKS,   L_P_SortMSE,    L_P_SortMAE,
                   L_P_MMD_Hit2,    L_P_MMD_HitKS2, L_P_ENR_Cut,
                   L_P_ENR_Cut_N,   L_P_KLD_sq,     L_P_KLD_quad,
                   L_P_Disco,       empt,           empt]

    
    default_params = {
        
        "model" : "VAE_ML",
        "suffix" : "_test",
        "E_cond" : False,
        "loss_List" : L_Losses,
        "lossP_List" : L_P_Losses,
        "E_true_trans" : E_true_trans,

        # IO
        "input_path"  : '',
        "output_path" : './results/',

        # False: Train; True: read weights file 
        "batch_size" : 32,
        "epochs" : 10,
        "no_cuda" : True,
        "seed" : 1,
        "log_interval" : 50,
        "train_size" : 1000,
        "shuffle" : True,
        "num_workers" : 1,
        "sample_interval" : 20,
        "save_interval": 40,       
        "continue_train":False,
        "continue_epoch":0,
        "start_PostProc_after_ep":0,
        "PostProc_pretrain":'MSE',
        "PostProc_train":'MSE',
        "latentSmL":24,
        "latent":100,        
        "lr_VAE":1e-3,
        "lr_Critic":1e-4,           
        "lr_PostProc":1e-4,
        "lr_Critic_E":1e-4,
        "lr_Critic_L":1e-4,
        "gamma_VAE":1.0,
        "gamma_PostProc":1.0,        
        "gamma_Critic":1.0,        
        "gamma_Critic_E":1.0,
        "gamma_Critic_L":1.0,
        "opt_VAE" :'Adam',
        "L_adv" : 1.0,
        "L_adv_E" : 1.0,
        "L_adv_L" : 1.0,
        'L_D_P' : 1.0,
        'L_Ang_Const': 0.1,
        "HitMMDKS_Ker" : 50,
        "HitMMDKS_Str" : 25,
        "HitMMDKS_Cut" : 2000,       
        "HitMMD_alpha" : 500.0,
        "HitMMD2_alpha" : 5.0,
        "ENR_Cut_Cutoff" : 0.1,
        "multi_gpu":False,
        'BIBAE_Train' : True,
        'SafeAfterEpochOnly' : True,
        "save_iter_interval": 40,
        'L_adv_reset' : 1.0, 
        'L_adv_L_reset' : 1.0,
        'DiscoPower' : 1.0,
        'DataLoaderType' : 'FastMaxwell_Angular'
    }

    params = {}
    for param in default_params.keys():

        if param in kwargs.keys():
            params[param] = kwargs[param]
        else:
            params[param] = default_params[param]


    cuda = not params["no_cuda"] and torch.cuda.is_available() 
    
    functions.create_output_folder(params["output_path"])

    
    torch.manual_seed(params["seed"])
    
    if cuda:
        if not params["multi_gpu"]:
            device = torch.device("cuda:0")
        if params["multi_gpu"]:
            device = torch.device("cuda:0")

    else:
         device = torch.device("cpu")        

    print(device)
    print(torch.cuda.current_device()) 
    loader_params = {'shuffle': params['shuffle'], 'num_workers': params['num_workers']}

    
    def threshold(x, threshmin, newval=0.0):
        x[x < threshmin] = newval
        return x

    
    tf_lin_F = lambda x:(x)
    
            
   # Cut
    def tf_Photons_thresh(x):
        x[x < 0.0001] = 0.0
        return x
    
   # Training cut and calibration
    def tf_Photons_thresh_calib(x):
        x[x < 0.0001] = 0.0 
        
        # Calibration factor in ILD config is 2.03
        x_calib = x.copy()
        
        for i in range(30):
            if i>=20:
                x_calib[:,i,:,:] = 2.03*x[:,i,:,:]    
                
        return x_calib
        
    print('start model create')

#############################################################################
# ECAL Photon energy and angle conditioning
#############################################################################    
    if params["model"] == "3D_M_BiBAEBatchS_Energy_2Angle_P_None_C_BatchSV2Core_CL_Default_30x30x49Thresh_Reset_DATA_CALIB":
        netD = models.Discriminator_F_BatchStat_Energy_Two_Angle_30x30x49().to(device)
        netD_Reset = models.Discriminator_F_BatchStat_Energy_Two_Angle_30x30x49().to(device)
        netD_L = models.Latent_Critic().to(device)
        model = models.BiBAE_F_3D_BatchStat_Energy_Two_Angle_30x30x49(params, device=device, 
                                                         z_rand=(params["latent"]-params["latentSmL"]),
                                                         z_enc=params["latentSmL"]).to(device)   
        model_P = models.PostProcess_Size1Conv_Energy_Two_Angle_30x30x49_withA_cond_6conv_no_norm_ESum(bias=True, out_funct='none').to(device)
        train = functions.train_BiBAE_F_linear_Eand2A_Reset
        #test = functions.test_BiBAE_F_linear
        LATENT_DIM = params["latent"]+1
        tf = tf_Photons_thresh_calib
        
        
    
        
################### ESum only in PP model ##########################################################################
    elif params["model"] == "3D_M_BiBAEBatchS_Energy_2Angle_P_None_C_BatchSV2Core_CL_Default_30x30x49Thresh_Reset_No_LNorm_PP_ESum_CALIB":
        netD = models.Discriminator_F_BatchStat_Energy_Two_Angle_30x30x49().to(device)
        #netD_Reset = models.Discriminator_F_BatchStat_Energy_Angle_30x30x60().to(device)
        netD_L = models.Latent_Critic().to(device)
        model = models.BiBAE_F_3D_BatchStat_Energy_Two_Angle_30x30x49(params, device=device, 
                                                         z_rand=(params["latent"]-params["latentSmL"]),
                                                         z_enc=params["latentSmL"]).to(device)   
        model_P = models.PostProcess_Size1Conv_Energy_Two_Angle_30x30x49_withA_cond_6conv_no_norm_ESum(bias=True, out_funct='none').to(device)
        train = functions.train_BiBAE_F_linear_Eand2A_Reset_withA_cond_PP_ESum
        #test = functions.test_BiBAE_F_linear
        LATENT_DIM = params["latent"]+1
        tf = tf_Photons_thresh_calib

    print('done model create')
        
        
    if params["opt_VAE"] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params["lr_VAE"], 
                               betas=(0.5, 0.9))
    elif  params["opt_VAE"] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=params["lr_VAE"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=params["gamma_VAE"])

    
    optimizerD = optim.Adam(netD.parameters(), lr=params["lr_Critic"],
                            betas=(0.5, 0.9))
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=1, gamma=params["gamma_Critic"])

    
    optimizerD_L = optim.Adam(netD_L.parameters(), lr=params["lr_Critic_L"], 
                              betas=(0.5, 0.9))        
    schedulerD_L = optim.lr_scheduler.StepLR(optimizerD_L, step_size=1, gamma=params["gamma_Critic_L"])

    
    optimizerP = optim.Adam(model_P.parameters(), lr=params["lr_PostProc"], 
                               betas=(0.5, 0.9))
    schedulerP = optim.lr_scheduler.StepLR(optimizerP, step_size=1, gamma=params["gamma_PostProc"])
    
    #if not netD_Reset == None:
    #    optimizerD_Reset = optim.Adam(netD_Reset.parameters(), lr=params["lr_Critic"],
    #                            betas=(0.5, 0.9))
    #    schedulerD_Reset = optim.lr_scheduler.StepLR(optimizerD_Reset, step_size=1, gamma=params["gamma_Critic"])


    
    
    if params["multi_gpu"]:
        print(torch.cuda.device_count(), " GPUs")
        model = nn.DataParallel(model)
        model_P = nn.DataParallel(model_P)
        netD = nn.DataParallel(netD)
        netD_L = nn.DataParallel(netD_L)
        try:
            netD_Reset = nn.DataParallel(netD_Reset)
        except:
            print('no reset net')

    
    if params["continue_train"]:
        checkpoint = torch.load(params["output_path"] + "check_" + params["model"] + params["suffix"] + '_' +
                                str(params["continue_epoch"]) + '.pth', map_location=torch.device(device))

        model.load_state_dict(checkpoint['model_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        netD_L.load_state_dict(checkpoint['netD_L_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerD_L.load_state_dict(checkpoint['optimizerD_L_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])
        schedulerD_L.load_state_dict(checkpoint['schedulerD_L_state_dict'])

        try:
            model_P.load_state_dict(checkpoint['model_P_state_dict'])
            optimizerP.load_state_dict(checkpoint['optimizerP_state_dict'])
            schedulerP.load_state_dict(checkpoint['schedulerP_state_dict'])
        except:
            for i in range(0,params["continue_epoch"]):
                schedulerP.step()
                
        #if not netD_Reset == None:
        #    netD_Reset.load_state_dict(checkpoint['netD_Reset_state_dict'])
        #    optimizerD_Reset.load_state_dict(checkpoint['optimizerD_Reset_state_dict'])
        #    schedulerD_Reset.load_state_dict(checkpoint['schedulerD_Reset_state_dict'])

        
    model = model.to(device)     
    model_P = model_P.to(device)     
    netD = netD.to(device)     
    netD_L = netD_L.to(device)     
            

    print('BIBAE: ', sum(p.numel() for p in model.parameters() if p.requires_grad))    
    print('PostProc : ', sum(p.numel() for p in model_P.parameters() if p.requires_grad))
    print('Critic_L : ', sum(p.numel() for p in netD_L.parameters() if p.requires_grad))
    print('Critic : ', sum(p.numel() for p in netD.parameters() if p.requires_grad))



    if params['DataLoaderType'] == 'FastMaxwell_Angular':
        train_loader = MaxwellBatchLoader_Angular(file_path=params["input_path"], train_size=params["train_size"], 
                                          batch_size=params["batch_size"], shuffle=params["shuffle"], 
                                          transform=tf)

    print(params)

#############################################################################
# ECAL Photon energy and two angle conditioning
#############################################################################            
    if params["model"] == "3D_M_BiBAEBatchS_Energy_2Angle_P_None_C_BatchSV2Core_CL_Default_30x30x49Thresh_Reset_DATA_CALIB":
                 
        print(device)
        print(model)     
        print(netD)
        print(netD_Reset)
        print(netD_L)
        print(params)
        
        # log network structure
        functions.experiment.set_model_graph(str(model), overwrite=False)
        functions.experiment.set_model_graph(str(netD), overwrite=False)
        functions.experiment.set_model_graph(str(netD_Reset), overwrite=False)
        functions.experiment.set_model_graph(str(netD_L), overwrite=False)
                
        for epoch in range(1+params["continue_epoch"], params["epochs"] + 1 + params["continue_epoch"]):
            print(optimizer)
            print(optimizerP)
            print(optimizerD)
            print(optimizerD_L)
            
            weight_reset(netD_Reset)
            optimizerD_Reset = optim.Adam(netD_Reset.parameters(), lr=params["lr_Critic"]*(params["gamma_Critic"]**(epoch-1)), 
                                          betas=(0.5, 0.9))
            print(optimizerD_Reset)
         

            time_start_ep = time.time()
            train(model=model, netD=netD, netD_L=netD_L, epoch=epoch, 
                  train_loader=train_loader, device=device, args=params, netD_Reset=netD_Reset,
                  optimizer=optimizer, optimizerD=optimizerD, optimizerD_L=optimizerD_L, optimizerD_Reset=optimizerD_Reset,
                  L_Losses=params["loss_List"],  
                  L_D=params['L_adv'], L_D_L=params['L_adv_L'],
                  scheduler=scheduler, schedulerD=schedulerD, schedulerD_L=schedulerD_L)
            #test(model=model, netD=netD, netD_L=netD_L, epoch=epoch, 
            #     test_loader=dataloader, device=device, args=params, 
            #     optimizer=optimizer, L_Losses=params["loss_List"],
            #     L_D=params['L_adv'], L_D_L=params['L_adv_L'])             

            scheduler.step()
            schedulerP.step()
            schedulerD.step()
            schedulerD_L.step()

            
            if epoch%params["save_interval"] == 0 or epoch == 1:
                print('Saving to ' + params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_P_state_dict': model_P.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_L_state_dict': netD_L.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizerP_state_dict': optimizerP.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'optimizerD_L_state_dict': optimizerD_L.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'schedulerP_state_dict': schedulerP.state_dict(),
                    'schedulerD_state_dict': schedulerD.state_dict(),
                    'schedulerD_L_state_dict': schedulerD_L.state_dict()
                    }, 
                    params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth'
                )
                
            time_stop_ep = time.time()

            
            
            print("Duration of this epoch in sec: ", time_stop_ep - time_start_ep)

#############################################################################
# ECAL Photon energy and two angle conditioning with Post Processing
#############################################################################              
            
    elif params["model"] == "3D_M_BiBAEBatchS_Energy_2Angle_P_None_C_BatchSV2Core_CL_Default_30x30x49Thresh_Reset_No_LNorm_PP_ESum_CALIB":
        
        print(device)
        print(model)
        print(model_P)
        print(netD)
        #print(netD_Reset)
        print(netD_L)
        print(params)
        
        # log network structure
        functions.experiment.set_model_graph(str(model), overwrite=False)
        functions.experiment.set_model_graph(str(model_P), overwrite=False)
        functions.experiment.set_model_graph(str(netD), overwrite=False)
        #functions.experiment.set_model_graph(str(netD_Reset), overwrite=False)
        functions.experiment.set_model_graph(str(netD_L), overwrite=False)
                
        for epoch in range(1+params["continue_epoch"], params["epochs"] + 1 + params["continue_epoch"]):
            print(optimizer)
            print(optimizerP)
            print(optimizerD)
            print(optimizerD_L)
            
            #weight_reset(netD_Reset)
            #optimizerD_Reset = optim.Adam(netD_Reset.parameters(), lr=params["lr_Critic"]*(params["gamma_Critic"]**(epoch-1)), 
            #                              betas=(0.5, 0.9))
            #print(optimizerD_Reset)
         

            time_start_ep = time.time()
            train(model=model, modelP=model_P, netD=netD, netD_L=netD_L, epoch=epoch, 
                  train_loader=train_loader, device=device, args=params, #netD_Reset=netD_Reset,
                  optimizer=optimizer, optimizerP=optimizerP, optimizerD=optimizerD, optimizerD_L=optimizerD_L, #optimizerD_Reset=optimizerD_Reset,
                  L_Losses=params["loss_List"], L_Losses_P=params["lossP_List"],
                  L_D=params['L_adv'], L_D_L=params['L_adv_L'], L_D_P=params["L_D_P"],
                  scheduler=scheduler, schedulerD=schedulerD, schedulerD_L=schedulerD_L)
            #test(model=model, netD=netD, netD_L=netD_L, epoch=epoch, 
            #     test_loader=dataloader, device=device, args=params, 
            #     optimizer=optimizer, L_Losses=params["loss_List"],
            #     L_D=params['L_adv'], L_D_L=params['L_adv_L'])             

            scheduler.step()
            schedulerP.step()
            schedulerD.step()
            schedulerD_L.step()

            
            if epoch%params["save_interval"] == 0 or epoch == 1:
                print('Saving to ' + params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_P_state_dict': model_P.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_L_state_dict': netD_L.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizerP_state_dict': optimizerP.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'optimizerD_L_state_dict': optimizerD_L.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'schedulerP_state_dict': schedulerP.state_dict(),
                    'schedulerD_state_dict': schedulerD.state_dict(),
                    'schedulerD_L_state_dict': schedulerD_L.state_dict()
                    }, 
                    params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth'
                )
                
            time_stop_ep = time.time()

            
            
            print("Duration of this epoch in sec: ", time_stop_ep - time_start_ep)