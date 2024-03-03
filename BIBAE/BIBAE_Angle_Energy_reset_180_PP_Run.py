import BIBAE_Angle_Energy_PP_Main 
import time
import os

def main():
    L_MSE,         L_P_MSE         = 0.0,    0.0
    L_MSE_PS,      L_P_MSE_PS      = 0.0,    0.0
    L_MSE_PM,      L_P_MSE_PM      = 0.0,    0.0
    L_MSE_PL,      L_P_MSE_PL      = 0.0,    0.0
    L_KLD,         L_P_KLD         = 0.1,    0.0
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
    L_MMD_Latent,  L_P_MMD_Latent  = 100.0,  0.0
    L_MMD_HitKS,   L_P_MMD_HitKS   = 0.0,    5.0
    L_SortMSE,     L_P_SortMSE     = 0.0,    10.0
    L_SortMAE,     L_P_SortMAE     = 0.0,    10.0
    L_MMD_Hit2,    L_P_MMD_Hit2    = 0.0,    0.0
    L_MMD_HitKS2,  L_P_MMD_HitKS2  = 0.0,    5.0
    L_ENR_Cut,     L_P_ENR_Cut,    = 0.0,    0.0
    L_ENR_Cut_N,   L_P_ENR_Cut_N,  = 0.0,    0.0
    L_KLD_sq,      L_P_KLD_sq      = 0.0,    0.0
    L_KLD_quad,    L_P_KLD_quad    = 0.0,    0.0
    L_Disco,       L_P_Disco       = 0.0,    0.0
    L_batchcomp,   L_P_batchcomp   = 0.0,    0.0001
    L_bcompLog,    L_P_bcompLog    = 0.0,    0.0


    empt = 0.0

    L_Losses =  [L_MSE,         L_MSE_PS,     L_MSE_PM,
                 L_MSE_PL,      L_KLD,        L_MMD,
                 L_ENR_PRED,    L_ENR_RECON,  L_ENR_LATEN,
                 L_VAR,         L_MAE,        L_MLE,
                 L_ENR_RECON_N, L_E_LogSum,   L_E_SumLog,
                 L_MMD_E,       L_MMD_Hit,    L_MMD_Latent,
                 L_MMD_HitKS,   L_SortMSE,    L_SortMAE,
                 L_MMD_Hit2,    L_MMD_HitKS2, L_ENR_Cut,
                 L_ENR_Cut_N,   L_KLD_sq,     L_KLD_quad,
                 L_Disco,       L_batchcomp,  L_bcompLog]


    L_P_Losses =  [L_P_MSE,         L_P_MSE_PS,     L_P_MSE_PM,
                   L_P_MSE_PL,      L_P_KLD,        L_P_MMD,
                   L_P_ENR_PRED,    L_P_ENR_RECON,  L_P_ENR_LATEN,
                   L_P_VAR,         L_P_MAE,        L_P_MLE,
                   L_P_ENR_RECON_N, L_P_E_LogSum,   L_P_E_SumLog,
                   L_P_MMD_E,       L_P_MMD_Hit,    L_P_MMD_Latent,
                   L_P_MMD_HitKS,   L_P_SortMSE,    L_P_SortMAE,
                   L_P_MMD_Hit2,    L_P_MMD_HitKS2, L_P_ENR_Cut,
                   L_P_ENR_Cut_N,   L_P_KLD_sq,     L_P_KLD_quad,
                   L_P_Disco,       L_P_batchcomp,  L_P_bcompLog]



    params = {'model':               
              ######## CALIB: BIBAE PP with no norm layer, ESum conditioning only in PP model
              '3D_M_BiBAEBatchS_Energy_2Angle_P_None_C_BatchSV2Core_CL_Default_30x30x49Thresh_Reset_No_LNorm_PP_ESum_CALIB',
              
              
              'suffix': 'Test_PP', 
              
              'E_cond': True,
              'loss_List': L_Losses,
              "lossP_List" : L_P_Losses,
              
              
              #### Input dataset
              'input_path' : 'YOUR_PATH',

              
              ######## Output
              'output_path' : 'YOUR OUTPUT PATH',
              
              

              'batch_size': 32*4, # batchsize*num_gpus
              'epochs': 150, #100,
              'no_cuda': False,
              'seed': 1,
              'log_interval': 100, 
              'train_size': 500000, 
              'shuffle': True,
              'num_workers': 1,
              'sample_interval': 1,
              'save_interval': 1,
              "continue_train": True,
              ###### NOTE: start_PostProc_after_ep must be ONE MORE than continue_epoch
              "continue_epoch": 51, #50, 
              "start_PostProc_after_ep": 52, #51, 
              "PostProc_pretrain":'MSE',
              "PostProc_train":'MSE',
              "latentSmL":24,
              "latent":512,
              'lr_VAE':1e-4*0.5,
              'lr_Critic':1e-4*0.5,
              "lr_PostProc":1e-4*0.5,
              'lr_Critic_E':0,
              'lr_Critic_L':2.0*1e-4,
              'gamma_VAE':0.97,
              'gamma_Critic':0.97,
              "gamma_PostProc":0.97,
              'gamma_Critic_E':0.97,
              'gamma_Critic_L':0.97,
              "HitMMDKS_Ker" : 150,
              "HitMMDKS_Str" : 25,
              "HitMMDKS_Cut" : 1000,
              'opt_VAE' :'Adam',
              'L_D_P' : 0.0,
              'L_adv' : 1.0,
              'L_adv_E' : 0.0,
              'L_adv_L' : 100.0,
              'HitMMD_alpha' : 40.0,
              "HitMMD2_alpha" : 4.0,
              'multi_gpu':True,
              'BIBAE_Train' : False,
              'SafeAfterEpochOnly' : True,
              "save_iter_interval": 50,
              'L_adv_reset' : 1.0,
              'L_adv_L_reset' : 100.0,
              'DataLoaderType' : 'FastMaxwell_Angular'
    }

    BIBAE_Angle_Energy_PP_Main.main(params)

if __name__ == '__main__':
    main()