import BIBAE_Angle_Energy_PP_Main
import ML_models.functions as functions

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
L_SortMSE,     L_P_SortMSE     = 0.0,    0.0
L_SortMAE,     L_P_SortMAE     = 0.0,    0.0
L_MMD_Hit2,    L_P_MMD_Hit2    = 0.0,    0.0
L_MMD_HitKS2,  L_P_MMD_HitKS2  = 0.0,    0.0
L_ENR_Cut,     L_P_ENR_Cut,    = 0.0,    0.0
L_ENR_Cut_N,   L_P_ENR_Cut_N,  = 0.0,    0.0
empt = 0.0

E_true_trans = lambda x:(x)

L_Losses =  [L_MSE,         L_MSE_PS,     L_MSE_PM,
             L_MSE_PL,      L_KLD,        L_MMD,
             L_ENR_PRED,    L_ENR_RECON,  L_ENR_LATEN,
             L_VAR,         L_MAE,        L_MLE,
             L_ENR_RECON_N, L_E_LogSum,   L_E_SumLog,
             L_MMD_E,       L_MMD_Hit,    L_MMD_Latent,
             L_MMD_HitKS,   L_SortMSE,    L_SortMAE,
             L_MMD_Hit2,    L_MMD_HitKS2, L_ENR_Cut,
             L_ENR_Cut_N,   empt,         empt,
             empt,          empt,         empt]

L_P_Losses =  [L_P_MSE,         L_P_MSE_PS,     L_P_MSE_PM,
               L_P_MSE_PL,      L_P_KLD,        L_P_MMD,
               L_P_ENR_PRED,    L_P_ENR_RECON,  L_P_ENR_LATEN,
               L_P_VAR,         L_P_MAE,        L_P_MLE,
               L_P_ENR_RECON_N, L_P_E_LogSum,   L_P_E_SumLog,
               L_P_MMD_E,       L_P_MMD_Hit,    L_P_MMD_Latent,
               L_P_MMD_HitKS,   L_P_SortMSE,    L_P_SortMAE,
               L_P_MMD_Hit2,    L_P_MMD_HitKS2, L_P_ENR_Cut,
               L_P_ENR_Cut_N,   empt,           empt,
               empt,            empt,           empt]

params = {'model':           
          '3D_M_BiBAEBatchS_Energy_2Angle_P_None_C_BatchSV2Core_CL_Default_30x30x49Thresh_Reset_DATA_CALIB', 
          
          
          'suffix': 
          'BIBAE_two_angle_499999_run_1', 
          
          'E_cond': True,
          'loss_List': L_Losses,
          "lossP_List" : L_P_Losses,
          'E_true_trans': E_true_trans,

          #### Input data
           'input_path' : 'YOUR_INPUT_PATH',
          
          ######### Output path
          'output_path' : 'YOUR_OUTPUT_PATH',
          
          
          'batch_size': 64*4, # batchsize*num_gpus
          'epochs': 100,
          'no_cuda': False,
          'seed': 1,
          'log_interval': 100,
          'train_size': 500000,
          'shuffle': True,
          'num_workers': 1,
          'sample_interval': 1,
          'save_interval': 1,
          "continue_train": True, #False, 
          "continue_epoch": 59, #43, #27, #11, #0,
          "start_PostProc_after_ep":10000,
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
          "HitMMDKS_Ker" : 150, #100,
          "HitMMDKS_Str" : 25,
          "HitMMDKS_Cut" : 1000, #2000,
          'opt_VAE' :'Adam',
          'L_D_P' : 0.0,
          'L_adv' : 1.0,
          'L_adv_E' : 0.0,
          'L_adv_L' : 100.0,
          'L_Ang_Const': 0.1,
          'HitMMD_alpha' : 40.0, #200.0,
          "HitMMD2_alpha" : 4.0, #20.0,
          'multi_gpu':True,
          'BIBAE_Train' : True,
          'SafeAfterEpochOnly' : True,
          "save_iter_interval": 0,
          'L_adv_reset' : 1.0,
          'L_adv_L_reset' : 100.0,
          'DataLoaderType' : 'FastMaxwell_Angular'}

BIBAE_Angle_Energy_PP_Main.main(params)

# log parameters of training to comet ML
for param in params.keys():
    functions.experiment.log_parameter(param, params[param])