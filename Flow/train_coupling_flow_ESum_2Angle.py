from comet_ml import Experiment

from functools import partial

import torch

from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN

from pyro.distributions import constraints
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms import SplineCoupling
from pyro.distributions.transforms.spline import ConditionalSpline



import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process

import time
import h5py

from custom_pyro import ConditionalAffineCouplingTanH

from pyro.nn import ConditionalDenseNN, DenseNN, ConditionalAutoRegressiveNN
import pyro.distributions as dist
import pyro.distributions.transforms as T
device = torch.device('cuda:0')


from API_keys import api_key 

# create CometML experiment
experiment = Experiment(
    api_key=api_key,
    project_name="YOUR PROJECT NAME",
    workspace="YOUR WORK SPACE",
)


experiment.add_tag('Coupling flow setup BIBAE latent with ESum, latentSmL=24')

def compile_Hybrid_model(num_blocks, num_inputs, num_cond_inputs, device):
    # the latent space distribution
    base_dist = dist.Normal(torch.zeros(num_inputs).to(device), torch.ones(num_inputs).to(device))

    input_dim = num_inputs
    count_bins = 8
    transforms = []
    transforms2 = []
      
    input_dim = num_inputs
    split_dim = num_inputs//2
    param_dims1 = [input_dim-split_dim, input_dim-split_dim]
    param_dims2 = [input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1), input_dim * count_bins]

    torch.manual_seed(66)

    for i in range(num_blocks):
        

                    
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*5, input_dim*5], param_dims1)
        
        ctf = T.ConditionalAffineCoupling(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
            
        hypernet = DenseNN(num_cond_inputs, [input_dim*5, input_dim*5], param_dims2)

        ctf = T.ConditionalSpline(hypernet, input_dim, count_bins)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)


    modules = nn.ModuleList(transforms2)

    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transforms)

    return modules, flow_dist



def compile_HybridTanH_model(num_blocks, num_inputs, num_cond_inputs, device):
    # the latent space distribution
    base_dist = dist.Normal(torch.zeros(num_inputs).to(device), torch.ones(num_inputs).to(device))

    input_dim = num_inputs
    count_bins = 8
    transforms = []
    transforms2 = []
      
    input_dim = num_inputs
    split_dim = num_inputs//2
    param_dims1 = [input_dim-split_dim, input_dim-split_dim]
    param_dims2 = [input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1), input_dim * count_bins]

    torch.manual_seed(66)

    for i in range(num_blocks):
        

                    
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)

        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        
        
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        hypernet = DenseNN(num_cond_inputs, [input_dim*4, input_dim*4], param_dims2)
        #hypernet.apply(init_weights)
        ctf = T.ConditionalSpline(hypernet, input_dim, count_bins)
        transforms2.append(ctf)
        transforms.append(ctf)
        

        

        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)

        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        
        
        
        
    modules = nn.ModuleList(transforms2)

    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transforms)

    return modules, flow_dist


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, a=0.0, b=0.001)
        m.bias.data.fill_(0.0001)


default_params = {

    "model" : '3D_M_BiBAEBatchS_Energy_Angle_P',
    "suffix" : '_coupling_flow', 
    # IO
    'input_path'  : '', 
    "output_path" : './results/',

    "batch_size" : 2048,
    "epochs" : 200,
    "train_size" : 500000,
    "shuffle" : True,
}


kwargs = {}
params = {}
for param in default_params.keys():

    if param in kwargs.keys():
        params[param] = kwargs[param]
    else:
        params[param] = default_params[param]
        

tf = lambda x:(x)

###### Path to latent file ######
latent_file_path = 'YOUR LATENT FILE PATH'

latent_file_name = 'YOUR_LATENT_FILE.h5'

###### Load latent file #########
latent_file = h5py.File(latent_file_path + latent_file_name,'r')


mu_full = latent_file['latent']['mu'][:]
logvar_full = latent_file ['latent']['logvar'][:]
e_full = latent_file['latent']['energy'][:]
theta_full = latent_file['latent']['theta'][:]
phi_full = latent_file['latent']['phi'][:]
e_sum = latent_file['latent']['energy_sum'][:]


mu_full_tensor = torch.tensor(mu_full)
logvar_full_tensor = torch.tensor(logvar_full)
e_full_tensor = torch.tensor(e_full)
theta_full_tensor = torch.tensor(theta_full)
phi_full_tensor = torch.tensor(phi_full)
e_sum_tensor = torch.tensor(e_sum)

dataset = torch.utils.data.TensorDataset(mu_full_tensor, logvar_full_tensor, 
                                         e_full_tensor, theta_full_tensor,
                                         phi_full_tensor, e_sum_tensor)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], pin_memory=True)



num_blocks =8



model, distribution = compile_HybridTanH_model(num_blocks, 
                                           num_inputs=mu_full_tensor.size(1) + e_sum_tensor.size(1), ### Want to generate ESum with the flow 
                                           num_cond_inputs=3, device=device)


lr = 1e-4
                   
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
                   
####### Make logs to comet ##########
# log graph structures to Comet ML
experiment.set_model_graph(str(model), overwrite=False)

experiment.log_parameter('Optimizer learning rates', lr)                   

torch.manual_seed(1)


for epoch in range(1, params["epochs"]+1):
    input_list = []
    for batch_idx, (mu, logvar, e, theta, phi, e_sum) in enumerate(train_loader):

        
        E_true = e.to(device).float()
        theta_true = theta.to(device).float()
        phi_true = phi.to(device).float()
        energy_sum = e_sum.to(device).float()

        std = np.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        z = z.float().to(device)/10
        
        
        # scale with max energy sum of the training dataset
        energy_sum = (energy_sum/3300).float()

        
        # scale conditional labels
        E_true = (E_true/126).float()
        theta_true = (theta_true/np.radians(95.0)).float()
        phi_true = (phi_true/np.radians(95.0)).float()
        
        input_data = torch.cat((energy_sum, z), 1)   #### input data structure required for network 
                                                     ###  with additional features in latent space (e.g. Esum)
        

        optimizer.zero_grad()

        # add context for conditioning by concatenating 
        context = torch.cat((E_true, theta_true, phi_true), 1)            
            
        nll = -distribution.condition(context).log_prob(input_data)
        loss = nll.mean()
        loss.backward()

        optimizer.step() 

        distribution.clear_cache()
        
        input_list.append(input_data.detach().cpu().numpy())

    print(epoch, loss.item())    
    #############################
    # Log to COMET ML
    #############################

    experiment.log_metric('Loss', loss.item(), epoch = epoch)

    torch.save({
        'model': model.state_dict()
        }, 
        latent_file_path + 'Flow_Latent_Data_ep_100/FLow_test_{}.pth'.format(epoch)
    )