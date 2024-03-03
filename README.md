# Two_Angle_BIBAE

PyTorch Code for training of a BIBAE model, conditioned on the energy and two angles of a particle incident to the face of a calorimeter. The BIB-AE model is designed to operate on a regular grid of size (x, y, z) = 30×49×30, where the z axis points perpendicular to the face of the calorimeter. At inference time, the model conisists of a normalising flow, which generates the latent space for the BIB-AE and the energy sum of the shower, the Core BIB-AE model, the Post Processing model and a rescaling procedure for the energy sum of each shower.

The code in this repository for the training of the model is structured as follows:

- BIBAE: Folder containing the code used for training the BIB-AE, including both the core and Post Processor training

- Flow: Folder containing the code used for training the Normalising flow

- `conda_env.yaml`: packages required to reproduce environment used for training

