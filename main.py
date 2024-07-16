import torch
import numpy as np
from Utils.DeviceUtil import devices
import time
import math
from tqdm.auto import tqdm, trange
import numpy as np
import cv2
from torch import tensor
from torch import nn
from collections import OrderedDict
from STDP import STDP

#### "model" ####

# make model with layers: 784, 1200, 1200, 10
model = torch.nn.Sequential(
    torch.nn.Linear(15, 20),
    torch.nn.Linear(20, 20),
    torch.nn.Linear(20, 10)
)

# initialize weights
for layer in model:
    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    torch.nn.init.normal_(layer.bias, mean=0.0, std=0.01)

# define dictionary spikeOutputsTimesCache. Key = layer name. Value = list of tensors of spikes. idx of tensor in list + 1 = time step, tensor size = num neurons in layer. 
# initialize all tensors to random 0 or 1
spikeOutputsTimeCache = OrderedDict()

spikeOutputsTimeCache["input"] = [torch.randint(0, 2, (15,)) for _ in range(10)] # 10 time steps
        
for layer in self.model:
    spikeOutputsTimeCache[layer.layerName] = [torch.randint(0, 2, (15,)) for _ in range(10)] # 10 time steps


# Parameters for the STDP rule (Gerstner)
A_plus = 0.01
A_minus = 0.012
tau_plus = 20.0
tau_minus = 20.0

# Create an instance of the STDP rule
stdp = STDP(model, A_plus, A_minus, tau_plus, tau_minus, spikeOutputsTimeCache)

# update the network
stdp.update_network()
