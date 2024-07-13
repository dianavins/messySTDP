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

class STDP:
    def __init__(self, model, A_plus, A_minus, tau_plus, tau_minus, spikeOutputsTimeCache):
        self.model = model
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.spikeOutputsTimeCache = spikeOutputsTimeCache 
        
        # Map layer names to model indices
        self.layer_map = {
            "input": -1,  # No layer in model, special case
            "layer1": 0,  # Index of first layer in model, weights that end up updating
            "layer2": 1,
            "layer3": 2
        }

    # update weight per layer pair
    def update_weight(self, weight, pre_layer, post_layer):
              
        model_layer_idx = self.layer_map[post_layer]
        print("model_layer_idx: ", model_layer_idx)
        model_layer = self.model[model_layer_idx]
        
        # Get the list spike time tensors of the pre- and post-synaptic layers
        pre_spike_lists = self.spikeOutputsTimeCache[pre_layer]
        #print(len(pre_spike_lists))
        post_spike_lists = self.spikeOutputsTimeCache[post_layer]
        #print(len(post_spike_lists))
        
        num_pre_neurons = pre_spike_lists[0].numel()
        #print(num_pre_neurons)
        num_post_neurons = post_spike_lists[0].numel()
        #print(num_post_neurons)
        
        # create empty delta_w tensor to update layer's weight matrix with
        weight_updates = torch.zeros_like(model_layer.weight.data)        
        
        # loop through every possible pair of pre and post spikes to find each delta_t
        # every pre post spike pair:
        for pre_neuron_idx, post_neuron_idx in zip(range(num_pre_neurons), range(num_post_neurons)):
            
            delta_ws = []
            
            # make list of numTimeStep indices from tensor for pre and post spikes
            # each time_idx of pre_spike_times is a timestep, each tensor's neuron_idx has 1 if neuron spiked and 0 if not.
            # find the list of time_idx of each neuron_idx's spike
            pre_spike_times = [time_idx for time_idx, spikes in enumerate(pre_spike_lists, start=1) if spikes[pre_neuron_idx] == 1]
            post_spike_times = [time_idx for time_idx, spikes in enumerate(post_spike_lists, start=1) if spikes[post_neuron_idx] == 1]
            
            for pre_spike_time, post_spike_time in zip(pre_spike_times, post_spike_times):
            
                delta_t = post_spike_time - pre_spike_time
                
                if abs(delta_t) < 100:

                    if delta_t > 0:
                        # Potentiation
                        delta_w = self.A_plus * np.exp(-delta_t / self.tau_plus)
                    else:
                        # Depression
                        delta_w = -self.A_minus * np.exp(delta_t / self.tau_minus)
                
                    delta_ws.append(delta_w)
            
            if delta_ws:
                    avg_delta_w = np.mean(delta_ws)
                    # add this avg_dela_w to corresponding position in delta_w_tensor
                    weight_updates[post_neuron_idx, pre_neuron_idx] = avg_delta_w
            
        # Update the weight 
        model_layer.weight.data += weight_updates 

        # Ensure weight remains within bounds (e.g., [0, 1])
        weight = torch.clamp(model_layer.weight.data, 0, 1)
                
    # update whole network
    def update_network(self):
        layers = list(self.spikeOutputsTimeCache.keys())
        for i in range(len(layers) - 1):
            pre_layer = layers[i]
            post_layer = layers[i + 1]
            self.update_weight(self.model[self.layer_map[post_layer]].weight, pre_layer, post_layer)
            print("layer updated: ", post_layer)
