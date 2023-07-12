import os
import torch
import numpy as np

def get_accuracy(prediction, labels_batch, dim=1):
    pred_index = prediction.argmax(dim)
    return (pred_index == labels_batch).float().mean()

def top3_accuracy(prediction, labels_batch, dim=1): 
    # best3 = prediction.argsort()
    # torch.topk(a.flatten(), 3).indices
    pass

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available[:-1])  # Skip the 7th card --- it is reserved for evaluation!!!
    return int(index)
 
 
def get_device():  
    if torch.cuda.is_available():
        gpu = get_free_gpu()
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device

def get_device_mps():  
    if torch.has_mps:
        gpu = 'mps'
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device