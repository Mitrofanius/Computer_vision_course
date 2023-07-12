from torch import nn
import torchvision as tv
import torchvision.transforms as tfms
from torch.utils.data import DataLoader
import numpy as np
import torch
import os

def get_dataloaders():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = tfms.Compose([tfms.Resize((256, 256)), tfms.ToTensor(), tfms.Normalize(mean, std)])
    val_transform = tfms.Compose([tfms.Resize((256, 256)), tfms.ToTensor(), tfms.Normalize(mean, std)])
    
    train_dataset = tv.datasets.ImageFolder(f'/local/temporary/vir/hw03/train', transform=train_transform)
    val_dataset = tv.datasets.ImageFolder(f'/local/temporary/vir/hw03/val', transform=val_transform)

    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    loss_sum = 0
    for idx, batch in enumerate(train_loader):
        data, labels = batch[0].to(device), batch[1].to(device)

        output = model(data)
        loss = loss_fn(output, labels) / 2
        loss.backward()

        if idx % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        loss_sum += loss.item()

        if idx % 100 == 0:
            print(loss.item())
        
    loss = loss_sum / (idx+1)
    return model, loss


def test_model_top3(model, test_loader, device):
    model.eval()
    acc = 0

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            data, labels = batch[0].to(device), batch[1].to(device)
            output = model(data)
            acc += top3_accuracy(output, labels, dim=1)
    
    acc /= (idx + 1)
    print(f'{acc * 100:.1f}%')
    return acc

def get_accuracy(prediction, labels_batch, dim=1):
    pred_index = prediction.argmax(dim)
    return (pred_index == labels_batch).float().mean()

def top3_accuracy(prediction, labels_batch, dim=1): 
    pred = torch.topk(prediction, 3).indices
    # labels_batch = labels_batch.cpu()
    acc = 0
    for i, x in enumerate(labels_batch):
        acc += (x in pred)
    return acc

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