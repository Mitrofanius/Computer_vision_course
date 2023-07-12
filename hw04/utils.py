import os
import torch
import numpy as np


from torch import nn
import torchvision as tv
import torchvision.transforms as tfms
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import torch.utils.data as tdata

class Dataset(tdata.Dataset):
    def __init__(self, rgbs, labels):
        super().__init__()
        self.rgbs = rgbs
        self.labels = labels

    def __len__(self):
        return self.rgbs.shape[0]

    def __getitem__(self, i):
        return (
            np.asarray(self.rgbs[i]).astype('f4').transpose((2, 0, 1)) / 255,
            np.asarray(self.labels[i]).astype('i8')
            )


def get_dataloaders():
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # train_transform = tfms.Compose([tfms.Resize((256, 256)), tfms.ToTensor(), tfms.Normalize(mean, std)])
    # val_transform = tfms.Compose([tfms.Resize((256, 256)), tfms.ToTensor(), tfms.Normalize(mean, std)])

    # train_dataset = tv.datasets.ImageFolder(f'/local/temporary/vir/hw03/train', transform=train_transform)
    # val_dataset = tv.datasets.ImageFolder(f'/local/temporary/vir/hw03/val', transform=val_transform)

    train_data = torch.from_numpy(np.load(f'/local/temporary/vir/hw04/train/rgbs.npy', allow_pickle=True))
    train_labels = torch.from_numpy(np.load(f'/local/temporary/vir/hw04/train/labels.npy', allow_pickle=True))

    train_dataset = Dataset(rgbs=train_data, labels=train_labels)


    BATCH_SIZE = 4
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    return train_dataloader

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    loss_sum = 0
    for idx, batch in enumerate(train_loader):
        data, labels = batch[0].to(device), batch[1].to(device)

        output = model(data)
        loss = loss_fn(output, labels) / 4
        loss.backward()

        if (idx+1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item()

        if idx % 100 == 0:
            print(f"loss for index {idx} is {loss.item()}")
    loss = loss_sum / (idx+1)
    return model, loss
                                 
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