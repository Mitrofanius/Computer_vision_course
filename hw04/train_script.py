import torch
import hw_4
from hw_4 import *
from utils import *

if __name__ == '__main__':
    train_dataloader, val_dataloader = get_dataloaders()
    device = get_device()
    print(device)
    model = UnetFromPretrained()
    model.load_state_dict(torch.load('saved_weights/', map_location='cpu'))
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epoch = 15
    max_acc = 0
    for i in range(0, epoch):
        print(f"epoch {i + 1}/{epoch}")
        _, loss = train_epoch(model, train_loader=train_dataloader, loss_fn=loss_fn, optimizer=opt, device=device)
        print(f"\rTrain loss: {loss:.04f}\r", end="")
        if i % 4 == 0:
            loss = np.round(loss, 3)
            torch.save(model.state_dict(), f'saved_weights/weights3_epoch_{i}_loss_{loss}.pts')