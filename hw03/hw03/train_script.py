import torch
import hw3
from hw3 import *
from utils import *


if __name__ == '__main__':
    train_dataloader, val_dataloader = get_dataloaders()


    device = get_device()
    print(device)
    model = hw3.AlmostResNet(block=hw3.Block, num_classes=50)
    # model.load_state_dict(torch.load('saved_weights/weights3_epoch_10_loss_1.0.pts', map_location='cpu'))
    model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    test_model_top3(model, test_loader=val_dataloader, device=device)


    # epoch = 15
    # max_acc = 0
    # for i in range(11, epoch + 1):
    #     print(f"epoch {i}/{epoch}")
    #     _, loss = train_epoch(model, train_loader=train_dataloader, loss_fn=loss_fn, optimizer=opt, device=device)
    #     print(f"\rTrain loss: {loss:.04f}\r", end="")
    #     acc = test_model_top3(model, test_loader=val_dataloader, device=device)
    #     if acc > max_acc:
    #         loss = np.round(loss)
    #         torch.save(model.state_dict(), f'saved_weights/weights3_epoch_{i}_loss_{loss}.pts')
