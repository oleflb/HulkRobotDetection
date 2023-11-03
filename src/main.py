from .engine import train_one_epoch, evaluate
from . import utils
from .models.ssdnet import SSDNet
from .models.fasterrcnn import FasterRCNN
from .dataloader.lightningdataset import DataModule
import torch

BATCH_SIZE=24
model = SSDNet((480, 640), 5)
print(model)
dataloader = DataModule(num_workers=12, batch_size=BATCH_SIZE)
dataloader.setup("fit")
data_loader = dataloader.train_dataloader()
data_loader_val = dataloader.val_dataloader()


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_val, device=device)

print("That's it!")
