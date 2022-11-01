import imp
from torchvision import datasets
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.optim as optim

from data import labels as lbl
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

BATCH_SIZE = 8
NUM_WORKERS = 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

T = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

voc_train_data = datasets.VOCDetection(root='./datasets/', image_set="train", transform=T, year='2012')
voc_test_data  = datasets.VOCDetection(root='./datasets/', image_set="val", transform=T, year='2012')

def coll_fn(x):
    inputs, labels = zip(*x)

    inputs = torch.stack(inputs, dim=0)

    labels = [l['annotation']['object'] for l in labels]
    labels = [{
            'boxes':  torch.tensor([list(map(int, b['bndbox'].values())) for b in l]),
            'labels': torch.tensor([lbl[b['name']] for b in l])
        } for l in labels]

    return (inputs, labels)

trainloader = DataLoader(voc_train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=coll_fn)
testloader = DataLoader(voc_test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=coll_fn)

model = fasterrcnn_resnet50_fpn()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    
    running_loss = 0.0
    for i, (images, targets) in enumerate(tqdm(trainloader)):

        images = images.to(device)

        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images, targets)
        # loss = criterion(outputs, targets)
        # loss.backward()
        
        loss = outputs['loss_classifier'] + outputs['loss_box_reg']

        # print(f'        [m_batch: {i}], loss: {loss}')

        loss.backward()
        
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 100 mini-batches

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')

            running_loss = 0.0

print('Finished Training')

a=1