import torch
import torch.nn as nn
import random
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T
from PIL import Image
device = 'cuda'
#%%

# def get_transform():
#     transforms = []
#     transforms.append(T.PILToTensor())
#     transforms.append(T.ConvertImageDtype(torch.float))
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

class animal_dataset(torch.utils.data.Dataset):
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.ds_folder = [os.path.join(ds_path, file) for file in os.listdir(ds_path)]
        self.class_names = [classes for classes in os.listdir(ds_path)]
        self.class_number = len(self.class_names)
        
    def __getitem__(self, idx):
        if idx % 2 == 1:
            class_name = random.choice(self.class_names)
            files = os.listdir(os.path.join(self.ds_path, class_name))
            files = [os.path.join(self.ds_path, class_name, file) for file in files]
            img1 = random.choice(files)
            img2 = random.choice(files)
            label = 1.0
        else:
            class_name1 = random.choice(self.class_names)
            class_name2 = random.choice(self.class_names)
            while class_name1 == class_name2:
                class_name2 = random.choice(self.class_names)
            files1 = os.listdir(os.path.join(self.ds_path, class_name1))
            files1 = [os.path.join(self.ds_path, class_name1, file) for file in files1]
            img1 = random.choice(files1)
            
            files2 = os.listdir(os.path.join(self.ds_path, class_name2))
            files2 = [os.path.join(self.ds_path, class_name2, file) for file in files2]
            img2 = random.choice(files2)
            
            label = 0.0
        
        image_tensor_transform = T.ToTensor()
        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")
        
        img1 = image_tensor_transform(img1)
        img2 = image_tensor_transform(img2)
        
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))
        
    def __len__(self):
        return 5175

# ds = animal_dataset('c:/users/armin/desktop/ds')
ds = animal_dataset('c:/arminpc/ds')
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
#%%
class siamese_network(nn.Module):
    def __init__(self):
        super(siamese_network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 2), nn.ReLU(), 
            )
        self.liner = nn.Sequential(nn.Linear(21632, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        # return self.out(out1)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

net = siamese_network().to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001 )
optimizer.zero_grad()
net.train()
#%%
import time
for i in range(0,10):
    epoch_loss = 0
    for i, (im1, im2, label)  in enumerate(dl):
        optimizer.zero_grad()
        output = net.forward(im1.to(device), im2.to(device))
        loss = loss_fn(output, label.to(device))
        epoch_loss = epoch_loss + loss
        loss.backward()
        optimizer.step()
    print(epoch_loss)
#%%
pytorch_total_params = sum(p.numel() for p in net.parameters())

def image_to_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    image_tensor_transform = T.ToTensor()
    return image_tensor_transform(img)

imgss1 = image_to_tensor('c:/arminpc/ddd/1.png')
imgss1 = imgss1.unsqueeze(dim=0)
imgss2 = image_to_tensor('c:/arminpc/ddd/2.png')
imgss2 = imgss2.unsqueeze(dim=0)

imgss5 = image_to_tensor('c:/arminpc/ddd/5.png')
imgss5 = imgss5.unsqueeze(dim=0)

imgss6 = image_to_tensor('c:/arminpc/ddd/6.png')
imgss6 = imgss6.unsqueeze(dim=0)

with torch.no_grad():
    out = net.forward(imgss3.to(device), imgss5.to(device))
print(out)