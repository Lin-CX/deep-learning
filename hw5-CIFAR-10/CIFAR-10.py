import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################
        self.n1 = nblk_stage1
        self.n2 = nblk_stage2
        self.n3 = nblk_stage3
        self.n4 = nblk_stage4

        self.in_channels = 3
        self.out_channels = 64

        # convolutional layer
        self.conv_convlayer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 1st stage
        self.block_first_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

        # 2nd stage
        self.preact_block1st_second_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_conn_second_stage = nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=1, stride=2)
        self.block1st_second_stage = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=3, stride=1, padding=1)
        )
        self.restblock_second_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=3, stride=1, padding=1)
        )

        # 3rd stage
        self.preact_block1st_third_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels*2),
            nn.ReLU(inplace=True)
        )
        self.skip_conn_third_stage = nn.Conv2d(self.out_channels*2, self.out_channels*4, kernel_size=1, stride=2)
        self.block1st_third_stage = nn.Sequential(
            nn.Conv2d(self.out_channels*2, self.out_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, kernel_size=3, stride=1, padding=1)
        )
        self.restblock_third_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*4, self.out_channels*4, kernel_size=3, stride=1, padding=1)
        )

        # 4th stage
        self.preact_block1st_fourth_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels*4),
            nn.ReLU(inplace=True)
        )
        self.skip_conn_fourth_stage = nn.Conv2d(self.out_channels*4, self.out_channels*8, kernel_size=1, stride=2)
        self.block1st_fourth_stage = nn.Sequential(
            nn.Conv2d(self.out_channels*4, self.out_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, kernel_size=3, stride=1, padding=1)
        )
        self.restblock_fourth_stage = nn.Sequential(
            nn.BatchNorm2d(self.out_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*8, self.out_channels*8, kernel_size=3, stride=1, padding=1)
        )

        # fully connected layer
        self.fully_conn = nn.Linear(self.out_channels*8, 10)

    ########################################
    # You can define whatever methods
    ########################################
    
    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        # convolutional layer
        out = self.conv_convlayer(x)                                # conv DONE

        # 1st stage
        for i in range(self.n1):
            residual = out
            out = self.block_first_stage(out) + residual            # Stage 1 DONE

        # 2nd stage
        # 1st block of 2nd stage
        out = self.preact_block1st_second_stage(out)
        residual = self.skip_conn_second_stage(out)
        out = self.block1st_second_stage(out)
        out += residual
        # n2-1 blocks of 2nd stage
        for i in range(self.n2-1):
            residual = out
            out = self.restblock_second_stage(out) + residual      # Stage 2 DONE

        # 3rd stage
        # 1st block of 3rd stage
        out = self.preact_block1st_third_stage(out)
        residual = self.skip_conn_third_stage(out)
        out = self.block1st_third_stage(out)
        out += residual
        # n3-1 blocks of 3rd stage
        for i in range(self.n3-1):
            residual = out
            out = self.restblock_third_stage(out) + residual        # Stage 3 DONE


        # 4th stage
        out = self.preact_block1st_fourth_stage(out)
        residual = self.skip_conn_fourth_stage(out)
        out = self.block1st_fourth_stage(out)
        out += residual
        # n4-1 blocks of 4th stage
        for i in range(self.n4-1):
            residual = out
            out = self.restblock_fourth_stage(out) + residual

        # average pooling layer
        out = F.avg_pool2d(out, kernel_size=(4, 4), stride=4)

        #print()
        #print(out.size())
        #print()

        # fully connected layer
        out = out.view(out.size(0), -1)
        out = self.fully_conn(out)
        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)


########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 4

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
model = net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
lr = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()
        
        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = model(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')