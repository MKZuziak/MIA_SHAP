import torch.nn as nn


class Attack_model(nn.Module):
    def __init__(self,input_size):
        super(Attack_model,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.fc(x)


class ShadowModelEmnist(nn.Module):
    def __init__(self):
        super(ShadowModelEmnist,self).__init__()

        self.conv1 = nn.Conv2d(1,32,kernel_size =3 , stride = 1 , padding =1)
        self.conv2 =  nn.Conv2d(32,64,kernel_size =3 , stride = 1 , padding =1)
        self.conv3 =  nn.Conv2d(64,128,kernel_size =3 , stride = 1 , padding =1)

        self.fc1 = nn.Linear(128*3*3,512)
        self.fc2 = nn.Linear(512,10)

        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)


    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0),-1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)


        x= self.fc2(x)
        return x