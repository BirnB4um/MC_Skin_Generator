import torch
import torch.nn as nn
import torch.optim as optim

print("finished importing")

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(3552, 1000),
            nn.ReLU(),
            nn.Linear(1000, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh()
        )

        self.decode = nn.Sequential(
            nn.Linear(100, 300),
            nn.Tanh(),
            nn.Linear(300, 600),
            nn.Tanh(),
            nn.Linear(600,1000),
            nn.ReLU(),
            nn.Linear(1000, 3552),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decode(self.encode(x))

load_model = False
if load_model:
    net = torch.load("model.pth").to(device)
else:
    net = Net().to(device)

#parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
num_epoches = 0
batch_size = 32

# Training loop
for epoch in range(num_epoches):
    pass


print("training finished")

#save model
torch.save(net, "model.pth")