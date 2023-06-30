import os
os.system("cls")

print("importing stuff... ", end="")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from time import sleep
from pathlib import Path
import matplotlib.pyplot as plt
import random
import keyboard
import pickle
import json
import threading

print("Done!")

torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("running model on GPU")
else:
    print("running model on CPU")

def progress_bar(percentage=0, length=40, text="Progress: "):
    percentage = round(max(0,min(100, percentage)),1)
    filled_blocks = int(percentage/100 * length)
    bar = "█" * filled_blocks + " " * (length - filled_blocks)
    print(text + f"[{bar}] {percentage}%                ", end="\r")


def save_model(path=""):
    torch.save(generator.state_dict(), path+"generator.pth")
    torch.save(discriminator.state_dict(), path+"discriminator.pth")

    print("model saved")

    with open("data/gan_loss_data.json", "w") as file:
        json.dump(loss_data, file)


def load_data(path):
    image = (np.array(Image.open(path).convert("RGBA")).transpose(2,0,1) / 127.5) - 1
    return torch.tensor(image, dtype=torch.float32)

#hyperparameter
new_loss_data = True
load_model = False
criterion = nn.BCELoss()
learning_rate = 0.0002
batch_size = 16
num_epoches = 100
num_iterations = 100
num_train_skins = 1000
num_test_skins = 1
data_path = "D:/Datasets/MC_Skins/Skins"
latent_dim = 128
gen_learning_rate = learning_rate
disc_learning_rate = learning_rate


#load old loss data
loss_data = {"g_loss_values":[], "d_loss_values":[], "g_validation_loss_values":[], "d_validation_loss_values":[]}
if not new_loss_data:
    with open("data/gan_loss_data.json", "r") as file:
        loss_data = json.load(file)

#load data paths     
paths = []
with open("D:/Datasets/MC_Skins/skin_paths.json", "r") as file:
    paths = json.load(file)["classic_skins"]
print(f"paths loaded: {len(paths)}")

#load test data
test_dataset = []
for i in range(num_test_skins):
    test_dataset.append(load_data(paths[i]))
    progress_bar(100*i/num_test_skins, 40, "Test-Data loading progress: ")
progress_bar(100, 40, "Test-Data loading progress: ")
print()
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(4, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngf = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d( latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf, 4, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

#load/generate model
print("create models...")
discriminator = Discriminator()
generator = Generator()
if load_model:
    print("loading model...")
    generator.load_state_dict(torch.load("generator.pth"))
    discriminator.load_state_dict(torch.load("discriminator.pth"))
generator = generator.to(device)
generator.apply(weights_init)
discriminator = discriminator.to(device)
discriminator.apply(weights_init)

d_optimizer = optim.Adam(discriminator.parameters(), lr=gen_learning_rate, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=disc_learning_rate, betas=(0.5, 0.999))

#load training data
train_dataset = []
for i in range(num_train_skins):
    train_dataset.append(load_data(paths[num_test_skins+i]))
    if i % 10 == 0:
        if keyboard.is_pressed("n"):
            break
        progress_bar(100*i/num_train_skins, 40, f"Train-Data loading progress ({i}): ")
progress_bar(100, 40, "Train-Data loading progress: ")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\n\n === Parameters ===\n"
      f" new loss data: {new_loss_data}\n"
      f" load model: {load_model}\n"
      f" learning rate: {learning_rate}\n"
      f" iterations: {num_iterations}\n"
      f" number of epochs: {num_epoches}\n"
      f" batch size: {batch_size}\n"
      f" number of training skins: {num_train_skins}\n"
      f" number of testing skins: {num_test_skins}\n"
      f" loss function: {type(criterion).__name__}\n"
      f" discriminator_optimizer: {type(d_optimizer).__name__}\n"
      f" generator_optimizer: {type(g_optimizer).__name__}\n"
      " ===================\n")

print("\nstart training...\npress X to close application (hold A to save the model)")

plt.ion()
plt.figure()
plt.show(block=False)

fixed_noise = torch.randn(1, latent_dim, 1, 1, device=device)

#number for output images
out_count = 1
for iteration in range(1,num_iterations+1):

    # Train/Test loop
    for epoch in range(num_epoches):

        #train model
        d_running_loss = 0.0
        g_running_loss = 0.0
        generator.train()
        discriminator.train()
        for i, real_images in enumerate(train_dataloader, 0):

            if keyboard.is_pressed("x"):
                print()
                if keyboard.is_pressed("a"):
                    save_model()
                print("training finished")
                plt.ioff()
                plt.show()
                exit()

            if keyboard.is_pressed("t"):
                save_model()
                sleep(3)

            real_images = real_images.to(device)
            b_size = real_images.size(0)
            noise = torch.randn((real_images.size(0), latent_dim, 1, 1), device=device)

            # ---------------------
            # Train the discriminator
            # ---------------------

            discriminator.zero_grad()
            label = torch.full((b_size,), 1., dtype=torch.float32, device=device)
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            fake = generator(noise)
            label.fill_(0.)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            d_optimizer.step()
            d_running_loss += errD.item()

            # fake_images = generator(noise)
            # discriminator.zero_grad()
            # disc_real = discriminator(real_images).reshape(-1)
            # loss_disc_real = criterion(disc_real, torch.ones_like(disc_real, device=device))
            # disc_fake = discriminator(fake_images).reshape(-1)
            # loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake, device=device))
            # loss_disc = (loss_disc_real + loss_disc_fake) # / 2
            # d_running_loss += loss_disc.item()
            # loss_disc.backward(retain_graph=True)
            # d_optimizer.step()
            
            
            # -----------------
            # Train the generator
            # -----------------

            generator.zero_grad()
            label.fill_(1.)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            g_optimizer.step()
            g_running_loss += errG.item()

            # generator.zero_grad()
            # output = discriminator(fake_images).reshape(-1)
            # loss_gen = criterion(output, torch.ones_like(output, device=device))
            # g_running_loss += loss_gen.item()
            # loss_gen.backward()
            # g_optimizer.step()


            if i%10==0:
                print(f"Iteration: {iteration}/{num_iterations}, Epoche: {epoch+1}/{num_epoches}, batch: {i:03d}/{len(train_dataloader)}, g_loss: {round(g_running_loss/(i+1), 8)}, d_loss: {round(d_running_loss/(i+1), 8)}                    ", end="\r")

        loss_data["g_loss_values"].append(g_running_loss/len(train_dataloader))
        loss_data["d_loss_values"].append(d_running_loss/len(train_dataloader))


        #test model
        if epoch % 10 == 0:
            generator.eval()
            discriminator.eval()
            with torch.inference_mode():
                fake_image = generator(fixed_noise).detach()
                fake_image = (fake_image.cpu()[0].numpy().transpose(1,2,0) + 1) * (255/2)
                img = Image.fromarray(fake_image.astype(np.uint8), mode="RGBA")
                img.save(f"data/outputs/{out_count}.png")
                out_count+=1

                

        #plot loss values
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.plot(loss_data["d_loss_values"], c="r")
        plt.plot(loss_data["g_loss_values"], c="b")
        plt.plot([0], c="g")
        plt.legend(["discriminator_loss", "generator_loss"])
        plt.draw()
        plt.pause(0.5)
        plt.show(block=False)

    save_model(f"data/models/saves/{iteration}_")

print("\ntraining finished")
save_model()

plt.ioff()
plt.show()