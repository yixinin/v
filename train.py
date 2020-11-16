'''
进行模型的训练
'''
import torch
 
import config
from model import ImdbModel
from dataset import get_dataloader
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from eval import eval
 
model = ImdbModel().to(config.device)
optimizer = Adam(model.parameters(),lr=0.001)
loss_list = []
 
def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader,total=len(train_dataloader))
 
    for idx,(input,target) in enumerate(bar):
        optimizer.zero_grad()
        input = input.to(config.device)
        target = target.to(config.device)
        output = model(input)
        loss = F.nll_loss(output,target)
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()
        bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch,idx,np.mean(loss_list)))
 
        if idx%10 == 0:
            torch.save(model.state_dict(),"./models/model.pkl")
            torch.save(optimizer.state_dict(),"./models/optimizer.pkl")
 
if __name__ == '__main__':
    for i in range(5):
        train(i)
        eval()
    plt.figure(figsize=(20,8))
    plt.plot(range(len(loss_list)),loss_list)