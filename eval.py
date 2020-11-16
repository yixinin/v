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
 
 
 
def eval():
    model = ImdbModel().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))
    model.eval()
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for input,target in test_dataloader:
            input = input.to(config.device)
            target = target.to(config.device)
            output = model(input)
            loss = F.nll_loss(output,target)
            loss_list.append(loss.item())
            #准确率
            pred= output.max(dim = -1)[-1]
            acc_list.append(pred.eq(target).cpu().float().mean())
        print("loss:{:.6f},acc:{}".format(np.mean(loss_list),np.mean(acc_list)))
 
 
if __name__ == '__main__':
    eval()