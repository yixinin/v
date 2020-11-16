"""
构建模型
"""
import torch.nn as nn
import config
import torch.nn.functional as F
 
class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=300,padding_idx=config.ws.PAD)
        self.fc = nn.Linear(config.max_len*300,2)
 
    def forward(self,input):
        '''
        :param input:
        :return:
        '''
        input_embeded = self.embedding(input)
 
        input_embeded_viewed = input_embeded.view(input_embeded.size(0),-1)
 
        out = self.fc(input_embeded_viewed)
        return  F.log_softmax(out,dim=-1)