from os import get_blocking
import torch
import numpy as np
import torch.nn as nn
import dgl
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #hyperparams
        self.ninput = 2  #input features
        self.nhid = 100  #features of node embedding
        self.npass = 3   #message-passing steps

        #encoding layer
        self.encode = nn.Sequential(nn.Linear(self.ninput,self.nhid),nn.ReLU())

        #message-passing layers
        self.update_blocks = nn.ModuleList()

        for i in range(self.npass):
            self.update_blocks.append(
                nn.Sequential(
                    nn.Linear(self.nhid*2 + self.ninput,self.nhid*3),
                    nn.BatchNorm1d(self.nhid*3),
                    nn.ReLU(),
                    nn.Linear(self.nhid*3,self.nhid*2),
                    nn.BatchNorm1d(self.nhid*2),
                    nn.ReLU(),
                    nn.Linear(self.nhid*2,self.nhid),
                    nn.BatchNorm1d(self.nhid),
                )
            )

        #classification layers
        self.classify = nn.Sequential(
            nn.Linear(self.nhid,200),
            nn.ReLU(),
            nn.Linear(200,50),
            nn.ReLU(),
            nn.Linear(50,10),
        )
       
    def forward(self,g):
        
        #embed 2d inputs --> hidden representation
        g.ndata['hidden rep'] = self.encode(g.ndata['xy'])

        #message passing loops
        for i in range(self.npass):

            #calculate global representation as mean over hidden reps of nodes
            global_rep            = dgl.mean_nodes(g,'hidden rep')
            g.ndata['global rep'] = dgl.broadcast_nodes(g,global_rep)

            #concatenate hidden, global, and xy
            hidden_global_xy     = torch.cat((g.ndata['hidden rep'],g.ndata['global rep'],g.ndata['xy']),dim=1)

            #deep learning layers
            g.ndata['hidden rep'] = self.update_blocks[i](hidden_global_xy)

        #the final global representation
        global_rep = dgl.mean_nodes(g,'hidden rep')

        #final classification
        output = self.classify(global_rep)

        return output