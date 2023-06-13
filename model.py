
import torch
import numpy as np
import torch.nn as nn
import dgl
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #hyperparams (optional)
        # - number of input features
        # - number of hidden features
        # - number of message-passing blocks

        #encoding layer
        self.encode = ## add neural network here

        #message-passing layers
        self.update_blocks = nn.ModuleList() #important to use a ModuleList rather than simply a list

        for i in range(# number of message-passing blocks):
            self.update_blocks.append( ## add neural network here

        #classification layers
        self.classify = ## add neural network here
       
    def forward(self,g):
        
        #encode 2d inputs --> hidden representation
        g.ndata['hidden rep'] = self.encode( ## input features 

        #message passing loops
        for i in range(#number of message-passing blocks

            #calculate global representation as mean over hidden reps of nodes
            global_rep            = dgl.mean_nodes( ## take mean over hidden reps of nodes
            g.ndata['global rep'] = # use dgl.broadcast_nodes to put the global rep tensor onto each node

            #concatenate hidden rep, global rep, and xy node features (last one optional but it helps a lot)
            hidden_global_xy     = # ...

            #deep learning layers
            g.ndata['hidden rep'] = self.update_blocks #use the ith update block to update the hidden rep

        #the final global representation
        global_rep = ## copy what we did above

        #final classification
        output = #...

        return output