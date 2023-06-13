
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import dgl
import h5py


def collate_graphs(samples):
    
    # The input `samples` is a list, a batch of whatever comes out of your dataset object
    
    graphs = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    
    batched_graph = dgl.batch(graphs)
    targets = torch.tensor(labels).long()
    
    return batched_graph, targets


class PointCloudMNISTdataset(Dataset):
    def __init__(self, path):
        
        with h5py.File(path,'r') as f:
            self.x = f['x'][:]
            self.y = f['y'][:]
            self.n_points = f['n_points'][:]
            self.label = f['label'][:]
        
    def __len__(self):
       
        return len(self.label)

    def __getitem__(self, idx):
            
        g = dgl.graph(([],[]),num_nodes=self.n_points[idx])
        x_tensor = torch.FloatTensor(self.x[idx])
        y_tensor = torch.FloatTensor(self.y[idx])
        g.ndata['xy'] = torch.stack((x_tensor,y_tensor),dim=1)
        ### Normalize to the range [-1,1] ###
        g.ndata['xy'] = (g.ndata['xy'] - 28/2) / (28/2)
        
        y = self.label[idx]
        
        return g, y