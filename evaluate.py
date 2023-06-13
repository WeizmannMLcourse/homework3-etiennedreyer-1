import torch
import numpy as np
from dataloader import PointCloudMNISTdataset, collate_graphs
from model import Net
import sys
from torch.utils.data import Dataset, DataLoader


def evaluate_on_dataset(path_to_ds=None,imin=-1,imax=-1):

	data = PointCloudMNISTdataset(path_to_ds)
	if imin == -1:
		imin = int(len(data)*0.9)
	if imax == -1:
		imax = len(data)
	test_ds = torch.utils.data.Subset(data,range(imin,imax))

	dataloader = DataLoader(test_ds,batch_size=1,collate_fn=collate_graphs)

	net = Net()


	state_dict = torch.load('trained_model.pt',map_location=torch.device('cpu'))
	net.load_state_dict(state_dict)

	total = 0
	correct = 0
    
	if torch.cuda.is_available():
		net.cuda()

	net.eval()

	n_batches = 0
	with torch.no_grad():
		for batched_g,y in dataloader:
			n_batches+=1

			if torch.cuda.is_available():
				batched_g = batched_g.to(torch.device('cuda'))
				y = y.cuda()
			pred = net(batched_g)
			
			pred = torch.argmax(pred,dim=1)

			correct+=len(torch.where(pred==y)[0])
			total+=len(y)
    
	return correct/total


if __name__ == "__main__":

	accuracy = evaluate_on_dataset(sys.argv[1])

	print(accuracy)

