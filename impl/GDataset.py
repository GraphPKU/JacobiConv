class GDataset:
    '''
    A class to put a splitted dataset.
    Args:
        x : node feature, of shape (number of nodes, F)
        mask : the mask to show whether a node is in the dataset, of shape (number of nodes) 
        y : the target
    '''
    def __init__(self, edge_index, edge_attr, mask, y):
        self.x = None 
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = mask

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.y[idx]

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        return self