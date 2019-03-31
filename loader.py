import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder

import torch.nn.functional as F
from torch import nn, optim

class CustomLoader(Dataset):
    def __init__(self, chemin):
        """
            dataset = CustomLoader('./iris.csv')
            trainloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
            (pin_memory=true and num_workers=n if GPU)
        """
        xy = pd.read_csv(chemin)
        xy.loc[(xy.age<=40) & (xy.age>=29), 'age'] = 1
        xy.loc[(xy.age<=50) & (xy.age>=41), 'age'] = 2
        xy.loc[(xy.age<=60) & (xy.age>=51), 'age'] = 3
        xy.loc[(xy.age<=70) & (xy.age>=61), 'age'] = 4
        xy.loc[(xy.age<=80) & (xy.age>=71), 'age'] = 5
        xy.loc[:,'trestbps'] /= 300
        xy.loc[:,'chol'] /= 564
        xy.loc[:,'thalach'] /= 202

        x = xy.iloc[:, 0:-1]
        y = xy.iloc[:, -1]
        self.len = x.shape[0]
        self.x_data = torch.tensor(x.values).float()
        self.y_data = torch.tensor(y.values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len