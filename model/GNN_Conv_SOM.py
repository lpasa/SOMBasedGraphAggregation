import torch
from torch_geometric.nn.conv import GraphConv
from SOM.SOM import SOM
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
from torch.functional import F
from torch.nn.init import eye_
import numpy as np
import os

class GNN_Conv_SOM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_class=2, som_grid_dims=(10,10), dropout=0,  device=None):
        super(GNN_Conv_SOM, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.n_class = n_class
        self.som_grid_dims=som_grid_dims

        self.conv0=GraphConv(self.in_channels, self.in_channels, bias=False)


        self.conv1=GraphConv(self.in_channels, self.out_channels)
        self.conv2=GraphConv(self.out_channels, out_channels*2)
        self.conv3 = GraphConv(self.out_channels*2, out_channels*3)

        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()
        self.act3 = torch.nn.LeakyReLU()


        self.norm1= torch.nn.BatchNorm1d(self.out_channels)
        self.norm2 = torch.nn.BatchNorm1d(self.out_channels*2)
        self.norm3 = torch.nn.BatchNorm1d(self.out_channels*3)

        self.dropout = torch.nn.Dropout(p=dropout)

        #define readout of GNN_only model
        self.lin_GNN = torch.nn.Linear((out_channels+out_channels*2+self.out_channels*3)*3, n_class)


        #define som for read out

        self.som1=SOM(out_channels, out_size=som_grid_dims,device=self.device)
        self.som2=SOM(out_channels*2, out_size=som_grid_dims ,device=self.device)
        self.som3=SOM(out_channels*3, out_size=som_grid_dims, device=self.device)

        # define  read_out
        self.out_conv1 = GraphConv(som_grid_dims[0] * som_grid_dims[1], self.out_channels)
        self.out_conv2 = GraphConv(som_grid_dims[0] * som_grid_dims[1], self.out_channels)
        self.out_conv3 = GraphConv(som_grid_dims[0] * som_grid_dims[1], self.out_channels)

        self.out_norm1=torch.nn.BatchNorm1d(self.out_channels)
        self.out_norm2=torch.nn.BatchNorm1d(self.out_channels)
        self.out_norm3=torch.nn.BatchNorm1d(self.out_channels)

        self.lin_out = torch.nn.Linear(self.out_channels * 3 *3, n_class)
        self.out_fun = torch.nn.LogSoftmax(dim=1)

        self.reset_prameters()


    def reset_prameters(self):
        eye_(self.conv0.weight)
        self.conv0.weight.requires_grad=False
        eye_(self.conv0.lin.weight)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

        self.lin_GNN.reset_parameters()

        self.out_conv1.reset_parameters()
        self.out_conv2.reset_parameters()
        self.out_conv3.reset_parameters()

        self.out_norm1.reset_parameters()
        self.out_norm2.reset_parameters()
        self.out_norm3.reset_parameters()

        self.lin_out.reset_parameters()

    def get_som_weights(self):
        return self.som1.weight, self.som2.weight, self.som3.weight

    def forward(self, data,conv_train=False):

        x = data.x

        edge_index = data.edge_index



        x1 = self.norm1(self.act1(self.conv1(x, edge_index)))
        x = self.dropout(x1)

        x2 = self.norm2(self.act2(self.conv2(x, edge_index)))
        x = self.dropout(x2)

        x3 = self.norm3(self.act3(self.conv3(x, edge_index)))

        h_conv = torch.cat([x1,x2,x3], dim=1)

        #compute GNN only output

        conv_batch_avg = gap(h_conv, data.batch)
        conv_batch_add = gadd(h_conv, data.batch)
        conv_batch_max = gmp(h_conv, data.batch)

        h_GNN = torch.cat([conv_batch_avg, conv_batch_add, conv_batch_max], dim=1)

        gnn_out = self.out_fun(self.lin_GNN(h_GNN))


        if conv_train:
            return None,None,gnn_out

        #SOM
        _,_,som_out_1=self.som1(x1)
        _,_,som_out_2=self.som2(x2)
        _,_,som_out_3=self.som3(x3)


        #READOUT
        h1=self.out_norm1(self.act1(self.out_conv1(som_out_1,edge_index)))
        h2=self.out_norm2(self.act2(self.out_conv2(som_out_2,edge_index)))
        h3=self.out_norm3(self.act3(self.out_conv3(som_out_3,edge_index)))


        som_out_conv=torch.cat([h1,h2,h3], dim=1)

        som_batch_avg = gap(som_out_conv, data.batch)
        som_batch_add = gadd(som_out_conv, data.batch)
        som_batch_max = gmp(som_out_conv, data.batch)

        h=torch.cat([som_batch_avg, som_batch_add, som_batch_max], dim=1)

        h = self.lin_out(h)
        h = self.out_fun(h)

        return h,h_conv,gnn_out