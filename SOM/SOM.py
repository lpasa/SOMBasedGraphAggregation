import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
from torchvision.utils import save_image


class SOM(nn.Module):
    def __init__(self, input_size, out_size=(10, 10), sigma=None, device=None):
        '''
        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.input_size = input_size
        self.out_size = out_size


        if sigma is None:
            self.sigma = max(out_size) / 2
        else:
            self.sigma = float(sigma)

        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]), requires_grad=True)
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)

    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.out_size[0]):
            for y in range(self.out_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, x,sigma=None):
        '''
        Find the location of best matching unit.
        :param x: data
        :return: location of best matching unit, loss
        '''

        n_nodes = x.size()[0]
        x = x.view(n_nodes, -1, 1).to(self.device)

        node_weight = self.weight.expand(n_nodes, -1, -1)

        dists = self.pdist_fn(x, node_weight)



        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)

        bmu_locations = self.locations[bmu_indexes]

        #coefficenti per i neighborhood
        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        if sigma is not None:
            lr_locations = self._neighborhood_fn(distance_squares, sigma)
        else:
            lr_locations = self._neighborhood_fn(distance_squares, self.sigma)


        som_output=torch.exp(-dists.to(self.device)+losses.expand_as(dists).to(self.device)) * lr_locations



        return bmu_locations, losses.sum().div_(n_nodes).item(), som_output

    def forward_winner_only(self,x):

        n_nodes = x.size()[0]
        x = x.view(n_nodes, -1, 1).to(self.device)
        # use the same weights for each row of the input
        node_weight = self.weight.expand(n_nodes, -1, -1)

        dists = self.pdist_fn(x, node_weight)

        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)

        bmu_locations = self.locations[bmu_indexes]

        som_output=torch.zeros(n_nodes, self.out_size[0]*self.out_size[1]).to(self.device)


        for node_index,(weight_index,loss) in enumerate(zip(bmu_indexes,losses)):
            som_output[node_index,weight_index]=torch.exp(-loss)

        return bmu_locations, losses.sum().div_(n_nodes).item(), som_output


    def self_organizing(self, x, current_iter, max_iter, lr):
        '''
        Train the Self Oranizing Map(SOM)
        :param x: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''

        batch_size = x.size()[0]

        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = lr * iter_correction
        sigma = self.sigma * iter_correction

        #Find best matching unit
        bmu_locations, loss, _ = self.forward(x)

        #bmu_locations contains the winner location for each node
        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)


        lr_locations = self._neighborhood_fn(distance_squares, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr_locations * (x.unsqueeze(2) - self.weight)

        delta = delta.sum(dim=0)


        delta.div_(batch_size)


        #update the weights
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])

