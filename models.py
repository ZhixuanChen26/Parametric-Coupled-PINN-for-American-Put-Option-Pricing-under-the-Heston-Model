import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh'):
        super(FCN, self).__init__()
        activations = {
            'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid,
            'gelu': nn.GELU, 'elu': nn.ELU
        }
        act = activations[activation.lower()]()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(act)

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(act)

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

def create_models(device):
    hidden_dims_price = [64, 128, 256, 128, 64]
    hidden_dims_free_boundary = [64, 128, 128, 128, 32]

    priceNN = FCN(input_dim=7, hidden_layers=hidden_dims_price, output_dim=1,
                     activation='gelu').to(device)

    freeBoundaryNN = FCN(input_dim=6, hidden_layers=hidden_dims_free_boundary, output_dim=1,
                     activation='gelu').to(device)

    last_layer = freeBoundaryNN.net[-1]
    nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
    nn.init.constant_(last_layer.bias, 0.90)

    return priceNN, freeBoundaryNN
