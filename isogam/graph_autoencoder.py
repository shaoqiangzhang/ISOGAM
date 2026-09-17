import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
import numpy as np


class GraphAutoencoder(nn.Module):
    """
    GraphAutoencoder: input→32 + orth(32→32)，
    The innovation: the reconstruction path incorporates GCN spatial smoothing, 
    while the sc_loss path retains the pure expression signal.
    """
    def __init__(self, input_dim, latent_dim=32, output_dim=None):
        super(GraphAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

        # layer1: compress to latent_dim
        self.encoder_compress = nn.Linear(input_dim, latent_dim)
        # layer3: Orthogonal rotation layer, 32×32 matrix
        # Orthogonal constraint on the square matrix = pure rotation, 
        # no information loss, spectral clustering structure preserved
        self.encoder_orth = orthogonal(nn.Linear(latent_dim, latent_dim))

        # === decoder ===
        # layer4: Decode back to the original dimension
        self.decoder = nn.Linear(latent_dim, self.output_dim)

        # === Innovation point: GCN spatial smoothing layer (used only for reconstruction path)===
        # For the sc_loss path, use the pure encoder output z (without passing through GCN) 
        # to ensure the clustering signal is free from spatial interference.
        # Use z_gcn obtained by smoothing the reconstructed path with GCN, 
        #and introduce spatial topological information to improve reconstruction quality
        self.graph_conv1 = GraphConvLayer(latent_dim, latent_dim)
        self.graph_conv2 = GraphConvLayer(latent_dim, latent_dim)

    def encode(self, x):
        """encode, No activation function"""
        z = self.encoder_compress(x)   # input→32
        z = self.encoder_orth(z)        # orth(32→32)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, adj=None):
        """
        Forward Propagation:
        - z = pure encoder output, for sc_loss (without passing through GCN)
        - reconstructed = Reconstructed output smoothed by GCN
        """
        z = self.encode(x)

        # GCN is only used for the reconstruction path (without affecting the clustering signal)
        if adj is not None:
            z_gcn = self.graph_conv1(z, adj)
            z_gcn = F.relu(z_gcn)
            z_gcn = self.graph_conv2(z_gcn, adj)
            reconstructed = self.decode(z_gcn)
        else:
            reconstructed = self.decode(z)

        return reconstructed, z

    def get_embeddings(self, x, adj=None):
        """Return the pure encoder output"""
        with torch.no_grad():
            return self.encode(x)


class GraphConvLayer(nn.Module):
    """Simple Graph Convolution Layer: AXW"""
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x


class GraphAutoencoderISOGAM(nn.Module):
    """GraphAutoencoder for each modality"""
    def __init__(self, input_shapes, latent_dims=None, hidden_dims=None):
        super(GraphAutoencoderISOGAM, self).__init__()

        self.num_views = len(input_shapes)
        self.input_shapes = input_shapes

        if latent_dims is None:
            self.latent_dims = [32] * self.num_views
        else:
            self.latent_dims = latent_dims

        self.autoencoders = nn.ModuleList([
            GraphAutoencoder(input_shapes[i], self.latent_dims[i])
            for i in range(self.num_views)
        ])

    def forward(self, x_list, adj_list=None):
        reconstructed_list, latent_list = [], []
        for i, x in enumerate(x_list):
            adj = adj_list[i] if adj_list is not None else None
            recon, z = self.autoencoders[i](x, adj)
            reconstructed_list.append(recon)
            latent_list.append(z)
        return reconstructed_list, latent_list

    def get_embeddings(self, x_list, adj_list=None):
        return [self.autoencoders[i].get_embeddings(x_list[i])
                for i in range(len(x_list))]