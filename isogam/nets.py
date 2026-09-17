import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import orthogonal

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
        self.decoder = nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.decoder(x)
        x = self.relu(x)
        return x

    def get_embeddings(self,x):
        x = self.encoder(x)
        x = self.relu(x)
        return x
        
class MLP(nn.Module):
    # def __init__(self, **kwargs):
    #     super().__init__()
    #     self.layer1 = nn.Linear(kwargs["input_shape"], 32)
    #     #self.layer2 = nn.Linear(64, kwargs["output_shape"])
    #     self.layer3 = orthogonal(nn.Linear(kwargs["output_shape"], kwargs["output_shape"]))
    #     self.layer4 = nn.Linear(32,kwargs["input_shape"])
    #     self.relu = nn.ReLU()
    #     self.tanh = nn.Tanh()
    #     self.dropout = nn.Dropout()
    #     #self.softmax = nn.Softmax()

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer1 = nn.Linear(input_shape, 32)
        # self.layer2 = nn.Linear(64, output_shape)
        self.layer3 = orthogonal(nn.Linear(32, 32))  # Match the output dimension of layer1
        self.layer4 = nn.Linear(32, input_shape)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        #x = self.tanh(x)
        #x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.relu(x)
        #x1 = x.T@x + torch.eye(x.shape[1]).to(x.device)*1e-7                             
        #l = torch.cholesky(x1)
        #x = x@((x.shape[0])**(1/2)*l.inverse().T)
        #x, _ = torch.qr(x)
        return x

    def get_embeddings(self, x):
        x = self.layer1(x)
        x = self.layer3(x)
        return x

class MLP1(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer1 = nn.Linear(32, kwargs["output_shape"])
        self.layer2 = orthogonal(nn.Linear(kwargs["output_shape"], kwargs["output_shape"]))
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        #x = self.relu(x)
        x = self.layer2(x)
        #x = self.relu(x)
        #x1 = x.T@x + torch.eye(x.shape[1]).to(x.device)*1e-7
        #l = torch.cholesky(x1)
        #x = x@((x.shape[0])**(1/2)*l.inverse().T)
        #x, _ = torch.qr(x)
        return x

# Add auxiliary network components for GraphAutoencoder
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.4)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.4)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GraphConvNet(nn.Module):
    """
    Simple Graph Convolutional Network
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GraphConvNet, self).__init__()

        self.gc1 = GraphConvLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvLayer(nn.Module):
    """
    simple graph convolutional layer
    """
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output