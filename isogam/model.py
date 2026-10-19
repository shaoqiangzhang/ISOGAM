import torch
from IPython import get_ipython
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
# from torchvision.ops import MLP

from . utils import calculate_affinity
from . nets import MLP
from . graph_autoencoder import GraphAutoencoderISOGAM
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score, confusion_matrix, confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import kron
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import PCA
from PIL import Image
import scipy
import torch.nn.functional as F

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class Isogam(nn.Module):
    def __init__(self, features, ind_views='all', combs='all', sparse=False, neighbors = None, device='cpu', learnable_weights=False, true_labels=None, use_graph_encoder=False, spatial_locs=None, spatial_neighbors=6):
        super(Isogam, self).__init__()
        self.device = device
        self.num_views = len(features)
        self.features = [torch.Tensor(i).to(self.device) for i in features] 
        self.sparse = sparse
        self.learnable_weights = learnable_weights
        self.true_labels = true_labels  # Add the true labels

        # Add a dict to store the projection layers
        self.projectors = nn.ModuleDict() if self.learnable_weights else None
        
        features = [StandardScaler().fit_transform(i) for i in features]
        if neighbors is None and self.sparse:
            neighbors=100

        # Store top-50 PCs of standardized features for paper-correct ICC computation.
        # Paper: "For each modality, we computed the ICC for all clusters using the
        # top 50 PCs of the preprocessed data for that modality." (Methods section)
        # Fix random_state=0，Prevent PCA from consuming the global random state and affecting subsequent training initialization
        self.pcs_50 = [PCA(min(50, f.shape[1]), random_state=0).fit_transform(f) for f in features]

        adj = [calculate_affinity(i, sparse = self.sparse, neighbors=neighbors) for i in features]
        self.adj1 = adj
        pcs = [PCA(128, random_state=0).fit_transform(i) if i.shape[1] > 128 else i for i in features]
        self.pcs = [torch.Tensor(i).to(self.device) for i in pcs] 
        if not self.sparse:
          self.adj = [torch.Tensor(i).to(self.device) for i in adj]
        else:
          adj = [coo_matrix(i) for i in adj]
          indices = [torch.LongTensor(np.vstack((i.row, i.col))) for i in adj]
          values = [torch.FloatTensor(i.data) for i in adj]
          shape = [torch.Size(i.shape) for i in adj]
          self.adj = [torch.sparse.FloatTensor(indices[i], values[i], shape[i]).to(self.device) for i in range(len(adj))]

        # Alternative approach: superimpose the spatial position KNN graph on the expression KNN graph
        self.use_spatial_adj = False
        if spatial_locs is not None:
            from sklearn.neighbors import kneighbors_graph
            coords = spatial_locs[['4', '5']].values.astype(float)
            coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
            sp_knn = kneighbors_graph(coords_norm, n_neighbors=spatial_neighbors, mode='connectivity', include_self=True)
            sp_knn = (sp_knn + sp_knn.T)
            sp_dense = (sp_knn.toarray() > 0).astype(np.float32)
            # Symmetric normalization D^{-1/2} A D^{-1/2}
            deg = sp_dense.sum(axis=1)
            deg[deg == 0] = 1.0
            d_inv = np.diag(1.0 / np.sqrt(deg))
            sp_norm = (d_inv @ sp_dense @ d_inv).astype(np.float32)
            sp_tensor = torch.Tensor(sp_norm).to(self.device)
            self.spatial_adj = [sp_tensor] * self.num_views
            self.use_spatial_adj = True
            print(f"[ISOGAM] Spatial KNN graph construction completed, each spot connects to the nearest {spatial_neighbors} neighbors!")

        if ind_views=='all':
            self.ind_views = list(range(len(self.pcs)))
        else:
            self.ind_views = ind_views   
        if combs=='all':
            self.combinations = list(combinations(list(range(len(self.pcs))),2))
        else:
            self.combinations = combs        

        # Set weights of modalities
        self.weights = {
            'rna': 1.0,      
            'protein': 1.0,  
            'image_emb': 1.0 
        }
        
        # Store the original data dimension information for subsequent separation of data from different modalities
        self.modality_dims = [pc.shape[1] for pc in pcs]
        self.final_recon_losses = [None] * len(features)
        # If learnable weights are enabled, trainable parameters will be created
        if self.learnable_weights:
            # Create learnable weight parameters (initialized in log space to ensure positive values)
            self.log_weights = nn.ParameterDict({
                'rna': nn.Parameter(torch.tensor(np.log(self.weights['rna']), dtype=torch.float32)),
                'protein': nn.Parameter(torch.tensor(np.log(self.weights['protein']), dtype=torch.float32)),
                'image_emb': nn.Parameter(torch.tensor(np.log(self.weights['image_emb']), dtype=torch.float32))
            })
            
            # Create an attention network to dynamically adjust weights
            self.attention_networks = nn.ModuleDict({
                'rna': self._create_attention_network(),
                'protein': self._create_attention_network(),
                'image_emb': self._create_attention_network()
            })
            
            # Move the module to the specified device
            self.log_weights = self.log_weights.to(self.device)
            self.attention_networks = self.attention_networks.to(self.device)
            
            # Print initialization information
            print("Initialized learnable weights:")
            initial_weights = self.get_current_weights()
            for modality, weight in initial_weights.items():
                print(f"  {modality}: {weight:.4f}")
            
    def _create_attention_network(self):
        """
        Create an attention network and output positive quality weights (Softplus activation).
        Stage 2 training: Use the variance of embeddings as the quality signal, and the network learns to predict modality quality.
        High quality (large variance) ---> large weight; low quality (small variance) ---->small weight.
        """
        network = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus()
        )
        return network.to(self.device)

    def compute_attention_scores(self, embeddings):
        attention_scores = {}
        modality_names = list(self.weights.keys())

        for i, emb in enumerate(embeddings):
            if i < len(modality_names):
                modality_name = modality_names[i]
                # Z is 64-dimensional. Take the mean directly and feed it into the attention network (64-->32-->1).
                emb_mean = torch.mean(emb, dim=0, keepdim=True)  # (1, 64)
                attention_score = self.attention_networks[modality_name](emb_mean)
                attention_scores[modality_name] = attention_score.squeeze()

        return attention_scores

    def get_current_weights(self):
        """Get the current weight values (if learnable, retrieve them from the parameters)"""
        if self.learnable_weights:
            return {key: torch.exp(self.log_weights[key]).item() for key in self.weights.keys()}
        else:
            return self.weights

    def set_true_labels(self, labels):
        """Set ground truth labels"""
        self.true_labels = labels

    def train(self, num_epochs=1000):
        """Train the model"""
        # GraphAutoencoderISOGAM
        # input-->32(compress)+ orth(32-->32),GCN is only used for path reconstruction
        input_shapes = [self.pcs[i].shape[1] for i in self.ind_views]
        network = GraphAutoencoderISOGAM(
            input_shapes=input_shapes,
            latent_dims=[32] * len(input_shapes)
        )
        network = network.to(self.device)

        def sc_loss(A, Y):
            if not self.sparse:
                return (torch.triu(torch.cdist(Y, Y)) * torch.triu(A)).mean()
            else:
                row = A.coalesce().indices()[0]
                col = A.coalesce().indices()[1]
                dist = torch.norm(Y[row] - Y[col], dim=1)
                return (dist * A.coalesce().values()).mean()

        # Precompute adjacency matrix:
        # sc_loss :Only use the expression affinity graph (free from contamination by spatial graphs to ensure pure spectral clustering signals)
        # GCN Reconstruction with Spatial Fusion Graph (Innovation 1: Explicitly model spatial topology)
        # ===== Innovation 1: Modal Adaptive Spatial Weight =====
        # RNA/Protein:Retain the original expression level signal without smoothing to ensure clear molecular boundaries
        # Image：Histological features have spatial correlation, and the spatial smoothing of GCN embodies the spatial topological modeling of Innovation 1.
        modality_spatial_weights = [0.0, 0.0, 0.2]

        if self.use_spatial_adj:
            gcn_adj_list = []
            for vi, i in enumerate(self.ind_views):
                sw = modality_spatial_weights[vi] if vi < len(modality_spatial_weights) else 0.3
                gcn_adj_list.append(sw * self.spatial_adj[i] + (1.0 - sw) * self.adj[i])
        else:
            gcn_adj_list = [self.adj[i] for i in self.ind_views]
        sc_adj_list = [self.adj[i] for i in self.ind_views]

        # ===== Innovation 1 Core: Separate training GraphAutoencoder for each modality  =====
            for vi, view_idx in enumerate(self.ind_views):
            ae = network.autoencoders[vi]
            ae.train()
            # ===== Stage 1: The autoencoder is trained normally without introducing the attention network. =====
            # Maintain the original balance between sc_loss and recon_loss, unaffected by attention perturbations
            opt_i = optim.Adam(ae.parameters(), lr=1e-3)

            for epoch in tqdm(range(num_epochs),
                              desc=f'GraphAE training modality {vi+1}/{len(self.ind_views)}'):
                opt_i.zero_grad()
                recon, z = ae(self.pcs[view_idx], gcn_adj_list[vi])
                loss_recon = F.mse_loss(recon, self.pcs[view_idx])
                loss_sc    = sc_loss(sc_adj_list[vi], z)
                loss = loss_recon + loss_sc
                loss.backward()
                opt_i.step()
            self.final_recon_losses[vi] = loss_recon.item()

            ae.eval()

        # Obtain embeddings of each modality
        with torch.no_grad():
            embeddings = [
                network.autoencoders[vi].get_embeddings(self.pcs[self.ind_views[vi]])
                for vi in range(len(self.ind_views))
            ]

        # ===== Innovation 2 Core: Adaptive Modal Quality Weight Based on Attention Network=====
        # Use the pre-trained embeddings of each modality as inputs to the attention network (32-->16-->1).
        # Automatically calculate the mass weight of each modality, down-weight low-quality modalities and up-weight high-quality modalities.
        # The attention network is executed after the embedding training is completed, using the global mean vector of the embeddings
        # as a proxy signal for modal mass, output scalar weight with Softplus
        if self.learnable_weights:
            weight_keys = list(self.weights.keys())

            # ===== Stage 2: Independently train the attention network (Core of Innovation 2)=====
            # After the autoencoder training is completed, the sum of variances of embeddings from each modality is used as the quality signal.
            #Large variance --> dispersed embedding distribution ---> high modal information --->the attention network learns to assign high weights
            # Small variance --> concentrated embeddings ---> low modal information --->the attention network learns to assign low weights
            # This way, innovation 2 truly learns the modal mass without interfering with the autoencoder training in the first stage.
            with torch.no_grad():
                quality_targets = []
                for vi, emb in enumerate(embeddings):
                    recon_loss = self.final_recon_losses[vi] if self.final_recon_losses[vi] is not None else 1.0
                    quality_targets.append(1.0 / (recon_loss + 1e-8))
                quality_targets = torch.tensor(quality_targets, dtype=torch.float32)
                quality_targets = quality_targets / (quality_targets.mean() + 1e-8)

            opt_attn = optim.Adam(
                [p for net in self.attention_networks.values() for p in net.parameters()],
                lr=1e-3
            )
            for _ in range(200):
                opt_attn.zero_grad()
                loss_attn = torch.tensor(0.0)
                for vi, emb in enumerate(embeddings):
                    if vi < len(weight_keys):
                        modality_name = weight_keys[vi]
                        emb_mean = emb.mean(0, keepdim=True)
                        pred = self.attention_networks[modality_name](emb_mean)
                        target = quality_targets[vi].view(1, 1).to(self.device)
                        loss_attn = loss_attn + F.mse_loss(pred, target)
                loss_attn.backward()
                opt_attn.step()

            # Calculate the final weights using the trained attention network
            attention_weight_values = []
            with torch.no_grad():
                for vi, emb in enumerate(embeddings):
                    if vi < len(weight_keys):
                        modality_name = weight_keys[vi]
                        emb_mean = emb.mean(0, keepdim=True)
                        w = self.attention_networks[modality_name](emb_mean)
                        attention_weight_values.append(float(w.item()))
                    else:
                        attention_weight_values.append(1.0)

            attn_arr = np.array(attention_weight_values)
            attn_arr = attn_arr / (attn_arr.mean() + 1e-8)
            attn_arr = np.clip(attn_arr, 0.1, 5.0)

            print("Attention-based modality quality weights (trained on embedding variance):")
            for vi, (k, w) in enumerate(zip(weight_keys, attn_arr)):
                print(f"  {k}: {w:.4f}")

            weighted_embeddings = []
            for vi, emb in enumerate(embeddings):
                w = float(attn_arr[vi]) if vi < len(attn_arr) else 1.0
                weighted_embeddings.append(emb.cpu().numpy() * w)
            embeddings_np = weighted_embeddings
        else:
            embeddings_np = [emb.cpu().numpy() for emb in embeddings]

        # Fusion Embedding
        Y = [torch.Tensor(e) for e in embeddings_np]
        if self.combinations is not None:
            interactions = [Y[i][:, :, None] * Y[j][:, None, :] for i, j in self.combinations]
            interactions = [inter.reshape(inter.shape[0], -1) for inter in interactions]
            interactions = [torch.matmul(inter, torch.pca_lowrank(inter, q=32)[2]) for inter in interactions]
            interactions = [StandardScaler().fit_transform(e.cpu().detach().numpy()) for e in interactions]
            interactions = np.concatenate(interactions, 1)
            Y_scaled = [StandardScaler().fit_transform(e) for e in embeddings_np]
            Y_scaled = np.concatenate(Y_scaled, 1)
            self.emb = np.concatenate((Y_scaled, interactions), 1)
        else:
            self.emb = np.hstack(embeddings_np)

        # modality_indices based on latent_dim=32
        latent_dim_per_view = 32
        self.modality_indices = []
        start_idx = 0
        for _ in self.ind_views:
            self.modality_indices.append((start_idx, start_idx + latent_dim_per_view))
            start_idx += latent_dim_per_view
            
        # If learnable weights are enabled, print the final learned weight values
        if self.learnable_weights:
            final_weights = self.get_current_weights()
            print("Final learned modality weights:")
            for modality, weight in final_weights.items():
                print(f"  {modality}: {weight:.4f}")

    def cluster(self, n_clusters=10, use_deep_graph_clustering=False):
        """
        Perform clustering on the embedding representations
        
        参数:
        n_clusters (int): Number of clusters
        use_deep_graph_clustering (bool): Whether to use deep graph clustering (reserved parameter)
        """
        # Check if trained (emb attribute exists)
        if not hasattr(self, 'emb'):
            raise ValueError("The model has not been trained. Please call the train() method first.")
        
        # Perform clustering using K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=100)
        communities = kmeans.fit_predict(self.emb)
        self.clusters = communities
        return communities

    def calculate_icc_distribution(self, modality_idx=0):
        """
        Return the list of ICC values in the box plot format for the paper: K clusters × D principal components = K×D ICC values.
        This fully corresponds to the box plot in Extended Data Fig.1 of the paper.

        Args:
            modality_idx: Modal index, 0=RNA,1=Protein,2=Image
        Returns:
            list: The ICC values for all (cluster, feature) pairs, which can be directly used to plot a box plot with matplotlib
        """
        if not hasattr(self, 'clusters'):
            raise ValueError("Please run the cluster() method first")
        if not hasattr(self, 'pcs_50') or modality_idx >= len(self.pcs_50):
            raise ValueError("pcs_50 does not exist or the mode index is out of range")

        data = self.pcs_50[modality_idx]
        unique_clusters = np.unique(self.clusters)

        cluster_means = np.array([data[self.clusters == c].mean(axis=0)
                                   for c in unique_clusters])
        sigma2_m = cluster_means.var(axis=0)

        icc_all = []
        for c in unique_clusters:
            cluster_data = data[self.clusters == c]
            if len(cluster_data) < 2:
                continue
            sigma2_i = cluster_data.var(axis=0)
            denom = sigma2_i + sigma2_m
            icc_per_feature = np.where(denom > 1e-12, sigma2_m / denom, 0.0)
            icc_all.extend(icc_per_feature.tolist())

        return icc_all

    def calculate_clustering_metrics(self, true_labels=None):
        """
        Calculate multiple clustering evaluation metrics
        """
        if not hasattr(self, 'clusters'):
            raise ValueError("Please run the cluster method for clustering first")
        
        # If the true_labels parameter is passed in, use it; otherwise, use the true_labels set during initialization.
        if true_labels is not None:
            actual_true_labels = true_labels
        else:
            actual_true_labels = self.true_labels
            
        # Calculate the silhouette coefficient
        silhouette = silhouette_score(self.emb, self.clusters)
        
        # Calculate the Within-cluster sum of squares
        # directly using self.clusters instead of re-running KMeans
        unique_labels = np.unique(self.clusters)
        centers = np.array([self.emb[self.clusters == c].mean(axis=0) for c in unique_labels])
        wcss = float(sum(
            np.sum((self.emb[self.clusters == c] - centers[i]) ** 2)
            for i, c in enumerate(unique_labels)
        ))
        
        # Calculate the Between-cluster sum of squares
        # Total sum of squares = Within-cluster sum of squares + Between-cluster sum of squares
        total_ss = np.sum((self.emb - np.mean(self.emb, axis=0)) ** 2)
        bcss = total_ss - wcss
        
        # Calculate the Calinski-Harabasz index
        n_clusters = len(np.unique(self.clusters))
        if n_clusters > 1:
            ch_score = (bcss / (n_clusters - 1)) / (wcss / (len(self.emb) - n_clusters))
        else:
            ch_score = 0
        
        # Calculate the Davies-Bouldin index
        db_score = self._calculate_davies_bouldin_index()
        
        # Calculate the overall ICC (Intraclass Correlation Coefficient)
        icc_score = self._calculate_icc()
        
        # Calculate the ICC for each modality
        rna_icc_score = self._calculate_rna_icc()
        protein_icc_score = self._calculate_protein_icc()
        image_icc_score = self._calculate_image_icc()
        
        metrics = {
            'silhouette_score': silhouette,
            'within_cluster_sum_of_squares': wcss,
            'between_cluster_sum_of_squares': bcss,
            'total_sum_of_squares': total_ss,
            'calinski_harabasz_index': ch_score,
            'davies_bouldin_index': db_score,
            'icc': icc_score,
            'rna_icc': rna_icc_score,
            'protein_icc': protein_icc_score,
            'image_icc': image_icc_score
        }
            
        return metrics

    def calculate_silhouette_score(self):
        """
        Calculate the silhouette coefficien
        """
        if not hasattr(self, 'clusters'):
            raise ValueError("Please run the cluster method for clustering first.")
        return silhouette_score(self.emb, self.clusters)

    def _calculate_davies_bouldin_index(self):
        """
        Calculate the Davies-Bouldin index
        """
        n_clusters = len(np.unique(self.clusters))
        cluster_centers = []
        
        # Calculate the center of each cluster
        for i in range(n_clusters):
            cluster_points = self.emb[self.clusters == i]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers)
        
        # Calculate the scatter measure (average distance to the cluster center) for each cluster
        scatter_metrics = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_points = self.emb[self.clusters == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(np.sum((cluster_points - cluster_centers[i]) ** 2, axis=1))
                scatter_metrics[i] = np.mean(distances)
        
        # Calculate the inter-cluster distance matrix
        cluster_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    cluster_distances[i, j] = np.sqrt(np.sum((cluster_centers[i] - cluster_centers[j]) ** 2))
        
        # Calculate Davies-Bouldin index
        db_values = np.zeros(n_clusters)
        for i in range(n_clusters):
            ratios = []
            for j in range(n_clusters):
                if i != j:
                    ratio = (scatter_metrics[i] + scatter_metrics[j]) / cluster_distances[i, j]
                    ratios.append(ratio)
            if ratios:
                db_values[i] = np.max(ratios)
        
        return np.mean(db_values)

    def _calculate_icc(self):
        """
        Overall ICC: mean across all modalities using paper's formula.
        Falls back to embedding-based computation when pcs_50 not available.
        """
        if hasattr(self, 'pcs_50') and len(self.pcs_50) > 0:
            icc_vals = [self._calculate_modality_icc(self.pcs_50[i])
                        for i in self.ind_views]
            return float(np.mean(icc_vals))
        # Fallback: use first 50 dims of embedding
        data = self.emb[:, :min(50, self.emb.shape[1])]
        return self._calculate_modality_icc(data)

    def _calculate_modality_icc(self, modality_pcs):
        """
        Paper-correct ICC formula (Methods section):
            ICC_i^(k) = σ²_m / (σ²_i + σ²_m)

        where for each cluster i and each feature j:
            σ²_i,j = within-cluster variance of feature j in cluster i
            σ²_m,j = variance of per-cluster means for feature j across all clusters

        Computed on top-50 PCs of preprocessed modality data.
        Returns the mean ICC across all (cluster, feature) pairs,
        matching the paper's box-plot of 1250 values (50 PCs × 25 clusters).

        Args:
            modality_pcs: np.ndarray of shape (N, D), D ≤ 50
        """
        unique_clusters = np.unique(self.clusters)
        K = len(unique_clusters)
        if K < 2:
            return 0.0

        data = modality_pcs  # (N, D)

        # Cluster means: (K, D)
        cluster_means = np.array([data[self.clusters == c].mean(axis=0)
                                   for c in unique_clusters])

        # σ²_m,j: variance of cluster means per feature — shape (D,)
        sigma2_m = cluster_means.var(axis=0)

        icc_all = []
        for k_idx, c in enumerate(unique_clusters):
            cluster_data = data[self.clusters == c]
            if len(cluster_data) < 2:
                continue
            # σ²_i,j: within-cluster variance per feature — shape (D,)
            sigma2_i = cluster_data.var(axis=0)
            denom = sigma2_i + sigma2_m
            # Per-feature ICC for this cluster
            icc_per_feature = np.where(denom > 1e-12, sigma2_m / denom, 0.0)
            icc_all.extend(icc_per_feature.tolist())

        return float(np.mean(icc_all)) if icc_all else 0.0

    def _calculate_rna_icc(self):
        """RNA ICC using paper formula on top-50 PCs of RNA data."""
        if hasattr(self, 'pcs_50') and len(self.pcs_50) > 0:
            return self._calculate_modality_icc(self.pcs_50[self.ind_views[0]])
        rna_data = self.emb[:, self.modality_indices[0][0]:self.modality_indices[0][1]]
        return self._calculate_modality_icc(rna_data)

    def _calculate_protein_icc(self):
        """Protein ICC using paper formula on top-50 PCs of protein data."""
        if hasattr(self, 'pcs_50') and len(self.pcs_50) > 1:
            return self._calculate_modality_icc(self.pcs_50[self.ind_views[1]])
        if len(self.modality_indices) > 1:
            protein_data = self.emb[:, self.modality_indices[1][0]:self.modality_indices[1][1]]
            return self._calculate_modality_icc(protein_data)
        return 0.0

    def _calculate_image_icc(self):
        """Image ICC using paper formula on top-50 PCs of image feature data."""
        if hasattr(self, 'pcs_50') and len(self.pcs_50) > 2:
            return self._calculate_modality_icc(self.pcs_50[self.ind_views[2]])
        if len(self.modality_indices) > 2:
            image_data = self.emb[:, self.modality_indices[2][0]:self.modality_indices[2][1]]
            return self._calculate_modality_icc(image_data)
        return 0.0