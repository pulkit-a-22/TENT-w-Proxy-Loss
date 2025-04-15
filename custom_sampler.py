import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

class NNBatchSampler(Sampler):
    """
    BatchSampler that, for each selected query image, returns its nearest neighbors.
    For a given batch_size, it selects (batch_size // nn_per_image) query images,
    and returns each query’s nn_per_image nearest neighbors as one batch.
    """
    def __init__(self, data_source, model, seen_dataloader, batch_size, nn_per_image=10, using_feat=True, is_norm=False):
        self.batch_size = batch_size
        self.nn_per_image = nn_per_image
        self.using_feat = using_feat
        self.is_norm = is_norm
        self.data_source = data_source
        self.num_samples = len(data_source)
        # Precompute the nearest neighbor matrix and distances
        self.nn_matrix, self.dist_matrix = self._build_nn_matrix(model, seen_dataloader)

    def __iter__(self):
        # Yield batches until we cover a full epoch.
        for _ in range(len(self)):
            yield self.sample_batch()

    def _predict_batchwise(self, model, seen_dataloader):
        # Collect features for the entire dataset.
        device = next(model.parameters()).device
        model_training_state = model.training
        model.eval()

        features = []
        # We assume that seen_dataloader returns batches (images, labels) or similar.
        for images, _ in tqdm(seen_dataloader, desc="Extracting features for NN"):
            images = images.to(device)
            with torch.no_grad():
                # Pass the images through the model to get features.
                # Here we assume model returns a tuple: (logits, features).
                out = model(images)
                if isinstance(out, tuple):
                    feat = out[1]
                else:
                    feat = out
                if self.is_norm:
                    feat = F.normalize(feat, p=2, dim=1)
            features.append(feat.cpu())
        model.train(model_training_state)
        features = torch.cat(features, dim=0)
        return features

    def _build_nn_matrix(self, model, seen_dataloader):
        # Compute the features for all samples.
        X = self._predict_batchwise(model, seen_dataloader)
        N = X.size(0)
        # Compute pairwise Euclidean distances (or cosine distances after normalization).
        # Here we use a simple L2 distance: d(i,j)=||x_i - x_j||^2.
        # Adding a small epsilon for numerical stability.
        eps = 1e-12
        # Compute squared norms for each row.
        squared_norms = X.pow(2).sum(dim=1, keepdim=True)
        # Compute the distance matrix using the expanded formula.
        dist_matrix = squared_norms + squared_norms.t() - 2 * X @ X.t() + eps
        # For each sample, find the indices of the nn_per_image nearest neighbors.
        # We use topk with largest=False.
        nn_matrix = dist_matrix.topk(self.nn_per_image, largest=False)[1]
        return nn_matrix, dist_matrix

    def sample_batch(self):
        # Determine the number of query images per batch.
        num_queries = self.batch_size // self.nn_per_image
        # Randomly select query indices from the dataset.
        query_indices = np.random.choice(self.num_samples, num_queries, replace=False)
        # For each query, get its nn_per_image nearest neighbors.
        sampled_indices = []
        for qi in query_indices:
            neighbors = self.nn_matrix[qi].numpy()
            sampled_indices.extend(neighbors.tolist())
        return sampled_indices

    def __len__(self):
        # Define the number of batches per epoch.
        return self.num_samples // self.batch_size
