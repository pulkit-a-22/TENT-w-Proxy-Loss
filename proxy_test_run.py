import torch
import numpy as np
from proxy_net.resnet import Resnet18  # or your chosen model
from loss_proxy import neighbor_proj_loss, Momentum_Update

# ------------------ Load Corrupted Data ------------------ #
# Load images and labels from the cifar10c folder.
# Here we test on "gaussian_noise.npy"
img_array = np.load("data/cifar10c/gaussian_noise.npy")  # Expected shape: (N, H, W, C)
labels_array = np.load("data/cifar10c/labels.npy")         # Expected shape: (N,)

# Convert images to torch tensor and permute to (N, C, H, W)
images = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
labels = torch.from_numpy(labels_array).long()

# For testing, select a small batch (e.g., first 8 images)
batch_size = 120
images_batch = images[:batch_size].cuda()
labels_batch = labels[:batch_size].cuda()
dummy_idx = torch.arange(batch_size).cuda()
dummy_epoch = 0

# ------------------ Initialize Models ------------------ #
# Instantiate student and teacher models (using Resnet18 as an example)
model_student = Resnet18(embedding_size=512, pretrained=False, is_norm=True, is_student=True).cuda()
model_teacher = Resnet18(embedding_size=512, pretrained=False, is_norm=True, is_student=False).cuda()

# Freeze teacher parameters
for param in model_teacher.parameters():
    param.requires_grad = False

# Forward passes
# Assume model_student returns a tuple (x_g, x_f); we use the embedding x_f
student_out = model_student(images_batch)
teacher_out = model_teacher(images_batch)

# ------------------ Initialize Proxy Loss ------------------ #
# For testing, we can use a dummy args object containing the required hyperparameters.
# In practice, these should come from your configuration.
dummy_args = type('dummy_args', (), {
    'embedding_size': 512,
    'bg_embedding_size': 512,
    'num_proxies': 100,
    'num_dims': 3,
    'num_neighbors': 10,
    'projected_power': 1.0,
    'residue_power': 1.0,
    'use_gaussian_sim': False,
    'use_projected': True,
    'use_additive': False,
    'proxy_norm': True,
    'num_local': 5,
    'no_proxy': False,
    'only_proxy': False
})

proxy_loss_fn = neighbor_proj_loss(
    args=dummy_args,
    sigma=1.0,
    delta=2.0,
    view=0,
    disable_mu=False,
    topk=5
).cuda()

# ------------------ Compute and Test Proxy Loss ------------------ #
student_embedding = student_out[1]
loss_dict = proxy_loss_fn(student_embedding, teacher_out, dummy_idx, labels_batch, dummy_epoch)
print("Proxy Loss Dictionary:", loss_dict)

# Test backward pass to ensure gradients flow
loss = loss_dict['loss']
loss.backward()
print("Backward pass successful!")
