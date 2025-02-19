import os
import math
import time
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# Import dataset, net, utils and loss_proxy from your repository
import proxy_dataset
import proxy_utils
import loss_proxy
from proxy_dataset import sampler

# Import models from provided files
from net.resnet import Resnet18, Resnet34, Resnet50
from net.inception import inception_v1
from net.bn_inception import bn_inception

# --------------------- Set Seeds and Parse Arguments --------------------- #
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(
    description='Proxy Training for Unsupervised Deep Metric Learning (Proxy Loss Integration)'
)
# Directories
parser.add_argument('--LOG_DIR', default='./logs_proxy', help='Path to log folder')
parser.add_argument('--DATA_DIR', default='./data', help='Path to dataset root')
# Dataset
parser.add_argument('--dataset', default='cub', choices=['cub', 'cars', 'SOP'],
                    help='Training dataset')
# GPU settings
parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU for training, -1 for DataParallel')
parser.add_argument('--workers', default=8, type=int, help='Number of workers for dataloader')
# Model selection and hyperparameters
parser.add_argument('--model', default='bn_inception', help='Model architecture to use')
parser.add_argument('--embedding_size', default=512, type=int, help='Embedding size')
parser.add_argument('--bg_embedding_size', default=512, type=int, help='Background embedding size')
parser.add_argument('--pretrained', default=True, type=utils.bool_flag, help='Use pretrained model')
parser.add_argument('--bn_freeze', default=0, type=int, help='Freeze BN parameters')
# Training hyperparameters
parser.add_argument('--batch-size', default=120, type=int, dest='sz_batch', help='Batch size')
parser.add_argument('--epochs', default=90, type=int, dest='nb_epochs', help='Number of epochs')
parser.add_argument('--optimizer', default='adam', help='Optimizer type')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--emb-lr', default=1e-4, type=float, help='Learning rate for embedding layer')
parser.add_argument('--fix_lr', default=False, type=utils.bool_flag, help='Fix the learning rate')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay')
# Proxy loss hyperparameters
parser.add_argument('--momentum', default=0.999, type=float, help='Momentum update parameter for teacher')
parser.add_argument('--proxy_momentum', default=0.999, type=float, help='Momentum update for proxy loss (if used)')
parser.add_argument('--proxy_lr_mult', default=100, type=float, help='Learning rate multiplier for proxy parameters')
parser.add_argument('--num_neighbors', default=5, type=int, help='Number of neighbors per query (for sampler)')
parser.add_argument('--num_dims', default=3, type=int, help='Dimensionality of proxy planes')
parser.add_argument('--num_local', default=5, type=int, help='Number of local neighbors for proxy plane loss')
parser.add_argument('--num_proxies', default=100, type=int, help='Number of proxies')
# Similarity options
parser.add_argument('--use_teacher', default=True, type=utils.bool_flag, help='Use teacher embedding for loss calculation')
parser.add_argument('--student_norm', default=True, type=utils.bool_flag, help='L2 normalize student embedding')
parser.add_argument('--teacher_norm', default=True, type=utils.bool_flag, help='L2 normalize teacher embedding')
parser.add_argument('--proxy_norm', default=True, type=utils.bool_flag, help='Normalize proxy embeddings')
parser.add_argument('--use_gaussian_sim', default=False, type=utils.bool_flag, help='Use Gaussian similarity formulation')
parser.add_argument('--use_projected', default=True, type=utils.bool_flag, help='Include projected distance similarity')
parser.add_argument('--use_additive', default=False, type=utils.bool_flag, help='Use additive similarity')
parser.add_argument('--projected_power', default=1.0, type=float, help='Exponent for projected similarity decay')
parser.add_argument('--residue_power', default=1.0, type=float, help='Exponent for residue similarity decay')
parser.add_argument('--view', default=2, type=int, help='Augmentation view multiplier')
parser.add_argument('--delta', default=1.0, type=float, help='Delta value for proxy loss')
parser.add_argument('--sigma', default=1.0, type=float, help='Sigma value for proxy loss')
parser.add_argument('--only_proxy', default=False, type=utils.bool_flag, help='Train with only proxy loss')
parser.add_argument('--no_proxy', default=False, type=utils.bool_flag, help='Disable proxy loss')
# Miscellaneous
parser.add_argument('--load_debug', default=False, type=utils.bool_flag, help='Load debug checkpoint')
parser.add_argument('--swav', default=False, type=utils.bool_flag, help='Use SwAV pretrained model')
parser.add_argument('--random_sampler', default=0, type=int, help='Use random sampler')
parser.add_argument('--remark', default='', help='Remark string')
parser.add_argument('--seed', default=None, type=int, help='Seed for training')

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('Seeding training. This may slow down training due to CUDNN deterministic setting!')

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Setup logging directories
model_name = f'proxy_UDML_{args.dataset}_{args.model}_emb{args.embedding_size}_{args.optimizer}_lr{args.lr}_batch{args.sz_batch}'
LOG_DIR = os.path.join(args.LOG_DIR, 'logs', args.dataset, model_name)
os.makedirs(LOG_DIR, exist_ok=True)
DATA_DIR = os.path.abspath(args.DATA_DIR)

# -------------------------- Data Loading -------------------------- #
is_inception = (args.model.lower().find('googlenet')+1 or args.model.lower().find('inception')+1 or args.model.lower().find('bn_inception')+1)
# Load sampling data (used for constructing neighbor-based batches)
dataset_sampling = dataset.load(name=args.dataset, root=DATA_DIR, mode='train',
                                  transform=dataset.utils.Transform_for_Sampler(is_train=False, is_inception=is_inception))
dl_sampling = torch.utils.data.DataLoader(dataset_sampling,
    batch_size=args.sz_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

# Training dataset with multi-view augmentations
trn_dataset = dataset.load(name=args.dataset, root=DATA_DIR, mode='train',
    transform=dataset.utils.MultiTransforms(is_train=True, is_inception=is_inception, view=args.view))
# Use balanced sampler if not using random sampling
if args.random_sampler == 1:
    dl_tr = torch.utils.data.DataLoader(trn_dataset, batch_size=args.sz_batch,
                                         shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
else:
    balanced_sampler = sampler.NNBatchSampler(trn_dataset, None, dl_sampling, args.sz_batch, args.num_neighbors, True)
    dl_tr = torch.utils.data.DataLoader(trn_dataset, num_workers=args.workers, pin_memory=True, batch_sampler=balanced_sampler)

# Evaluation dataset
ev_dataset = dataset.load(name=args.dataset, root=DATA_DIR, mode='eval',
                          transform=dataset.utils.make_transform(is_train=False, is_inception=is_inception))
dl_ev = torch.utils.data.DataLoader(ev_dataset, batch_size=args.sz_batch, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

# ----------------------- Model Architecture ----------------------- #
# Instantiate student and teacher models based on args.model
if 'googlenet' in args.model.lower():
    model_student = inception_v1(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True)
    model_teacher = inception_v1(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False)
elif 'bn_inception' in args.model.lower():
    model_student = bn_inception(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True, bn_freeze=args.bn_freeze)
    model_teacher = bn_inception(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False, bn_freeze=args.bn_freeze)
elif 'resnet18' in args.model.lower():
    model_student = Resnet18(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True, bn_freeze=args.bn_freeze)
    model_teacher = Resnet18(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False, bn_freeze=args.bn_freeze)
elif 'resnet50' in args.model.lower():
    model_student = Resnet50(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True, bn_freeze=args.bn_freeze, swav_pretrained=args.swav)
    model_teacher = Resnet50(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False, bn_freeze=args.bn_freeze, swav_pretrained=args.swav)
else:
    raise ValueError("Unknown model architecture")

model_student = model_student.cuda()
model_teacher = model_teacher.cuda()

# Freeze teacher parameters
for param in model_teacher.parameters():
    param.requires_grad = False

if args.gpu_id == -1:
    model_student = nn.DataParallel(model_student)
    model_teacher = nn.DataParallel(model_teacher)

# --------------------- Proxy Loss Initialization --------------------- #
# Initialize proxy loss (neighbor_proj_loss) as in main_proxy.py
stml_criterion = loss_proxy.neighbor_proj_loss(
    args, 
    sigma=args.sigma,
    delta=args.delta,
    view=args.view,
    disable_mu=args.student_norm,
    topk=args.num_neighbors * args.view
).cuda()

# Initialize momentum update for teacher-student
momentum_update = loss_proxy.Momentum_Update(momentum=args.momentum).cuda()

# -------------------- Optimizer Configuration -------------------- #
# Separate embedding parameters (student embedding_f) from the rest
if args.gpu_id != -1:
    embedding_param = list(model_student.model.embedding_f.parameters())
else:
    embedding_param = list(model_student.module.model.embedding_f.parameters())

# Parameter groups: one for the rest of the model, one for the embedding, and one for proxy parameters
param_groups = [
    {'params': list(set(model_student.parameters()).difference(set(embedding_param)))},
    {'params': embedding_param, 'lr': args.emb_lr, 'weight_decay': float(args.weight_decay)},
]
# Include proxy loss parameters (for example, proxies_f, proxies_g, and one set of proxy_planes)
proxy_params = [stml_criterion.proxies_f, stml_criterion.proxies_g, stml_criterion.proxy_planes_g]
param_groups.append({'params': proxy_params, 'lr': args.lr * args.proxy_lr_mult})

if args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
elif args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = optim.RMSprop(param_groups, lr=args.lr, alpha=0.9, weight_decay=args.weight_decay, momentum=0.9)
elif args.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError("Unknown optimizer type")

if not args.fix_lr:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs)
else:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.nb_epochs)

print("Training parameters:", vars(args))
print(f"Training for {args.nb_epochs} epochs.")

# -------------------------- Training Loop -------------------------- #
losses_list = []
best_recall = [0]
best_epoch = 0
iteration = 0
start_time = time.time()

for epoch in range(args.nb_epochs):
    # If using a balanced sampler, reinitialize it every epoch
    if args.random_sampler == 1:
        dl_tr = torch.utils.data.DataLoader(trn_dataset, batch_size=args.sz_batch, shuffle=True,
                                              num_workers=args.workers, drop_last=True, pin_memory=True)
    else:
        balanced_sampler = sampler.NNBatchSampler(trn_dataset, model_student, dl_sampling, args.sz_batch, args.num_neighbors, True)
        dl_tr = torch.utils.data.DataLoader(trn_dataset, num_workers=args.workers, pin_memory=True, batch_sampler=balanced_sampler)
    
    model_student.train()
    model_teacher.eval()  # Teacher remains in eval mode

    losses_per_epoch = []
    pbar = tqdm(enumerate(dl_tr), total=len(dl_tr))
    for batch_idx, data in pbar:
        # Assume data returns (images, labels, indices)
        x, y, idx = data
        y = y.squeeze().cuda(non_blocking=True)
        idx = idx.squeeze().cuda(non_blocking=True)
        # If using multiple views, duplicate labels/indices accordingly
        y = torch.cat([y] * args.view)
        idx = torch.cat([idx] * args.view)
        x = torch.cat(x, dim=0).cuda(non_blocking=True)
        
        # Forward passes: student and teacher
        # Student returns a tuple (x_g, x_f); we use x_f (the embedding)
        s_out = model_student(x)
        # For teacher, no gradient is needed
        with torch.no_grad():
            t_out = model_teacher(x)
        
        # stml_criterion expects: student embedding, teacher embedding, idx, labels, and epoch
        loss_dict = stml_criterion(s_out, t_out, idx, y, epoch)
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        momentum_update(model_student, model_teacher)
        losses_per_epoch.append(loss.item())
        # Pop other loss components (if needed for display)
        rc_loss = loss_dict.get('RC', torch.tensor(0.0)).item()
        proxy_loss_val = loss_dict.get('proxy', torch.tensor(0.0)).item()

        pbar.set_description(f"Epoch {epoch} [{batch_idx+1}/{len(dl_tr)}] Loss: {loss.item():.4f} RC: {rc_loss:.4f} Proxy: {proxy_loss_val:.4f}")
        iteration += 1

    scheduler.step()
    epoch_loss = np.mean(losses_per_epoch)
    losses_list.append(epoch_loss)
    print(f"\nEpoch {epoch} average loss: {epoch_loss:.4f}")

    # ------------------- Evaluation ------------------- #
    with torch.no_grad():
        print("\n**Evaluating...**")
        if args.dataset.lower() != 'sop':
            k_list = [1, 2, 4, 8]
            Recalls = utils.evaluate_euclid(model_student, dl_ev, k_list)
        else:
            k_list = [1, 10, 100, 1000]
            Recalls = utils.evaluate_euclid(model_student, dl_ev, k_list)
        print("Recalls:", Recalls)

    if Recalls[0] > best_recall[0]:
        best_recall = Recalls
        best_epoch = epoch
        print(f"Achieved best performance at epoch {epoch}! Best Recall@1: {best_recall[0]:.4f}")
        # Save best checkpoint
        best_ckpt = {
            'model_state_dict': model_student.state_dict() if args.gpu_id != -1 else model_student.module.state_dict(),
            'epoch': epoch,
            'recall': best_recall
        }
        torch.save(best_ckpt, os.path.join(LOG_DIR, "best.pth"))

    # Optionally save all checkpoints
    if args.load_debug or args.save:
        ckpt = {
            'model_state_dict': model_student.state_dict() if args.gpu_id != -1 else model_student.module.state_dict(),
            'epoch': epoch,
            'recall': Recalls
        }
        torch.save(ckpt, os.path.join(LOG_DIR, f"checkpoint_{epoch}.pth"))

    print(f"Epoch {epoch} complete. Best Recall@1 so far: {best_recall[0]:.4f}")

total_time = time.time() - start_time
print(f"Training complete in {total_time:.2f} seconds over {args.nb_epochs} epochs.")
print(f"Best epoch: {best_epoch}, Best Recall@1: {best_recall[0]:.4f}")
