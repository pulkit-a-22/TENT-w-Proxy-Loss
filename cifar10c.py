import logging

import torch
import torch.optim as optim

# Original robustbench & TENT imports (unchanged)
from robustbench.data import load_cifar10c
from robustbench.data import load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
from tent import Tent, TentProxy
from tent import forward_and_adapt_proxy
import norm

from conf import cfg, load_cfg_fom_args

from loss_proxy import Momentum_Update

import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import time
from proxy_net.resnet import Resnet18
from contextlib import redirect_stdout
from custom_sampler import NNBatchSampler  # import the custom sampler defined above
from torch.utils.data import TensorDataset, DataLoader
from cifar10c_custom import CIFAR10_C, test_transforms, train_transforms


import os



logger = logging.getLogger(__name__)

def my_custom_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # Predicted class = argmax of logits
    _, predicted = logits.max(dim=1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)

def evaluate(description):
    load_cfg_fom_args(description)
    total_eval_start = time.time()

    corruption_type = "gaussian_noise"
    severity = 5

    # Setup model based on adaptation mode.
    if cfg.MODEL.ADAPTATION == "tent_proxy":
        logger.info("test-time adaptation: TENT-PROXY")
        base_model = Resnet18(
            embedding_size=cfg.PROXY.EMBEDDING_SIZE,
            bg_embedding_size=cfg.PROXY.BG_EMBEDDING_SIZE,
            pretrained=False,
            is_norm=True,
            is_student=True,
            bn_freeze=False
        ).cuda()
        checkpoint = torch.load("cifar10_pretrained_resnet18.pth")
        base_model.load_state_dict(checkpoint, strict=True)
        model = setup_tent_proxy(base_model)  # This now returns a TentProxy with a reset() method.
    else:
        # Implement other branches as needed.
        raise ValueError("This testing script is configured for tent_proxy only.")
    

    # -------------------- Load Full CIFAR10-C Dataset --------------------
    cache_dir = os.path.join(cfg.SAVE_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    full_cache_file = os.path.join(cache_dir, f"cached_cifar10c_{corruption_type}_{severity}_full.pt")

    if os.path.exists(full_cache_file):
        logger.info(f"Loading cached full dataset from {full_cache_file}")
        cached_data = torch.load(full_cache_file)
        full_dataset = cached_data["dataset"]
    else:
        logger.info("Creating new full CIFAR10-C dataset instance...")
        full_dataset = CIFAR10_C(
            root=cfg.DATA_DIR,
            corruption=corruption_type,
            level=severity,
            transform=test_transforms   # Use test transforms as default here
        )
        logger.info(f"Full dataset created with {len(full_dataset)} samples. Caching to {full_cache_file}.")
        torch.save({"dataset": full_dataset}, full_cache_file)

    # -------------------- Split Dataset into Adaptation and Evaluation --------------------
    from torch.utils.data import random_split
    n_total = len(full_dataset)
    n_train = n_total // 2  # e.g., 50/50 split (you can change this ratio as needed)
    adapt_dataset, eval_dataset = random_split(full_dataset, [n_train, n_total - n_train])
    logger.info(f"Adaptation dataset: {len(adapt_dataset)} samples; Evaluation dataset: {len(eval_dataset)} samples.")

    # Create a default loader for feature extraction from the adaptation dataset.
    default_loader = DataLoader(adapt_dataset, batch_size=200, shuffle=False, num_workers=4)
    model_for_sampling = model.student_model if hasattr(model, "student_model") else model

    # Setup the NNBatchSampler using the adaptation dataset.
    nn_sampler = NNBatchSampler(
        data_source=adapt_dataset,
        model=model_for_sampling,
        seen_dataloader=default_loader,
        batch_size=200,      # Total images per batch (must be divisible by nn_per_image)
        nn_per_image=10,     # Query image and its 10 nearest neighbors.
        using_feat=True,
        is_norm=False        # Change if your model already normalizes features.
    )
    train_loader = DataLoader(adapt_dataset, batch_sampler=nn_sampler, num_workers=8)

    # Adapt the model over 10 epochs (adapt on the whole dataset).
    #REVIEWCHANGE
    #The adaptation loop now runs over 10 epochs covering the entire dataset instead of processing only a single pass.
    #At the beginning of each epoch, the model is “reset” (using the reset method from TentProxy) to ensure continuity and consistency.
    num_epochs = 10
    for epoch in range(num_epochs):
        logger.info(f"Adaptation epoch {epoch+1}/{num_epochs}")
        for images, _ in train_loader:
            images = images.cuda()
            _, loss_value = forward_and_adapt_proxy(
                images,
                model.student_model,
                model.teacher_model,
                model.optimizer,
                model.momentum_updater,
                model.proxy_loss_fn,
                epoch
            )
            logger.debug(f"Adaptation loss: {loss_value:.4f}")
        logger.info("Completed adaptation epoch.")

    # -------------------- Evaluation on the Evaluation Dataset --------------------
    logger.info("Building full evaluation tensors from evaluation dataset...")
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=8)
    all_imgs = []
    all_labels = []
    from tqdm import tqdm
    for batch_idx, (images, labels) in enumerate(tqdm(eval_loader, desc="Evaluating dataset")):
        print(f"Processing evaluation batch {batch_idx+1}/{len(eval_loader)}")
        all_imgs.append(images)
        all_labels.append(labels)
    x_full = torch.cat(all_imgs, dim=0).cuda()
    y_full = torch.cat(all_labels, dim=0).cuda()

    # Evaluate accuracy.
    acc = accuracy(lambda x: model(x)[0] if isinstance(model(x), tuple) else model(x),
                   x_full, y_full, cfg.TEST.BATCH_SIZE)
    logger.info(f"Post-adaptation accuracy for {corruption_type} severity {severity}: {acc*100:.2f}%")
        
    total_eval_time = time.time() - total_eval_start
    logger.info(f"Overall evaluation took {total_eval_time:.3f}s")


############################################################
# Original "source" & "norm" & "tent" setup (unchanged)    
############################################################

def setup_source(model):
    """No adaptation baseline."""
    model.eval()
    logger.info(f"model for evaluation: {model}")
    return model

def setup_norm(model):
    """Test-time feature normalization (no gradient)."""
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: {model}")
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: {stat_names}")
    return norm_model

def setup_tent(model):
    """TENT adaptation with BN param updates + entropy loss."""
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(
        model, optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC
    )
    logger.info(f"model for adaptation: {model}")
    logger.info(f"params for adaptation: {param_names}")
    logger.info(f"optimizer for adaptation: {optimizer}")
    return tent_model

############################################################
# New "setup_tent_proxy" for TENT + Proxy                  
############################################################
def setup_tent_proxy(model):
    """
    TENT-PROXY adaptation:
     - Uses a proxy-model (from your proxy repository).
     - Adapts BN parameters (and optionally proxy parameters).
     - Uses a teacher copy + a proxy loss function that now expects only (s_f, t_f, epoch).
    """
    import copy
    from loss_proxy import neighbor_proj_loss

    # 1) Configure BN for TENT-style updates.
    model = tent.configure_model(model)

    # 2) Gather BN parameters.
    params, param_names = tent.collect_params(model)

    dummy_args = type('dummy_args', (), {
        'embedding_size': 512,
        'bg_embedding_size': 1024,
        #stop using proxies 
        'num_proxies': cfg.PROXY.NUM_PROXIES,
        'num_dims': cfg.PROXY.NUM_DIMS,
        'num_neighbors': 20,
        'projected_power': 0.0,
        'residue_power': 3.0,
        'use_gaussian_sim': False,
        'use_projected': True,
        'use_additive': False,
        'proxy_norm': True,
        'num_local': 10,
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

    #added
    proxy_params = list(proxy_loss_fn.parameters())
    proxy_param_names = [f"proxy_loss_fn.{n}" for n, p in proxy_loss_fn.named_parameters()]

    all_params = params + proxy_params
    all_param_names = param_names + proxy_param_names

    # 3) Create an optimizer.
    #student
    #changed to all params
    optimizer = setup_optimizer(all_params)

    #teacher
    #Question - what should momentum updated be?
    momentum_updater = Momentum_Update(0.999)

    # 4) Build a teacher model as a deep copy.
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    teacher_model.requires_grad_(False)


    # 6) Wrap everything in TentProxy.
    tent_proxy_model = tent.TentProxy(
        student_model=model,
        teacher_model=teacher_model,
        proxy_loss_fn=proxy_loss_fn,
        optimizer=optimizer,
        momentum_updater = momentum_updater,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        epoch=0
    )
    # logger.info(f"model for adaptation (tent_proxy): {model}")
    # logger.info(f"params for adaptation: {param_names}")
    # logger.info(f"optimizer for adaptation: {optimizer}")
    return tent_proxy_model

def setup_optimizer(params):
    """Common TENT/TENT-PROXY optimizer setup."""
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD
        ) 
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(
            params,
            lr=cfg.OPTIM.LR,
            momentum=cfg.OPTIM.MOMENTUM,
            dampening=cfg.OPTIM.DAMPENING,
            weight_decay=cfg.OPTIM.WD,
            nesterov=cfg.OPTIM.NESTEROV
        )
    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation."')