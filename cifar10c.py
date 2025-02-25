import logging

import torch
import torch.optim as optim

# Original robustbench & TENT imports (unchanged)
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
from tent import Tent, TentProxy
from tent import forward_and_adapt_proxy
import norm

from conf import cfg, load_cfg_fom_args

from loss_proxy import Momentum_Update

logger = logging.getLogger(__name__)

def evaluate(description):
    load_cfg_fom_args(description)

    ############################################################
    # 1) If "tent": use the original code (RobustBench model)  
    ############################################################
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        base_model = load_model(
            cfg.MODEL.ARCH,
            cfg.CKPT_DIR,
            cfg.CORRUPTION.DATASET,
            ThreatModel.corruptions
        ).cuda()
        model = setup_tent(base_model)

    ############################################################
    # 2) If "source", "norm", or other original modes          
    ############################################################
    elif cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE (source)")
        base_model = load_model(
            cfg.MODEL.ARCH,
            cfg.CKPT_DIR,
            cfg.CORRUPTION.DATASET,
            ThreatModel.corruptions
        ).cuda()
        model = setup_source(base_model)

    elif cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        base_model = load_model(
            cfg.MODEL.ARCH,
            cfg.CKPT_DIR,
            cfg.CORRUPTION.DATASET,
            ThreatModel.corruptions
        ).cuda()
        model = setup_norm(base_model)

    ############################################################
    # 3) If "tent_proxy": use your proxy repository's model + 
    #    proxy loss, adapting BN parameters similar to TENT.
    ############################################################
    elif cfg.MODEL.ADAPTATION == "tent_proxy":
        logger.info("test-time adaptation: TENT-PROXY")
        # Use bn_inception from your proxy repository
        from proxy_net.resnet import Resnet18
        base_model = Resnet18(
            embedding_size=512,
            bg_embedding_size=1024,
            pretrained=True,   # or False, per your preference
            is_norm=True,
            is_student=True,
            bn_freeze=False
        ).cuda()
        model = setup_tent_proxy(base_model)

    else:
        raise ValueError(f"Unknown adaptation method: {cfg.MODEL.ADAPTATION}")

    # Evaluate on each severity & corruption type.
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            try:
                model.reset()
                logger.info("resetting model")
            except Exception as e:
                logger.warning("not resetting model: " + str(e))

            x_test, y_test = load_cifar10c(
                cfg.CORRUPTION.NUM_EX,
                severity,
                cfg.DATA_DIR,
                False,
                [corruption_type]
            )
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(lambda x: model(x)[0] if isinstance(model(x), tuple) else model(x), x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


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

    # 3) Create an optimizer.
    #student
    optimizer = setup_optimizer(params)

    #teacher
    #Question - what should momentum updated be?
    momentum_updater = Momentum_Update(0.999)

    # 4) Build a teacher model as a deep copy.
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # 5) Build your proxy loss function.
    dummy_args = type('dummy_args', (), {
        'embedding_size': 512,
        'bg_embedding_size': 512,
        'num_proxies': 10,
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
    logger.info(f"model for adaptation (tent_proxy): {model}")
    logger.info(f"params for adaptation: {param_names}")
    logger.info(f"optimizer for adaptation: {optimizer}")
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
