from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from loss_proxy import Momentum_Update
import time

################################################################
# Original TENT code (unchanged)                               
################################################################

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        #assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, teacher=None):
    """Forward and adapt model on batch of data (original TENT).

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)

    if isinstance(outputs, tuple):
        logits, _ = outputs
    else:
        logits = outputs

    # adapt using TENT's entropy
    loss = softmax_entropy(logits).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


################################################################
# New Proxy-Loss Code                                          
################################################################

@torch.enable_grad()
def forward_and_adapt_proxy(x, student_model, teacher_model, optimizer, momentum_updater, proxy_loss_fn, epoch: int) -> torch.Tensor:
    """
    Forward pass using student & teacher models and adapt the student model
    using the proxy loss (instead of TENT's entropy).

    Args:
      x (Tensor): input batch.
      student_model (nn.Module): model that should be updated at test time.
      teacher_model (nn.Module): teacher model (usually fixed or momentum-updated).
      optimizer (Optimizer): updates BN (and proxy) parameters.
      proxy_loss_fn (nn.Module): an instance of neighbor_proj_loss.
      epoch (int): epoch-like variable if needed by proxy_loss_fn.

    Returns:
      Tensor: the student model's output embedding for the batch.
    """
    start_time = time.time()
    #] => Entering forward_and_adapt_proxy()")

    # Student forward.
    student_out = student_model(x)

    if isinstance(student_out, tuple):
        student_logits, student_features = student_out
    else:
        print("Not getting student tuple out of resnet")

    

    # Teacher forward (no grad).
    with torch.no_grad():
        teacher_out = teacher_model(x)
        if isinstance(teacher_out, tuple):
            teacher_logits, teacher_features = teacher_out
        else:
            print("Not getting teacher tuple out of resnet")



    # Compute the proxy loss (note: idx and y are no longer used).
    loss_dict = proxy_loss_fn(student_features, teacher_features, epoch)
    loss = loss_dict['loss']

    # Backpropagation & update.
    optimizer.zero_grad()
    
    loss.backward()

    
    optimizer.step()

    momentum_updater(student_model, teacher_model)

    optimizer.zero_grad()


    loss_value = loss.item()

    elapsed = time.time() - start_time
    #print(f"[DEBUG] forward_and_adapt_proxy() => loss={loss.item():.4f}, took {elapsed:.3f}s")


    return student_out, loss_value


class TentProxy(nn.Module):
    """
    TentProxy adapts a model by proxy loss during testing,
    similar to TENT but uses forward_and_adapt_proxy instead of entropy minimization.

    This class parallels the Tent class, preserving episodic resets if needed.
    By default, it expects that you have already configured the optimizer
    to update only BN parameters (and optionally proxy parameters).
    """

    def __init__(self, student_model, teacher_model, proxy_loss_fn, optimizer, momentum_updater,
                 steps=1, episodic=False, epoch=0):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.proxy_loss_fn = proxy_loss_fn
        self.optimizer = optimizer
        self.momentum_updater = momentum_updater
        self.steps = steps
        #assert steps > 0, "TentProxy requires >= 1 step(s)"
        self.episodic = episodic
        self.epoch = epoch  # epoch-like variable if needed by proxy_loss_fn
        self.loss_history = []

        # Copy initial states so we can reset if episodic is True.
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.student_model, self.optimizer)
        
        #teacher intial state 
        self.teacher_state = deepcopy(self.teacher_model.state_dict())

    def forward(self, x) -> torch.Tensor:

        if self.steps == 0:
            return self.student_model(x)
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, loss_value = forward_and_adapt_proxy(
                x=x,
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                optimizer=self.optimizer,
                momentum_updater = self.momentum_updater,
                proxy_loss_fn=self.proxy_loss_fn,
                epoch=self.epoch
            )
            self.loss_history.append(loss_value)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.student_model, self.optimizer,
                                 self.model_state, self.optimizer_state)

        # Reset the teacher model to its initial state
        self.teacher_model.load_state_dict(self.teacher_state)


################################################################
# Helper functions (unchanged)                                  
################################################################

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent (or tent-proxy).

    - train mode
    - disable grad for entire model
    - re-enable grad for BN parameters
    - force batch stats usage
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatibility with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: check which require grad"
    assert not has_all_params, ("tent should not update all params: "
                                "check which require grad")
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"