# Filename: run_experiments.py

import sys
import logging

# Import your config and evaluate from cifar10c
from conf import cfg, load_cfg_fom_args
from cifar10c import evaluate
from contextlib import redirect_stdout

logger = logging.getLogger(__name__)

def run_experiments(description):
    """
    Loops over dims in [3,5] and lr in [2e-5, 5*2e-5], updates config,
    then calls 'evaluate(...)' from your cifar10c.py.
    """
    load_cfg_fom_args(description)  # parse --cfg etc. from sys.argv

    dims_list = [3, 5]
    lr_list = [2e-5 * 5]  # => 2e-5, 1e-4

    logger.info(f"[run_experiments] Starting loops for dims={dims_list}, lr={lr_list}")

    for dims in dims_list:
        for lr in lr_list:
            # 1) Defrost the config so we can modify fields
            cfg.defrost()

            # 2) Set the desired dims & LR
            cfg.PROXY.NUM_DIMS = dims
            cfg.OPTIM.LR = lr

            # 3) Freeze again
            cfg.freeze()

            logger.info(f"=== Running evaluate(...) with dims={dims}, lr={lr} ===")

            # 4) Call your evaluate function from cifar10c.py
            evaluate(f"CIFAR-10-C evaluation [dims={dims}, lr={lr}]")

    logger.info("[run_experiments] All experiments done.")


if __name__ == "__main__":
    # Pass command-line arguments to run_experiments
    # e.g., python run_experiments.py --cfg cfgs/tent_proxy.yaml
    run_experiments("Run experiments loop.")
