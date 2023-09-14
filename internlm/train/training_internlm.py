#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
from functools import partial
from typing import Callable, Iterable, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.random import set_mode
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.trainer import TrainState
from internlm.data.batch_sampler import StaticBatchSampler, get_dpsampler_dataloader
from internlm.data.collaters import jsonl_ds_collate_fn, packed_collate_fn
from internlm.data.dataset import get_dataset_dict
from internlm.data.dummy_dataset import RandomDataset
from internlm.data.packed_dataset import (
    PackedDataset,
    PackedDatasetWithoutCuSeqlen,
    get_packed_dataset_without_short_length,
)
from internlm.data.utils import DATASET_TYPE_IDS_MAP, unpack_data
from internlm.model.moe import create_moe_param_groups
from internlm.monitor import send_heartbeat, set_env_var
from internlm.monitor.monitor import monitor_manager as mm
from internlm.solver.beta2_scheduler import Beta2Scheduler
from internlm.solver.lr_scheduler import FineTuneCosineAnnealingWarmupLR
from internlm.solver.optimizer import HybridZeroOptimizer
from internlm.solver.optimizer.utils import ParamBcastSyncHandler
from internlm.utils.common import DummyProfile
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.parallel import (
    is_no_pp_or_last_stage,
    sync_model_param,
    sync_model_param_within_tp,
)
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.utils.timeout import llm_timeout

logger = get_logger(__file__)


@llm_timeout(func_name="initialize_model")
def initialize_model():
    """
    Initialize model with Automatic Mixed Precision.

    Returns:
        torch.nn.Module:
            The neural network model to be trained or evaluated.
    """

    model = MODEL_INITIALIZER.get_module(module_name=gpc.config.model_type)(**(gpc.config.model))
    if isinstance(model, nn.ModuleList):
        model = nn.ModuleList(
            [
                NaiveAMPModel(
                    model=_m,
                    output_to_fp32=False,  # manually controlled by interleaved pipleline scheduler
                    dtype=gpc.config.model.get("dtype", torch.half),
                    sync_buffer=False,
                )
                for _m in model
            ]
        )
    else:
        model = NaiveAMPModel(
            model=model,
            output_to_fp32=is_no_pp_or_last_stage(),
            dtype=gpc.config.model.get("dtype", torch.half),
            sync_buffer=False,
        )

    # This sync is very important, cause the model weights kept in optimizer are copied
    # from the origin parameters in the memory, so we should make sure the dp sync
    # does not influence the model weights in optimizer be different with the origin parameters.
    sync_model_param(model, parallel_mode=ParallelMode.DATA)

    # This function is needed to make sure parameters that are not splitted by tensor parallelism are
    # the same across tensor parallelism.
    sync_model_param_within_tp(model)

    # Change random state mode to ParallelMode.DATA after model is built, guaranteeing the random
    # state in the same dp group are all the same.
    set_mode(ParallelMode.DATA)

    def wrapper(name):  # pylint: disable=W0613
        def hook_backward_function(module, module_input_grad, module_output_gard):  # pylint: disable=W0613
            from internlm.core.context.parallel_context import global_context as gpc

            # print("hook!!!",flush=True)
            with torch.no_grad():
                for test_tensor in module_input_grad:
                    if test_tensor is None:
                        continue
                    test_tensor = test_tensor.contiguous()
                    gathered_tensors = [
                        torch.zeros_like(test_tensor) for _ in range(gpc.get_world_size(ParallelMode.TENSOR))
                    ]
                    torch.distributed.all_gather(
                        gathered_tensors, test_tensor, group=gpc.get_group(ParallelMode.TENSOR)
                    )
                    all_equal = all(
                        tensor.eq(gathered_tensors[0]).all() for tensor in gathered_tensors
                    )  # pylint: disable=R1729

                    if not all_equal:
                        print(name, flush=True)
                        assert False

                for test_tensor in module_output_gard:
                    if test_tensor is None:
                        continue
                    test_tensor = test_tensor.contiguous()
                    gathered_tensors = [
                        torch.zeros_like(test_tensor) for _ in range(gpc.get_world_size(ParallelMode.TENSOR))
                    ]
                    torch.distributed.all_gather(
                        gathered_tensors, test_tensor, group=gpc.get_group(ParallelMode.TENSOR)
                    )
                    all_equal = all(
                        tensor.eq(gathered_tensors[0]).all() for tensor in gathered_tensors
                    )  # pylint: disable=R1729

                    if not all_equal:
                        print(name, flush=True)
                        assert False

        return hook_backward_function

    def grad_hook(name):
        def hook_func(grad):
            with torch.no_grad():
                test_tensor = grad
                gathered_tensors = [
                    torch.zeros_like(test_tensor) for _ in range(gpc.get_world_size(ParallelMode.TENSOR))
                ]
                torch.distributed.all_gather(gathered_tensors, test_tensor, group=gpc.get_group(ParallelMode.TENSOR))
                all_equal = all(
                    tensor.eq(gathered_tensors[0]).all() for tensor in gathered_tensors
                )  # pylint: disable=R1729

                if not all_equal:
                    print(name, flush=True)
                    assert False

        return hook_func

    for name, module in model.model.blocks.named_modules():
        if "gate" in name:
            module.register_full_backward_hook(wrapper(name))
            for name, param in module.named_parameters():
                param.register_hook(grad_hook(name))

    return model


@llm_timeout(func_name="initialize_optimizer")
def initialize_optimizer(model: Union[nn.Module, nn.ModuleList]):
    """
    Initialize optimizer.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        A tuple of (optimizer, beta2_scheduler, lr_scheduler).
    """
    if gpc.config.hybrid_zero_optimizer.overlap_sync_param:
        param_bcast_sync_handler = ParamBcastSyncHandler(model)
    else:
        param_bcast_sync_handler = None

    adam_cfg = gpc.config.adam
    # split the moe parameters into different groups
    if gpc.config.model.num_experts > 1:
        params = create_moe_param_groups(model, adam_cfg.weight_decay)
    else:
        params = [{"params": model.parameters(), "weight_decay": adam_cfg.weight_decay}]
    print((len(params)), "==================================", flush=True)
    naive_optimizer = torch.optim.AdamW(
        params=params,
        lr=adam_cfg.lr,
        betas=(adam_cfg.adam_beta1, adam_cfg.adam_beta2),
        eps=adam_cfg.adam_eps,
    )
    optimizer = HybridZeroOptimizer(
        naive_optimizer,
        grad_scal_cfg=gpc.config.grad_scaler,
        zero_cfg=gpc.config.hybrid_zero_optimizer,
        param_bcast_sync_handler=param_bcast_sync_handler,
    )

    beta2_scheduler = Beta2Scheduler(optimizer=naive_optimizer, **gpc.config.beta2_scheduler)

    lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, **gpc.config.lr_scheduler)

    return optimizer, beta2_scheduler, lr_scheduler


@llm_timeout(func_name="get_train_data_loader")
def get_train_data_loader(
    num_worker: int = 0, dataset_generate_func: Callable = None, train_sampler=None, train_collate_fn=None
):
    """
    Generate and return the training data loader.

    Args:
        num_worker (:class:`int`): number of subprocesses used for dataloader.
        dataset_generate_func (:class:`Callable`, optional): generate function for dataset.
        train_sampler (:class:`torch.utils.data.sampler`, optional): dataset sampler for training dataloader.
        train_collate_fn (:class:`Callable`, optional): collate function for training dataloader.

    Returns:
        A tuple of (train_dl, dataset_types).
    """

    # Get the dataset types
    dataset_types = None
    dataset_types = list(DATASET_TYPE_IDS_MAP.keys())
    data_cfg = gpc.config.data

    # Get the sample weight dictionary
    train_folder = data_cfg.train_folder

    if not train_folder:
        train_ds = RandomDataset(num_samples=1000000, max_len=data_cfg.seq_len)
        if data_cfg.pack_sample_into_one:
            train_ds = PackedDatasetWithoutCuSeqlen(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
        else:
            train_ds = PackedDataset(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
    else:
        if dataset_generate_func is not None:
            train_ds = dataset_generate_func()
        else:
            train_ds = get_packed_dataset_without_short_length(
                folder=data_cfg.train_folder,
                packed_length=data_cfg.packed_length,
                max_length_per_sample=data_cfg.seq_len,
                show_progress=dist.get_rank() == 0,
                min_length=data_cfg.min_length,
                min_length_dict=data_cfg.get("min_length_dict", {}),
                pack_into_one_sample=data_cfg.pack_sample_into_one,
            )

    if dataset_generate_func is None or not train_folder:
        # partition already completed
        assert isinstance(train_ds, (PackedDataset, PackedDatasetWithoutCuSeqlen, ConcatDataset))
        # Create the training dataset sampler
        train_sampler = StaticBatchSampler(
            train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
            batch_size=data_cfg.micro_num,
            rampup_batch_size=data_cfg.rampup_batch_size,
            micro_bsz=data_cfg.micro_bsz,
            seed=1024,
            drop_last=True,
            data_rank=gpc.get_local_rank(ParallelMode.DATA),
            data_world_size=gpc.get_world_size(ParallelMode.DATA),
        )

    if dataset_generate_func is None or not train_folder:
        train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    # Create the training data loader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=num_worker > 0,
    )

    return train_dl, dataset_types


@llm_timeout(func_name="get_validation_data_loader")
def get_validation_data_loader(
    num_worker: int = 0, dataset_generate_func: Callable = None, val_collate_fn=None, dataloader_func=None
):
    """Generate and return the validation data loader."""

    data_cfg = gpc.config.data

    if not data_cfg.valid_folder:
        val_ds = RandomDataset(num_samples=gpc.get_world_size(ParallelMode.DATA) * 500, max_len=data_cfg.seq_len)
    else:
        if dataset_generate_func is not None:
            assert val_collate_fn and dataloader_func is not None
            val_ds = dataset_generate_func()
        else:
            val_ds = get_dataset_dict(folder=data_cfg.valid_folder, split="")

    if not isinstance(val_ds, dict):
        val_ds = {"val": val_ds}

    if val_collate_fn is None or not data_cfg.valid_folder:
        val_collate_fn = partial(jsonl_ds_collate_fn, max_length_per_sample=data_cfg.seq_len)

    val_dls = {}
    for val_name, ds in val_ds.items():
        if dataloader_func and data_cfg.valid_folder is not None:
            val_dls[val_name] = dataloader_func(dataset=ds, collate_fn=val_collate_fn)
            if gpc.is_rank_for_log():
                logger.info(
                    f"load validation dataset {val_name} with valid batch size {str(data_cfg.valid_micro_num)} and "
                    f"{ds.size} Byte samples."
                )
        else:
            # making the batch_size of validate larger can speed up the evaluation, but it should not be too large,
            # otherwise too much data may be dropped
            batch_size = min(
                data_cfg.valid_micro_num * data_cfg.micro_bsz, len(ds) // gpc.get_world_size(ParallelMode.DATA)
            )
            batch_size = batch_size // data_cfg.micro_bsz * data_cfg.micro_bsz

            if batch_size == 0 and gpc.is_rank_for_log():
                logger.info(f"skip validate {val_name}.")
                continue

            val_dls[val_name] = get_dpsampler_dataloader(
                ds,
                shuffle=False,
                num_workers=num_worker,
                batch_size=batch_size,
                collate_fn=val_collate_fn,
                drop_last=True,
            )  # drop_last=True, otherwise it may cause problems in the last batch

            if gpc.is_rank_for_log():
                logger.info(
                    f"load validation dataset {val_name} with valid batch size {str(batch_size)} and "
                    f"samples {str(len(val_dls[val_name]))}."
                )

    return val_dls


@llm_timeout(func_name="load_new_batch")
def load_new_batch(train_dl: DataLoader, train_iter: Iterable, train_state: TrainState):
    """
    Load and return the new batch data based on training data loader.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader for training.
        train_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
        train_state (TrainState): Current training state.

    Returns: A batch data and the updated train_iter.
    """

    timer("batch-gen").start()
    try:
        batch = next(train_iter)  # structure is ({'input_ids': Tensor, 'cu_seqlens': Tensor}, Tensor)
        if hasattr(train_state, "batch_sampler_iter"):
            next(train_state.batch_sampler_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)
        train_state.num_consumed_samples_in_epoch = 0
        if hasattr(train_state, "batch_sampler"):
            train_state.batch_sampler_iter = iter(train_state.batch_sampler)
            next(train_state.batch_sampler_iter)
    timer("batch-gen").stop()

    if batch[0].get("type_ids", None) is not None:
        # if use_flash_attn is False, we need to unpack type_ids
        if not gpc.config.model.use_flash_attn:
            batch[0]["type_ids"] = unpack_data(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter


def initialize_llm_profile(profiling: bool = False, start_time: str = None):
    """Initialize and return the profiler context manager instance."""

    if profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        llm_profile = torch.profiler.profile
        logger.info(f"Do profiling in rank {gpc.get_global_rank()}!")
    else:
        llm_profile = DummyProfile

    return llm_profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"{gpc.config.JOB_NAME}/{start_time}/traces/rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"pp{gpc.get_local_rank(ParallelMode.PIPELINE)}",
        ),
        with_stack=True,
        with_modules=True,
    )


@llm_timeout(func_name="record_current_batch_training_metrics")
def record_current_batch_training_metrics(
    get_tflops_func,
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    trainer,
    start_time,
    loss,
    moe_loss,
    grad_norm,
    metric,
    update_panel,
):
    """
    Print some training metrics of current batch.
    """

    set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))

    timer.store_last_timers()
    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)
    if is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(trainer.engine.optimizer, "grad_scaler"):
            scaler = trainer.engine.optimizer.grad_scaler._scale.item()
        elif hasattr(trainer.engine.optimizer.optim, "grad_scaler"):
            scaler = trainer.engine.optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        num_samples_in_batch = sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        max_length_in_batch = max([(b[1:] - b[:-1]).max().item() for b in batch[0]["cu_seqlens"]])
        max_samples_in_batch = max([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        min_samples_in_batch = min([len(b) - 1 for b in batch[0]["cu_seqlens"]])

        tk_per_gpu = 0
        tk_per_gpu = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / (time.time() - start_time),
            2,
        )

        tflops = get_tflops_func((time.time() - start_time))

        infos = {
            "tflops": tflops,
            "step": batch_count,
            "loss": loss.item() - moe_loss.item(),
            "moe_loss": moe_loss.item(),
            "tgs (tokens/gpu/second)": tk_per_gpu,
            "lr": lr,
            "loss_scale": scaler,
            "grad_norm": grad_norm,
        }

        infos["micro_num"] = len(batch[1])
        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["num_samples_in_batch"] = num_samples_in_batch  # the number of batches which have the most samples
        infos["largest_length"] = max_length_in_batch  # the longest input
        infos["largest_batch"] = max_samples_in_batch  # the batch with the most samples
        infos["smallest_batch"] = min_samples_in_batch
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        line = ""
        for key, value in infos.items():
            line += f"{key}={value} "
            if isinstance(value, dict):
                writer.add_scalars(key=key, value=value, step=train_state.step_count)
            else:
                writer.add_scalar(key=key, value=value, step=train_state.step_count)

        if gpc.config.monitor.alert.get("light_monitor_address", None) and batch_count % 50 == 0:
            send_heartbeat("train_metrics", infos)

        if update_panel:
            # metrics shown with dashboard panels
            panel_metrics = {
                "step": batch_count,
                "lr": lr,
                "num_consumed_tokens": train_state.num_consumed_tokens,
                "loss": loss.item() - moe_loss.item(),
                "flops": tflops,
                "tgs": tk_per_gpu,
                "acc": acc_perplex["acc"],
                "perplexity": acc_perplex["perplexity"],
                "fwd_bwd_time": fwd_bwd_time,
            }
            for norm_key, norm_value in grad_norm.items():
                panel_metrics[norm_key] = norm_value

            logger.info(
                "{line}",
                line=line,
                extra=panel_metrics,
            )
        else:
            logger.info(line)

        # if loss spike occurs, send alert info to feishu
        mm.monitor_loss_spike(
            alert_address=gpc.config.monitor.alert.feishu_alert_address,
            step_count=batch_count,
            cur_step_loss=loss.item(),
        )
