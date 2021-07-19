# coding=utf-8

import numpy as np
import pickle
import random
import os
import argparse
import pandas as pd
import sys

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from deepspeed.utils import RepeatingLoader

from apex.optimizers import FusedAdam as Adam
from megatron import print_rank_0, get_tensorboard_writer, get_timers, mpu, get_args
from megatron.initialize import initialize_megatron
from megatron.model import get_params_for_weight_decay_optimization
from megatron.learning_rates import AnnealingLR
from megatron.utils import report_memory, reduce_losses
from megatron.training import evaluate

from megatron_bart import MegatronBART
from tokenizer import load_tokenizer
from decoder import DecodeSampler
from csv_data import MoleculeDataLoader
from util import DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH, DEFAULT_MAX_SEQ_LEN, REGEX
from checkpointing import save_megatron_checkpoint, load_deepspeed_iteration

tokenizer = load_tokenizer(vocab_path=DEFAULT_VOCAB_PATH, chem_token_start=DEFAULT_CHEM_TOKEN_START, regex=REGEX)
num_batches_processed = 0
epochs = 0

def get_deepspeed_checkpoint_dir(save_dir):
    return os.path.join(*os.path.split(save_dir)[:-1], 'deepspeed')

class RepeatingLoader:

    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.
        Args:
            loader (iterator): The data loader to repeat.
        """

        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        global epochs
        global num_batches_processed
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
            if torch.distributed.get_rank() == 0:
                epochs += 1
                num_batches_processed = 0
        return batch


def build_model(args):

    VOCAB_SIZE = len(tokenizer)
    MAX_SEQ_LEN = 512
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    sampler = DecodeSampler(tokenizer, MAX_SEQ_LEN)

    model = MegatronBART(
        sampler,
        pad_token_idx,
        VOCAB_SIZE,
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.hidden_size * 4,
        MAX_SEQ_LEN,
        dropout=0.1,
        )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters()
                   if p.requires_grad)

    print_rank_0('Number of parameters in MegatronBART: '
                 + str(count_parameters(model)))
    return model


def get_optimizer(model, args):
    param_groups = get_params_for_weight_decay_optimization(model)
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False
    optimizer = AdamW(param_groups, lr=args.lr,
                      weight_decay=args.weight_decay,
                      betas=(args.adam_beta1, args.adam_beta2))
    return optimizer


def get_learning_rate_scheduler(optimizer, args):

    # Add linear learning rate scheduler.

    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup * args.train_iters,
        total_iters=args.train_iters,
        decay_style=args.lr_decay_style,
        min_lr=args.min_lr,
        last_iter=0,
        use_checkpoint_lr_scheduler=False,
        override_lr_scheduler=False,
        )

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = build_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    print_rank_0('DeepSpeed is enabled.')

    # (mpu if args.pipe_parallel_size == 0 else None)
    localrankmpi = int(os.getenv('LOCAL_RANK', '0'))
    rankmpi = int(os.getenv('RANK', '0'))
    args.rank = rankmpi
    args.local_rank = localrankmpi
    (model, optimizer, _, lr_scheduler) = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=(mpu if args.pipe_parallel_size == 0 else None),
        dist_init_required=False,
        )

    return (model, optimizer, lr_scheduler)


def get_batch(data_iterator):
    """Generate a batch"""

    global num_batches_processed
    keys = [
        'encoder_input',
        'encoder_pad_mask',
        'decoder_input',
        'decoder_pad_mask',
        'target',
        'target_pad_mask'
        ]
    datatype = torch.int64
    data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.

    encoder_tokens = data_b['encoder_input'].long()
    encoder_pad_mask = data_b['encoder_pad_mask'].bool()
    decoder_tokens = data_b['decoder_input'].long()
    decoder_pad_mask = data_b['decoder_pad_mask'].bool()
    target = data_b['target'].long()
    target_pad_mask = data_b['target_pad_mask'].long()
    num_batches_processed += 1

    return {
        'encoder_input': encoder_tokens,
        'encoder_pad_mask': encoder_pad_mask,
        'decoder_input': decoder_tokens,
        'decoder_pad_mask': decoder_pad_mask,
        'target': target,
        'target_pad_mask': target_pad_mask
        }


def forward_step(data_iterator, model):
    """Forward step."""

    timers = get_timers()

    # Get the batch.

    timers('batch generator').start()
    batch = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.

    tokens = batch['target']
    pad_mask = batch['target_pad_mask']
    outputs = model(batch)
    token_output = outputs['token_output']
    loss = model.module._calc_loss(batch, outputs)
    acc = model.module._calc_char_acc(batch, outputs)
    reduced_loss = reduce_losses([loss])
    
    return (loss, {'mask loss': reduced_loss[0], 'acc': acc})


def backward_step(optimizer, model, loss):
    """Backward step."""

    timers = get_timers()

    # Backward pass.

    timers('backward-backward').start()
    model.backward(loss)
    timers('backward-backward').stop()
    timers('backward-allreduce').reset()


def eval_step(data_iterator, model):
    """Forward step."""

    timers = get_timers()

    # Get the batch.

    timers('batch generator').start()
    batch = next(data_iterator)
    timers('batch generator').stop()

    # Forward model.

    val_outputs = model.module.validation_step(batch)
    invalid_smiles = val_outputs['val_invalid_smiles']
    val_loss = val_outputs['val_loss']
    token_acc = val_outputs['val_token_acc']
    val_perplexity= val_outputs['val_perplexity']
    val_molecular_accuracy= val_outputs['val_molecular_accuracy']
    # Reduce loss for logging.

    reduced_invalid_smiles = reduce_losses([invalid_smiles])
    
    return {'val_invalid_smiles': reduced_invalid_smiles[0], 'val_molecular_accuracy':val_molecular_accuracy}


def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,
    pipe_parallel_size,
    ):
    """Single training step."""

    timers = get_timers()

    # Forward model for one step.

    timers('forward').start()
    (loss, loss_reduced) = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.

    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.

    timers('optimizer').start()
    model.step()
    timers('optimizer').stop()

    return loss_reduced


def train(
    forward_step_func,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    trainloader,
    val_data_iterator,
    pipe_parallel_size,
    args,
    ):
    """Train the model function."""

    global num_batches_processed
    writer = get_tensorboard_writer()
    timers = get_timers()
    model.train()
    timers('interval time').start()
    report_memory_flag = True

    while args.iteration < args.train_iters:
        loss = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            lr_scheduler,
            pipe_parallel_size,
            )

        args.iteration += 1
        print_rank_0('Iteration: ' + str(args.iteration) + '/'
                     + str(args.train_iters) + ', Loss: '
                     + str(loss['mask loss'].item()) + ', Acc: '
                     + str(loss['acc']) + ', Num batches: '
                     + str(num_batches_processed) + '/'
                     + str(len(trainloader.loader)) + ', Epoch: '
                     + str(epochs))

        if torch.distributed.get_rank() == 0:
            writer.add_scalar('training mask loss',loss['mask loss'], args.iteration)
            writer.add_scalar('training acc',loss['acc'], args.iteration)

        # Checkpointing
        if args.iteration % args.save_interval == 0:
            # Deepspeed checkpoint
            path = get_deepspeed_checkpoint_dir(args.save)
            model.save_checkpoint(path)
            # Megatron checkpoint
            save_megatron_checkpoint(args.iteration, model, optimizer, lr_scheduler)

        # Evaluation
        if args.iteration % args.eval_interval == 0:
            loss_dict_val= evaluate(forward_step_func, val_data_iterator, model)
            if torch.distributed.get_rank() == 0:
                writer.add_scalar('validation mask loss',loss_dict_val['mask loss'], args.iteration)
                writer.add_scalar('validation acc',loss_dict_val['acc'], args.iteration)

    return args.iteration



def run_training(ckpt_dir='megatron_molbart_checkpoint'):
    deepspeed.init_distributed()
    initialize_megatron()
    args = get_args()
    args.iteration = 0

    os.makedirs(args.save, exist_ok=True)
    if args.xz:
        deepspeed_path = get_deepspeed_checkpoint_dir(args.save)
        os.makedirs(deepspeed_path, exist_ok=True)

    print_rank_0('Loading dataset(s) ...')
    path = os.path.dirname(os.path.realpath(__file__))
    loader = MoleculeDataLoader(args.dataset_path,
                                batch_size=args.batch_size, num_workers=32)
    (train_dataloader, val_dataloader) = loader.get_data()
    
    print_rank_0('Setting up model ...')
    (model, optimizer, lr_scheduler) = setup_model_and_optimizer(args)

    if ckpt_dir is not None:
        path = get_deepspeed_checkpoint_dir(args.save) if args.deepspeed else args.save
        model.load_checkpoint(path)
        args.iteration = load_deepspeed_iteration(path)
    
    print_rank_0('Starting training ...')
    train_dataloader = RepeatingLoader(train_dataloader)
    val_dataloader = RepeatingLoader(val_dataloader)

    train(
        forward_step,
        model,
        optimizer,
        lr_scheduler,
        iter(train_dataloader),
        train_dataloader,
        iter(val_dataloader),
        args.pipe_parallel_size,
        args,
        )


def load_model():
    initialize_megatron()
    args = get_args()
    (model, optimizer, lr_scheduler) = setup_model_and_optimizer(args)
    path = get_deepspeed_checkpoint_dir(args.save) if args.deepspeed else args.save
    ckpt = model.load_checkpoint(path)


if __name__ == '__main__':
    run_training()
