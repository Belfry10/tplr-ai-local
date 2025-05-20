# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Standard library
import os
import sys
import time
import random
import asyncio
import argparse
from datetime import datetime
import threading
import contextlib

# Third party
import torch
import bittensor as bt
import numpy as np
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

# Local
import tplr
from neurons.demo import DeMo  # Import DeMo optimizer

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

async def retry_call(func, *args, attempts=3, delay=1, context="", **kwargs):
    """
    Calls an async function with retries.

    Args:
        func (Callable): An async function.
        *args: Positional arguments to pass to func.
        attempts (int): Number of retries.
        delay (int): Delay between attempts in seconds.
        context (str): Context description for logging.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The result of func(*args, **kwargs) or None if all attempts fail.
    """
    for attempt in range(attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            tplr.logger.error(
                f"Attempt {attempt + 1}/{attempts} failed for {context}: {e}"
            )
            await asyncio.sleep(delay)
    tplr.logger.error(f"Failed to complete {context} after {attempts} attempts.")
    return None

class AdamBaseline:
    """
    Baseline training implementation using AdamW and DDP for comparison with
    gradient compression and peer training.
    """
    
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='AdamW DDP Baseline')
        parser.add_argument('--project', type=str, default='boom', help='Wandb project.')
        parser.add_argument('--run_name', type=str, default='', help='Wandb run name.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        
        # DDP specific args
        parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), 
                            help='Number of GPUs to use for distributed training')
    #    parser.add_argument('--local_rank', type=int, default=-1, 
     #                       help='Local rank for distributed training. Set by torch.distributed.launch')
        
        # Optimizer args
        parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'demo'],
                           help='Optimizer to use for training (adamw or demo)')
        parser.add_argument('--learning_rate', type=float, default=4e-4, 
                            help='Learning rate for optimizer')
        parser.add_argument('--weight_decay', type=float, default=0.1, 
                            help='Weight decay for optimizer')
        parser.add_argument('--warmup_steps', type=int, default=250, 
                            help='Warmup steps for learning rate scheduler')
        
        # DeMo specific args
        parser.add_argument('--compression_decay', type=float, default=0.999,
                            help='Compression decay for DeMo optimizer')
        parser.add_argument('--compression_topk', type=int, default=32,
                            help='Compression topk for DeMo optimizer')
        parser.add_argument('--compression_chunk', type=int, default=64,
                            help='Compression chunk size for DeMo optimizer')
        parser.add_argument('--use_quantization', action='store_true',
                            help='Enable 8-bit quantization for gradient compression in DeMo optimizer')
        
        # Dataset args

        parser.add_argument('--pages_per_worker', type=int, default=1,
                            help='Number of dataset pages per window')
        parser.add_argument('--max_steps', type=int, default=20000,
                            help='Maximum number of training steps (None for unlimited)')
        parser.add_argument('--seed', type=str, default="adam_baseline",
                            help='Seed for deterministic page selection')
        
        # Checkpoint args
        parser.add_argument('--save_path', type=str, default='./checkpoints', 
                            help='Path to save model checkpoints')
        parser.add_argument('--save_interval', type=int, default=500, 
                            help='Save checkpoint every N windows')
        parser.add_argument('--load_checkpoint', type=str, default=None,
                            help='Path to checkpoint file to resume training from')
        
        bt.logging.add_args(parser)
        config = bt.config(parser)
        
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
            
        return config
    
    def __init__(self):
        tplr.logger.debug("Starting AdamW baseline initialization...")
        
        # Init config and load hparams
        self.config = AdamBaseline.config()
        self.hparams = tplr.load_hparams()
        # Update hparams with command line args
        self.hparams.learning_rate = self.config.learning_rate
        self.hparams.weight_decay = self.config.weight_decay
        self.hparams.pages_per_window = self.config.pages_per_worker
        
        # No bittensor initialization needed
        # Set up distributed training
        if self.config.local_rank != -1:
            # When using torchrun, use env vars
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"]) 
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            # Single-node, manually set up distribution
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
        
        # Initialize the distributed process group
        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            init_process_group(backend="nccl", rank=self.global_rank, world_size=self.world_size)
            tplr.logger.info(f"Initialized DDP: rank {self.global_rank}/{self.world_size-1} on device {self.local_rank}")
            
        # Set device
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize model, tokenizer, and DDP wrapper
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.device)
        
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        self.tokenizer = self.hparams.tokenizer
        
        # Initialize optimizer based on config
        if self.config.optimizer.lower() == 'demo':
            tplr.logger.info("Using DeMo optimizer")
            self.optimizer = DeMo(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.config.weight_decay,
                compression_decay=self.config.compression_decay,
                compression_topk=self.config.compression_topk,
                compression_chunk=self.config.compression_chunk,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
                use_quantization=self.config.use_quantization
            )
            if self.config.use_quantization:
                tplr.logger.info("Enabled 8-bit quantization for gradient compression")

        else:
            tplr.logger.info("Using AdamW optimizer")
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95)
            )
            
            # Set up scheduler similar to miner.py
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20000,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )
        
        # Create save directory if it doesn't exist
        if self.global_rank == 0:
            os.makedirs(self.config.save_path, exist_ok=True)
        
        # Init state params
        self.step_counter = 0
        self.global_step = 0
        self.window_step = 0
        
        # For tracking metrics
        self.total_tokens_processed = 0
        self.batch_times = []
        
        # Resume from checkpoint if specified
        if self.config.load_checkpoint is not None:
            self._load_checkpoint(self.config.load_checkpoint)
        
        # Initialize WandB on main process only
        if self.global_rank == 0:
            self.wandb = tplr.initialize_wandb(
                run_prefix='Dist',
                uid=self.config.run_name,
                config=self.config,
                group='baseline',
                job_type='adam_training'
            )
        else:
            self.wandb = None
    
    async def run(self):
        """Main training loop."""

        for window in range(self.window_step, self.config.max_steps):
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                tplr.logger.info(f"Reached maximum steps {self.config.max_steps}. Stopping.")
                break
                
            if self.global_rank == 0:
                tplr.logger.info(f"\n{'-' * 40} Window: {window} {'-' * 40}")
            
            seed = random.randint(0, 100000)
            # Get deterministic pages for this window
            pages = await retry_call(tplr.r2_dataset.R2DatasetLoader.next_pages,
                offset=window,
                n_pages=self.hparams.pages_per_window,
                seed=seed
            )
            
            # Create data loader
            loader = await retry_call(tplr.r2_dataset.R2DatasetLoader.create,
                batch_size=self.hparams.batch_size,
                sequence_length=self.hparams.sequence_length,
                pages_info=pages,
                tokenizer=self.tokenizer,
            )
            
            if self.global_rank == 0:
                tplr.logger.info(f"Pages: {[p[1] for p in pages]} for Window: {window}")
            
            # Training loop
            start_time = time.time()
            compute_start_time = time.time()  # New timing for computation phase
            if self.global_rank == 0:
                tplr.logger.info("Start accumulating gradients...")
            
            # Initialize optimizer state once at the beginning
            self.optimizer.zero_grad()
            self.model.zero_grad()
            
            total_loss = 0
            batch_tokens = 0
            batch_count = 0
            
            # Only rank 0 determines total batches in loader
            if self.global_rank == 0:
                # Count batches in the loader
                temp_batches = []
                for batch in loader:
                    temp_batches.append(batch)
                total_batches = len(temp_batches)
                # Free memory
                del temp_batches
                tplr.logger.info(f"Determined total batches: {total_batches}")
            else:
                total_batches = 0
                
            # Broadcast total_batches from rank 0 to all ranks
            if self.world_size > 1:
                total_batches_tensor = torch.tensor([total_batches], device=self.device)
                torch.distributed.broadcast(total_batches_tensor, src=0)
                total_batches = total_batches_tensor.item()
            
            # Reset the loader for actual processing
            pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                offset=window,
                n_pages=self.hparams.pages_per_window,
                seed=seed
            )
            
            # Create data loader again
            loader = await tplr.r2_dataset.R2DatasetLoader.create(
                batch_size=self.hparams.batch_size,
                sequence_length=self.hparams.sequence_length,
                pages_info=pages,
                tokenizer=self.tokenizer,
            )
            
            # Process batches with DDP no_sync for all batches
            if isinstance(self.model, DDP):
                ddp_context = self.model.no_sync()
            else:
                # Dummy context for non-DDP case
                ddp_context = contextlib.nullcontext()
            
            with ddp_context:
                for i, batch in enumerate(loader):
                    batch_count += 1
                    input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                    
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = self.model(input_ids=input_ids, labels=labels)
                    
                    # Normalize loss by total number of batches
                    loss = outputs.loss / total_batches
                    total_loss += outputs.loss.item()  # Track original loss for logging
                    loss.backward()
                    
                    # Track tokens
                    batch_tokens += (labels != -100).sum().item()
                    
                    if self.global_rank == 0 and i % 5 == 0:
                        tplr.logger.info(f'Batch {i}/{total_batches-1}, loss: {outputs.loss.item():.4f}')
            
            # Manually synchronize gradients after all batches
            if self.world_size > 1:
                # Only synchronize manually for AdamW, DeMo handles this internally
                if self.config.optimizer.lower() != 'demo':
                    if isinstance(self.model, DDP):
                        synchronize_gradients(self.model.module)
                    else:
                        synchronize_gradients(self.model)
            
            # Apply a single optimization step for all accumulated gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update learning rate schedule once per window
            self.scheduler.step()
            
            # Calculate compute time (from gradient accumulation to optimization)
            compute_duration = time.time() - compute_start_time
            
            # Calculate processing metrics
            duration = time.time() - start_time
            self.batch_times.append(duration)
            self.total_tokens_processed += batch_tokens
            
            if self.global_rank == 0:
                tplr.logger.info(f"Window {window} completed: {i+1} batches with {batch_tokens} tokens in {duration:.2f}s")
                tplr.logger.info(f"Compute time: {compute_duration:.2f}s, Tokens/sec: {batch_tokens/duration:.2f}")
                
                # Log gradient metrics
                grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                weight_norms = [p.norm().item() for p in self.model.parameters()]
                
                # Wandb logging
                metrics_dict = {
                    # Training metrics
                    "baseline/loss": total_loss/(i+1),
                    "baseline/tokens_per_sec": batch_tokens/duration,
                    "baseline/batch_duration": duration,
                    "baseline/compute_duration": compute_duration,  # New compute time metric
                    "baseline/total_tokens": self.total_tokens_processed,
                    "baseline/batch_tokens": batch_tokens,
                    "baseline/global_step": self.global_step,
                    
                    # Resource metrics
                    "baseline/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                    "baseline/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,  # MB
                    
                    # Network metrics
                    "baseline/num_gpus": self.world_size,
                    "baseline/effective_batch_size": self.world_size * self.hparams.batch_size,
                    
                    # Optimization metrics
                    "baseline/learning_rate": self.scheduler.get_last_lr()[0],
                    
                    # Gradient statistics as points
                    "baseline/mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0,
                    "baseline/max_grad_norm": max(grad_norms) if grad_norms else 0,
                    "baseline/min_grad_norm": min(grad_norms) if grad_norms else 0,
                    "baseline/grad_norm_std": torch.tensor(grad_norms).std().item() if grad_norms else 0,
                    "baseline/mean_weight_norm": sum(weight_norms) / len(weight_norms),
                }
                
                # Add DeMo specific metrics if using DeMo optimizer
                if self.config.optimizer.lower() == 'demo':
                    metrics_dict.update({
                        "baseline/data_transmit": self.optimizer.data_transmit / 1024**2,  # MB
                        "baseline/data_receive": self.optimizer.data_receive / 1024**2,  # MB
                    })
                    
                    # Add quantization metrics if enabled - use 1 instead of True for better W&B compatibility
                    if self.config.use_quantization:
                        metrics_dict.update({
                            "baseline/quantization_enabled": 1,  # Use numeric value instead of boolean
                        })
                    else:
                        metrics_dict.update({
                            "baseline/quantization_enabled": 0,  # Use numeric value instead of boolean
                        })
                
                self.wandb.log(metrics_dict, step=self.global_step)
                
                # Save checkpoint
                if window % self.config.save_interval == 0:
                    self._save_checkpoint(window)
            
            self.global_step += 1
            self.window_step += 1
            
            # Synchronize across processes
            if self.world_size > 1:
                torch.distributed.barrier()
                
                
    def _save_checkpoint(self, window):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.save_path, f"demo_checkpoint_window_{window}_{timestamp}.pt")
        
        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
            
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'window': window,
            'global_step': self.global_step,
        }
        
        torch.save(checkpoint, path)
        tplr.logger.info(f"Saved checkpoint to {path}")
        
    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            tplr.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        # Resume from the next window after the saved one
        self.window_step = checkpoint.get('window', 0) + 1
        
        tplr.logger.info(f"Resumed training from window {self.window_step-1}, global step {self.global_step}")
        
    def cleanup(self):
        """Clean up resources."""
        if self.world_size > 1:
            destroy_process_group()
        
        if self.wandb is not None:
            self.wandb.finish()

async def main():
    """Entry point."""
    baseline = AdamBaseline()
    
    try:
        await baseline.run()
    except KeyboardInterrupt:
        tplr.logger.info("Training interrupted by user")
    except Exception as e:
        tplr.logger.error(f"Error during training: {e}")
        raise
    finally:
        baseline.cleanup()

def synchronize_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()
            
if __name__ == "__main__":
    asyncio.run(main()) 
