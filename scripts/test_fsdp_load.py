"""Minimal FSDP load test for 14B model. No training, no vLLM."""
import os, torch, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM

os.environ["NCCL_P2P_DISABLE"] = "1"

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

if rank == 0:
    print(f"FSDP test: {dist.get_world_size()} GPUs, rank {rank}")
    print("Loading 14B model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-14B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

if rank == 0:
    print("Wrapping with FSDP FULL_SHARD...")

import functools
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Qwen2DecoderLayer},
)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=wrap_policy,
    device_id=local_rank,
    use_orig_params=True,
    sync_module_states=True,
)

dist.barrier()
if rank == 0:
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  rank 0: {alloc:.1f}GB allocated")
    print("FSDP LOAD SUCCESS!")

# Quick forward test
if rank == 0:
    print("Testing forward pass...")
dummy = torch.randint(0, 1000, (1, 32)).cuda()
with torch.no_grad():
    out = model(dummy)

dist.barrier()
if rank == 0:
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  Forward pass OK, rank 0: {alloc:.1f}GB")
    print("ALL TESTS PASSED!")

dist.destroy_process_group()
