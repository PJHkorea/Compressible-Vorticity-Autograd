# ====================================================================
# [COMPRESSIBLE-VORTICITY-AUTOGRAD HARDWARE INSERTER - V1.0]          #
# @file: fng_cva_dynamic_adapter.py                                   #
# ====================================================================

import torch
import jax
import jax.numpy as jnp
from typing import Any, Dict

# Inherit upper static parameters and mathematical infrastructure constants
from fng_cva_config import CVA_BUCKET_SIZES, FEATURE_DIM, NUM_EXPERTS, compute_expert_register_capacity
from fng_cva_sharding_tower import FngCvaShardingTower
from fng_cva_autograd_bridge import FngCvaAutogradBridge

class FngCvaDynamicShapeAdapter:
    """
    [DYNAMIC SHAPE INSULATION CONTROLLER]
    
    An adapter layer that pre-freezes static powers-of-2 execution buckets 
    and handles runtime zero-latency kernel hot-swapping to eliminate XLA 
    compiler re-tracing stalls under highly dynamic token sequences.
    """
    def __init__(self, e2e_core_pipeline_factory: Any, mesh: Any):
        """
        [OFFLINE STATIC GRAPH FREEZE REGISTRY]
        Pre-warms and captures XLA-optimized algebraic execution graphs into native 
        machine code for all registered CVA bucket sizes upon object initialization.
        """
        self.mesh = mesh
        self.bucket_sizes = CVA_BUCKET_SIZES
        self.fabric_bucket_registry: Dict[int, FngCvaAutogradBridge] = {}
        
        print(f"[PRECOMPILER] Igniting offline static graph freeze for CVA buckets: {self.bucket_sizes}")
        
        for b_size in self.bucket_sizes:
            # Compute static register slot capacity per expert lane matching the current bucket size
            tokens_per_expert = compute_expert_register_capacity(b_size)
            
            # Spin up a dedicated multi-node sharding control tower for this bucket dimension
            sharding_tower = FngCvaShardingTower(
                mesh=self.mesh,
                bucket_size=b_size,
                tokens_per_expert=tokens_per_expert
            )
            
            # Form a permanent binding with the autograd bridge function to freeze the target execution track
            self.fabric_bucket_registry[b_size] = FngCvaAutogradBridge(
                sharding_tower=sharding_tower,
                mesh=mesh,
                bucket_size=b_size,
                tokens_per_expert=tokens_per_expert
            )
            print(f"   ├─ [FREEZE SUCCESS] Bucket Size {b_size:4d} ➔ XLA SPMD HLO Object Cached.")
        print(f" └─ [COMPILER LOCK] All dynamic boundary conditions structurally secured behind registry.\n")

    def _find_optimal_bucket(self, actual_tokens: int) -> int:
        """
        Retrieves the optimal static bucket size capable of covering the runtime 
        incoming token stream count via a zero-overhead sequential scan.
        """
        for b_size in self.bucket_sizes:
            if actual_tokens <= b_size:
                return b_size
        # Trigger an infrastructure firewall exception when input exceeds maximum pre-allocated bucket boundaries
        raise ValueError(f"[🚨 CVA ADAPTER ERROR] Inflow token count ({actual_tokens}) exceeds maximum hardlocked bucket window ({self.bucket_sizes[-1]}).")

    def inject_dynamic_inference_pass(self, hidden_states: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        [INTERLOCK RUNTIME ENTRANCE]
        Ingests dynamic streams -> applies static padding with negative vacuum masking -> returns via zero-copy slicing.
        """
        actual_tokens = hidden_states.size(0)
        target_bucket_size = self._find_optimal_bucket(actual_tokens)
        pad_size = target_bucket_size - actual_tokens

        # [🛡️ ALGEBRAIC VACUUM MASKING]: 
        # Fill padding areas with 0.0 and gating logits with an extreme negative vacuum (-1e9).
        # This isolates the downstream CUDA kernel's __argmax lane from routing padding rows into valid execution paths.
        if pad_size > 0:
            hidden_states_padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size), value=0.0)
            gate_logits_padded = torch.nn.functional.pad(gate_logits, (0, 0, 0, pad_size), value=-1e9)
        else:
            hidden_states_padded = hidden_states
            gate_logits_padded = gate_logits

        # Perform zero-latency runtime hot-swapping by dispatching the pre-frozen compiled pipeline registry
        matched_bridge_runner = self.fabric_bucket_registry[target_bucket_size]
        torch_combined_padded = matched_bridge_runner(hidden_states_padded, gate_logits_padded)

        # [🔒 ZERO-COPY VIRTUAL SLICING VIEW]: 
        # Eliminate dummy areas with zero additional copy overhead and restore the active virtual pointer layout.
        torch_final_out = torch_combined_padded[:actual_tokens, :]
        
        # Enforce an active lifetime fence to prevent the asynchronous GC from prematurely destroying underlying accelerator memory maps
        if hasattr(torch_combined_padded, "_source_tensors"):
            torch_final_out._source_tensors = torch_combined_padded._source_tensors

        return torch_final_out

    def __call__(self, hidden_states: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        return self.inject_dynamic_inference_pass(hidden_states, gate_logits)


print("====================================================================")
print("🛡️ CVA DYNAMIC BUCKET SHAPE INSULATION ADAPTER COMPLETE")
print("   ├─ [REGISTRY] Multi-Node SPMD Compiler Matrices Fully Defrosted.")
print("   └─ [MASKING] Extreme Negative Vacuum (-1e9) Hardware Firewall Active.")
print("====================================================================")
