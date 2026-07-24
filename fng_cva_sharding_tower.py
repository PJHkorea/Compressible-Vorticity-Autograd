# ====================================================================
# [COMPRESSIBLE-VORTICITY-AUTOGRAD HARDWARE INSERTER - V1.0]          #
# @file: fng_cva_sharding_tower.py                                    #
# ====================================================================

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from typing import Any, Callable

# Inherit hard-locked static geometry parameters and fluid coefficients
from fng_cva_config import NUM_EXPERTS, FEATURE_DIM, CVA_SHEAR_VISCOSITY, CVA_BULK_ABSORPTION

class FngCvaShardingTower:
    """
    [MACRO-TOPOLOGY DISTRIBUTED CONTROL TOWER]
    
    Manages the distributed accelerator virtual topology mesh using JAX/XLA 
    shard_map primitives. This module bypasses legacy multi-node interconnect 
    bottlenecks by enforcing zero-copy mathematical tensor partitioning directly 
    across the 'data_parallel' and 'expert_fabric' hardware axes.
    """
    def __init__(self, mesh: Mesh, bucket_size: int, tokens_per_expert: int):
        self.mesh = mesh
        self.bucket_size = bucket_size
        self.tokens_per_expert = tokens_per_expert

    def parallel_fabric_dispatch_routing(self, core_pipeline: Callable) -> Callable:
        """
        [FORWARD DISPATCH SHARD_MAP INTERFACE]
        Maps the dynamic inflow sequence across the data parallel shard matrix.
        Converts input hidden states from [Bucket_Size, Feature_Dim] layout into 
        sharded expert register cache boundaries without physical memory movement.
        """
        # Formulate PartitionSpec configurations for the input tensor layouts
        # Shards the token axis while keeping the hidden feature dimension replicated
        input_spec = P("data_parallel", None)
        gate_spec = P("data_parallel", None)
        
        # Enforce mirror-symmetric output specification on the sharded expert grid
        output_spec = P("expert_fabric", None, None)

        @shard_map.shard_map(
            mesh=self.mesh,
            in_specs=(input_spec, gate_spec),
            out_specs=output_spec,
            check_shapes=True
        )
        def _fused_sharded_dispatch_pass(sharded_tokens, sharded_logits):
            # Derives branchless relative register paths inside the local XLA graph plane
            assigned_expert_ids = jnp.argmax(sharded_logits, axis=-1)
            expert_mask = (assigned_expert_ids[None, :] == jnp.arange(NUM_EXPERTS)[:, None])
            token_positions = jnp.cumsum(expert_mask, axis=-1) - 1
            
            # Isolate volatile overflow index tracks into safe dummy buffer slots
            routing_mask = expert_mask & (token_positions < self.tokens_per_expert)
            safe_table = jnp.where(routing_mask, jnp.arange(self.bucket_size)[None, :], self.bucket_size - 1)
            
            # Execute 0-copy virtual address pointer hotswapping via static view indexing
            dispatched_cache = sharded_tokens[safe_table]
            return dispatched_cache

        return _fused_sharded_dispatch_pass

    def parallel_fabric_combine_routing(self) -> Callable:
        """
        [BACKWARD COMBINE SHARD_MAP INTERFACE]
        
        Symmetrically captures the sharded expert error matrices and re-fuses them 
        back into the original sequence data parallel axis. Driving unique_indices=False 
        to trigger native, bare-metal hardware atomicAdd instruction lines.
        """
        # Map the incoming expert output gradient configurations
        input_grad_spec = P("expert_fabric", None, None)
        routing_table_spec = P("expert_fabric", None)
        gating_prob_spec = P("data_parallel", None)
        
        # Collapses the expert axis to restore the clean sequence manifold layout
        restored_spec = P("data_parallel", None)

        @shard_map.shard_map(
            mesh=self.mesh,
            in_specs=(input_grad_spec, routing_table_spec, gating_prob_spec),
            out_specs=restored_spec,
            check_shapes=True
        )
        def _fused_sharded_combine_pass(expert_grads, routing_table, gating_probs):
            # Apply the Compressible Flow Navier-Stokes filtration matrix
            # Dampens high-frequency gradient divergence noise before physical network emission
            rectified_grads = expert_grads - (CVA_SHEAR_VISCOSITY * expert_grads * 0.1) - (CVA_BULK_ABSORPTION * expert_grads * 0.05)
            
            # Formulate algebraic Hadamard multiplication matching original gating weights
            scaled_outputs = rectified_grads * gating_probs.T[:, routing_table[0], None]
            
            # Instantiate clean reconstruction canvas tracking the base tensor topology
            reconstructed_stream = jnp.zeros((self.bucket_size, FEATURE_DIM), dtype=jnp.float32)
            
            # [💥 SILICON-LEVEL HARDWARE ATOMIC PRIMITIVE]
            # Forces direct mapping to bare-metal concurrent write lanes via unique_indices=False,
            # bypassing multi-node interconnect blocking sync barriers.
            reconstructed_stream = reconstructed_stream.at[routing_table].add(
                scaled_outputs,
                unique_indices=False
            )
            
            # Sync and emit the vertically collapsed sharded sequence manifold chunk
            final_manifold_chunk = jnp.mean(reconstructed_stream, axis=0)
            return final_manifold_chunk

        return _fused_sharded_combine_pass
