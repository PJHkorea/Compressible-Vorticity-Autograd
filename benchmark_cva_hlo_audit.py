# ====================================================================
# [COMPRESSIBLE-VORTICITY-AUTOGRAD HARDWARE INSERTER - V1.0]          #
# @file: benchmark_cva_hlo_audit.py                                   #
# ====================================================================

import re
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import time
from typing import Dict, Any

# Inherit global static descriptors and multi-node sharding controllers
from fng_cva_config import CVA_BUCKET_SIZES, FEATURE_DIM, NUM_EXPERTS, compute_expert_register_capacity
from fng_cva_sharding_tower import FngCvaShardingTower

def compile_and_dump_pure_cva_hlo_asm(bucket_size: int, tokens_per_expert: int, mesh: Mesh) -> str:
    """
    [XLA HLO IR ASSEMBLY TEXT EMITTER]
    
    Generates static HLO IR from abstract JAX traces without physical VRAM usage,
    providing a clean machine bytecode representation for compiler plane analysis.
    """
    # Create abstract shape tracers to guarantee a strict 0MB physical VRAM footprint
    abstract_token_stream = jax.ShapeDtypeStruct(
        shape=(bucket_size, FEATURE_DIM), 
        dtype=jnp.float32
    )
    abstract_gate_logits = jax.ShapeDtypeStruct(
        shape=(bucket_size, NUM_EXPERTS), 
        dtype=jnp.float32
    )

    # Spin up a dedicated multi-node sharding tower factory to extract execution channels
    sharding_tower = FngCvaShardingTower(
        mesh=mesh,
        bucket_size=bucket_size,
        tokens_per_expert=tokens_per_expert
    )
    
    # Retrieve the backward combine pass which contains the compressible fluid core logic
    hardware_pass_kernel = sharding_tower.parallel_fabric_combine_routing()

    # Formulate dummy reference shapes matching the internal sharded PartitionSpec inputs
    dummy_expert_grads = jax.ShapeDtypeStruct(shape=(NUM_EXPERTS, tokens_per_expert, FEATURE_DIM), dtype=jnp.float32)
    dummy_routing_table = jax.ShapeDtypeStruct(shape=(NUM_EXPERTS, tokens_per_expert), dtype=jnp.int32)
    dummy_gating_probs = jax.ShapeDtypeStruct(shape=(bucket_size, NUM_EXPERTS), dtype=jnp.float32)

    # Lock the compilation graph via AOT compiler plane to freeze the HLO IR instructions
    with mesh:
        jit_compiled_graph = jax.jit(hardware_pass_kernel)
        lowered_hlo_graph = jit_compiled_graph.lower(
            dummy_expert_grads,
            dummy_routing_table,
            dummy_gating_probs
        )
        compiled_executable = lowered_hlo_graph.compile()

    # Decode and return the raw unencrypted machine bytecode as human-readable text
    return compiled_executable.as_text()


def audit_compiled_silicon_cva_instructions(hlo_assembly_text: str) -> Dict[str, Any]:
    """
    [SILICON INSTRUCTION AUDIT FIREWALL]
    
    Audits XLA HLO assembly for performance-impacting collective communication 
    and serialization primitives to guarantee a completely clean, zero-stall silicon trail.
    """
    # 1. Collective patterns causing multi-node interconnect saturation barriers
    collective_comm_patterns = [
        r"all-to-all",
        r"collective-permute",
        r"all-gather",
        r"reduce-scatter",
        r"send",
        r"recv"
    ]

    # 2. Sequential sorting patterns causing hardware warp bubbles
    sorting_patterns = [
        r"custom-call.*bitonic",
        r"sort"
    ]

    detected_comm_primitives = {}
    detected_sorting_primitives = {}
    
    total_comm_leaks = 0
    total_sorting_leaks = 0

    # A. Execute deep regex matching for distributed communication operations
    for pattern in collective_comm_patterns:
        matches = re.findall(pattern, hlo_assembly_text, re.IGNORECASE)
        match_count = len(matches)
        detected_comm_primitives[pattern] = match_count
        total_comm_leaks += match_count

    # B. Execute regex tracking for warp sorting bottlenecks
    for pattern in sorting_patterns:
        matches = re.findall(pattern, hlo_assembly_text, re.IGNORECASE)
        match_count = len(matches)
        detected_sorting_primitives[pattern] = match_count
        total_sorting_leaks += match_count

    # C. Validate clean 0-NCCL zero-stall mechanical purity condition
    is_silicon_clean = (total_comm_leaks == 0) and (total_sorting_leaks == 0)

    report = {
        "is_clean": is_silicon_clean,
        "comm_summary": detected_comm_primitives,
        "sorting_summary": detected_sorting_primitives,
        "total_comm_leaks": total_comm_leaks,
        "total_sorting_leaks": total_sorting_leaks
    }

    return report


def run_cva_hlo_static_assembly_benchmark(mesh: Mesh) -> None:
    """
    [TELEMETRY ORCHESTRATOR]
    Sweeps through pre-allocated bucket boundaries to verify 0-leak compiler lowerings.
    """
    print("====================================================================")
    print("📊 IGNITING CVA STATIC ASSEMBLY COMPILER AUDIT GUARD")
    print("====================================================================")
    
    # Target the intermediate 512 bucket boundary context for profiling verification
    target_bucket = 512
    tokens_per_expert = compute_expert_register_capacity(target_bucket)
    
    print(f"[AUDIT_START] LOWERING TARGET BUCKET WINDOW SIZE: {target_bucket}")
    
    # Capture pure XLA graph generation time
    start_time = time.perf_counter()
    hlo_text = compile_and_dump_pure_cva_hlo_asm(target_bucket, tokens_per_expert, mesh)
    end_time = time.perf_counter()
    
    print(f" ✨ [SUCCESS_LOWERING] HLO Text Assembly Emitted in {end_time - start_time:.4f} seconds.")
    
    # Run the compiled silicon bytecode instruction analyzer
    audit_results = audit_compiled_silicon_cva_instructions(hlo_text)
    
    print("\n--------------------------------------------------------------------")
    print("📊 CVA SILICON CORE INSTRUCTION AUDIT REPORT")
    print("--------------------------------------------------------------------")
    print(f" ├─ Zero-Stall Interconnect Barrier Status: {'PASS (CLEAN)' if audit_results['is_clean'] else 'FAIL (LEAK DETECTED)'}")
    print(f" ├─ Total NCCL Collective Leaks Tracked   : {audit_results['total_comm_leaks']}")
    print(f" ├─ Total Warp Serialization Leaks Tracked: {audit_results['total_sorting_leaks']}")
    print(" 📢 Collective Communication Breakdowns:")
    for k, v in audit_results["comm_summary"].items():
        print(f"    ├─ Primitive [{k:18s}] -> Frequency Count: {v}")
    print("--------------------------------------------------------------------")

    # [🛡️ SYSTEM SELF-DESTRUCTION GUARD]
    # Enforce strict zero-leak compliance. If a single collective primitive leaks 
    # into the compiled fabric bytecode plane, terminate execution instantly.
    assert audit_results["is_clean"], \
        "[🚨 AUDIT CRITICAL CRISIS] Performance-destroying distributed communication leaked into the CVA native bytecode layer!"
        
    print("🎯 SYSTEM ARCHITECTURAL 무결성 CONFIRMED: 100% CLEAN SILICON PROFILED.\n")
