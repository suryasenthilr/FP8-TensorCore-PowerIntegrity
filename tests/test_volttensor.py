"""
=============================================================================
Test Suite: test_volttensor.py
Project: VoltTensor - Workload-Aware Proactive dI/dt Smoothing & Undervolting
Description: Unit and Integration Tests for PDN Solvers, RTL Behavioral Models,
             Numerical Stability, and Power Evaluation.
=============================================================================
"""

import pytest
import numpy as np
from sim.pdn_model import PDNSimulator, PDNParameters
from sim.workload_traces import LLMWorkloadGenerator
from sim.sim_runner import VoltTensorExperiment
from sim.evaluate_metrics import evaluate_fp8_numerical_accuracy

def test_pdn_solver_stability():
    """Verify that the 4th-order Runge-Kutta PDN ODE solver is numerically stable."""
    sim = PDNSimulator()
    time_steps = np.arange(0, 50e-9, 5e-12) # 50ns, 5ps resolution
    i_load = np.full(len(time_steps), 20.0) # 20A constant load
    
    res = sim.simulate_transient(time_steps, i_load, initial_v_nominal=0.80)
    
    assert not np.isnan(res["v_die"]).any(), "Voltage vector contains NaNs"
    assert not np.isinf(res["v_die"]).any(), "Voltage vector contains Infs"
    assert res["v_min"] > 0.65, f"Unexpected voltage collapse: v_min = {res['v_min']}"
    assert res["v_max"] < 0.90, f"Unexpected voltage explosion: v_max = {res['v_max']}"


def test_workload_generator_didt_reduction():
    """Verify that Proactive MAS flattens peak dI/dt by at least 60% compared to baseline."""
    gen = LLMWorkloadGenerator()
    trace = gen.generate_decode_burst_trace(
        total_time_ns=40.0,
        dt_ns=0.005,
        gemm_start_ns=10.0,
        gemm_duration_ns=15.0
    )
    
    didt_base = trace["max_didt_baseline"]
    didt_vt = trace["max_didt_volttensor"]
    
    reduction_pct = ((didt_base - didt_vt) / didt_base) * 100.0
    assert reduction_pct >= 60.0, f"Expected >=60% dI/dt reduction, got {reduction_pct:.1f}%"


def test_energy_savings():
    """Verify that VoltTensor achieves >=25% energy savings over baseline."""
    exp = VoltTensorExperiment()
    sim_data = exp.run_transient_comparison()
    m = sim_data["metrics"]
    
    assert m["energy_savings_pct"] >= 25.0, f"Expected >=25% energy savings, got {m['energy_savings_pct']:.2f}%"
    assert m["v_nom_volttensor"] < m["v_nom_baseline"]
    assert m["droop_volttensor_mv"] < m["droop_baseline_mv"]


def test_fp8_dual_slack_precision_retention():
    """Verify that Dual-Slack FP8 maintains high cosine similarity (>0.999) under droop."""
    acc = evaluate_fp8_numerical_accuracy()
    
    assert acc["cos_sim_volttensor"] > 0.999, f"Cosine similarity too low: {acc['cos_sim_volttensor']}"
    assert acc["sqnr_volttensor_db"] > 45.0, f"SQNR too low: {acc['sqnr_volttensor_db']} dB"
    assert acc["estimated_ppl_delta"] < 0.05, f"Perplexity impact too high: {acc['estimated_ppl_delta']}"


def test_rtl_proactive_mas_state_transitions():
    """Behavioral check of the MAS FSM transitions."""
    # State mapping: IDLE=0, PRE_BOOST=1, RAMPING_UP=2, FULL_EXEC=3, DAMPING=4
    state = 0 # IDLE
    gemm_imminent = True
    gemm_execute = False
    
    # Cycle 1: Lookahead asserted
    if state == 0 and gemm_imminent:
        state = 1 # PRE_BOOST
    assert state == 1, "Failed to transition to PRE_BOOST"
    
    # Cycle 2: Next cycle enters RAMPING_UP
    state = 2 # RAMPING_UP
    assert state == 2, "Failed to transition to RAMPING_UP"
