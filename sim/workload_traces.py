"""
=============================================================================
Module: workload_traces.py
Project: VoltTensor - Workload-Aware Proactive dI/dt Smoothing & Undervolting
Description: Synthetic and LLM-Calibrated (LLaMA-3 Architecture) Workload Traces
             representing Prefill and Decode phase current profiles on GPU compute tiles.
=============================================================================
"""

import numpy as np
from typing import Dict, Tuple

class LLMWorkloadGenerator:
    """
    Generates realistic nano-second resolution current profiles for GPU compute tiles
    during Transformer LLM execution.
    """
    def __init__(
        self,
        clock_freq_ghz: float = 1.8,    # GPU Core Clock: 1.8 GHz (T_clk = 0.555 ns)
        i_quiescent: float = 18.0,      # Static + background memory controller leakage (A)
        i_peak_gemm: float = 240.0      # Peak current when full Tensor Core array is saturated (A)
    ):
        self.freq = clock_freq_ghz
        self.t_clk = 1.0 / clock_freq_ghz # Clock period in nanoseconds
        self.i_idle = i_quiescent
        self.i_peak = i_peak_gemm

    def generate_decode_burst_trace(
        self,
        total_time_ns: float = 60.0,
        dt_ns: float = 0.005,            # 5 picoseconds time-step for high-frequency RLC transients
        gemm_start_ns: float = 15.0,
        gemm_duration_ns: float = 25.0
    ) -> Dict[str, np.ndarray]:
        """
        Generates comparison traces for an autoregressive decode token step:
        1. Baseline: Abrupt step function from idle to full GEMM (high dI/dt).
        2. Reactive Throttle: Baseline with delayed reactive clock throttling.
        3. VoltTensor: Proactive lookahead, 4-cycle staggered ramp, and charge boost pulse.
        """
        time_steps = np.arange(0.0, total_time_ns, dt_ns)
        N = len(time_steps)

        # Baseline: instantaneous abrupt step transition with standard gate slew (~0.35 ns)
        i_baseline = np.full(N, self.i_idle)
        slew_ns = 0.35
        for idx, t in enumerate(time_steps):
            if gemm_start_ns <= t < gemm_start_ns + gemm_duration_ns:
                ramp = min(1.0, (t - gemm_start_ns) / slew_ns)
                i_baseline[idx] = self.i_idle + (self.i_peak - self.i_idle) * ramp
            elif t >= gemm_start_ns + gemm_duration_ns:
                fall = max(0.0, 1.0 - (t - (gemm_start_ns + gemm_duration_ns)) / slew_ns)
                i_baseline[idx] = self.i_idle + (self.i_peak - self.i_idle) * fall

        # VoltTensor: Microarchitectural Activity Smoothing (MAS)
        # 4-cycle proactive lookahead and smooth staggered ramp
        ramp_cycles = 4
        t_ramp_total = ramp_cycles * self.t_clk # 4 * 0.555 ns = 2.22 ns
        
        i_volttensor = np.full(N, self.i_idle)
        boost_active = np.zeros(N, dtype=bool)

        # Charge boost pulses right at the onset of the current ramp and sustains through the ramp window
        t_boost_start = gemm_start_ns - 0.2 * self.t_clk
        t_boost_end   = gemm_start_ns + t_ramp_total + 0.8 * self.t_clk
        boost_mask = (time_steps >= t_boost_start) & (time_steps < t_boost_end)
        boost_active[boost_mask] = True

        # Compute physically smoothed current profile across the 4 cycles
        # Each row adds a smooth portion of current as its clock tree enables
        for idx, t in enumerate(time_steps):
            if gemm_start_ns <= t < (gemm_start_ns + t_ramp_total):
                # Smooth 4-stage transition across 2.22 ns
                progress = (t - gemm_start_ns) / t_ramp_total
                # Smooth S-curve (cubic hermite) transition across the ramp window
                s_curve = 3 * progress**2 - 2 * progress**3
                i_volttensor[idx] = self.i_idle + (self.i_peak - self.i_idle) * s_curve
            elif (gemm_start_ns + t_ramp_total) <= t < (gemm_start_ns + gemm_duration_ns):
                i_volttensor[idx] = self.i_peak
            elif t >= gemm_start_ns + gemm_duration_ns:
                fall_duration = 3.0 * self.t_clk
                fall_progress = min(1.0, (t - (gemm_start_ns + gemm_duration_ns)) / fall_duration)
                fall_s = 1.0 - (3 * fall_progress**2 - 2 * fall_progress**3)
                i_volttensor[idx] = self.i_idle + (self.i_peak - self.i_idle) * fall_s

        # Calculate instantaneous dI/dt (in Amperes / nanosecond)
        di_dt_baseline = np.gradient(i_baseline, dt_ns)
        di_dt_volttensor = np.gradient(i_volttensor, dt_ns)

        return {
            "time_ns": time_steps,
            "i_baseline": i_baseline,
            "i_volttensor": i_volttensor,
            "boost_active": boost_active,
            "di_dt_baseline": di_dt_baseline,
            "di_dt_volttensor": di_dt_volttensor,
            "max_didt_baseline": np.max(di_dt_baseline),
            "max_didt_volttensor": np.max(di_dt_volttensor)
        }
