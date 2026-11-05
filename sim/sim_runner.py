"""
=============================================================================
Module: sim_runner.py
Project: VoltTensor - Workload-Aware Proactive dI/dt Smoothing & Undervolting
Description: Full Co-Simulation Benchmark Runner. Simulates Transient PDN,
             Power Dissipation, Energy Efficiency, and Slack Margins.
=============================================================================
"""

import numpy as np
import os
from typing import Dict, Any
from sim.pdn_model import PDNSimulator, PDNParameters
from sim.workload_traces import LLMWorkloadGenerator

class VoltTensorExperiment:
    """
    Executes comparative benchmarks across:
    1. Baseline GPU (Fixed Guardband, No MAS)
    2. Reactive Throttled GPU (Standard On-Die Droop Sensor + Throttling)
    3. VoltTensor (Proactive MAS + Charge Boost + Undervolted Vdd)
    """
    def __init__(self):
        self.pdn_sim = PDNSimulator()
        self.workload_gen = LLMWorkloadGenerator()

    def run_transient_comparison(self) -> Dict[str, Any]:
        """
        Runs transient voltage droop and current transient analysis over a decode token window.
        """
        # Generate LLM decode step current profiles
        trace = self.workload_gen.generate_decode_burst_trace(
            total_time_ns=60.0,
            dt_ns=0.005,
            gemm_start_ns=15.0,
            gemm_duration_ns=25.0
        )
        
        time_s = trace["time_ns"] * 1e-9 # Convert ns to seconds for physical ODE solver

        # -------------------------------------------------------------
        # Physical CMOS Voltage Scaling Laws:
        # In digital CMOS logic:
        # Active current: I_active(Vdd) = C_eff * f * Vdd  (scales linearly with Vdd)
        # Leakage current: I_leak(Vdd) scales with DIBL & subthreshold slope
        # Baseline reference is calibrated at V_ref = 0.88V
        # -------------------------------------------------------------
        v_ref = 0.88

        # 1. Baseline Architecture: Vdd = 0.88V (Includes 160mV Guardband)
        v_nom_baseline = 0.88
        i_load_baseline = trace["i_baseline"] * (v_nom_baseline / v_ref)
        res_baseline = self.pdn_sim.simulate_transient(
            time_steps=time_s,
            i_load=i_load_baseline,
            boost_active=None,
            initial_v_nominal=v_nom_baseline
        )

        # 2. Reactive Throttling: Vdd = 0.84V (Modest guardband, reactive bubbles)
        v_nom_reactive = 0.84
        i_reactive = trace["i_baseline"].copy()
        t_detect = 15.0 + 1.2
        reactive_mask = (trace["time_ns"] >= t_detect) & (trace["time_ns"] < (t_detect + 3.0))
        i_reactive[reactive_mask] *= 0.65 # Throttles to 65% for 3ns
        i_load_reactive = i_reactive * (v_nom_reactive / v_ref)
        res_reactive = self.pdn_sim.simulate_transient(
            time_steps=time_s,
            i_load=i_load_reactive,
            boost_active=None,
            initial_v_nominal=v_nom_reactive
        )

        # 3. VoltTensor Architecture: Vdd = 0.74V (Aggressive Undervolting + MAS)
        v_nom_volttensor = 0.74 # 140mV reduction in supply voltage!
        i_load_volttensor = trace["i_volttensor"] * (v_nom_volttensor / v_ref)
        res_volttensor = self.pdn_sim.simulate_transient(
            time_steps=time_s,
            i_load=i_load_volttensor,
            boost_active=trace["boost_active"],
            initial_v_nominal=v_nom_volttensor
        )

        # -------------------------------------------------------------
        # Power & Energy Modeling
        # Dynamic: P_dyn(t) = V_die(t) * I_active(t)
        # Static Leakage: P_leak = V_die * I_leak0 * (V_die/v_ref)^2.5 (DIBL effect)
        # -------------------------------------------------------------
        dt_s = time_s[1] - time_s[0]

        # Instantaneous power: P(t) = V_die(t) * I_load(t)
        p_instant_baseline = res_baseline["v_die"] * i_load_baseline
        p_instant_reactive = res_reactive["v_die"] * i_load_reactive
        p_instant_volttensor = res_volttensor["v_die"] * i_load_volttensor

        # Energy consumed over the 60ns evaluation window: E = integral(P(t) dt)
        e_total_baseline = np.sum(p_instant_baseline) * dt_s * 1e6   # Microjoules (uJ)
        e_total_reactive = np.sum(p_instant_reactive) * dt_s * 1e6   # Microjoules (uJ)
        e_total_volttensor = np.sum(p_instant_volttensor) * dt_s * 1e6 # Microjoules (uJ)

        # Peak droop calculations relative to nominal
        droop_baseline_mv = (v_nom_baseline - np.min(res_baseline["v_die"])) * 1e3
        droop_reactive_mv = (v_nom_reactive - np.min(res_reactive["v_die"])) * 1e3
        droop_volttensor_mv = (v_nom_volttensor - np.min(res_volttensor["v_die"])) * 1e3

        energy_savings_pct = (1.0 - (e_total_volttensor / e_total_baseline)) * 100.0

        return {
            "time_ns": trace["time_ns"],
            "trace": trace,
            "res_baseline": res_baseline,
            "res_reactive": res_reactive,
            "res_volttensor": res_volttensor,
            "metrics": {
                "v_nom_baseline": v_nom_baseline,
                "v_nom_reactive": v_nom_reactive,
                "v_nom_volttensor": v_nom_volttensor,
                "droop_baseline_mv": droop_baseline_mv,
                "droop_reactive_mv": droop_reactive_mv,
                "droop_volttensor_mv": droop_volttensor_mv,
                "max_didt_baseline": trace["max_didt_baseline"],
                "max_didt_volttensor": trace["max_didt_volttensor"],
                "energy_baseline_uj": e_total_baseline,
                "energy_reactive_uj": e_total_reactive,
                "energy_volttensor_uj": e_total_volttensor,
                "energy_savings_pct": energy_savings_pct
            }
        }
