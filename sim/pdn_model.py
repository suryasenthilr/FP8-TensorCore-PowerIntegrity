"""
=============================================================================
Module: pdn_model.py
Project: VoltTensor - Workload-Aware Proactive dI/dt Smoothing & Undervolting
Description: Multi-Stage Lumped RLC Power Delivery Network (PDN) Transient Solver.
             Models VRM, Motherboard, Package Bumps, and On-Die Decoupling Network.
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class PDNParameters:
    """
    Physical PDN parasitics calibrated to 4nm/5nm class GPU compute packages
    (similar to NVIDIA Hopper H100 / Blackwell B200 class substrate specifications).
    """
    # Nominal supply voltage
    v_nominal: float = 0.80           # Nominal operating Vdd (Volts)
    
    # Board / Voltage Regulator Module (VRM) Stage
    r_vrm: float = 0.25e-3            # VRM internal resistance (0.25 mOhm)
    l_vrm: float = 0.8e-9             # Multiphase VRM effective filter inductance (0.8 nH)
    c_bulk: float = 1200e-6           # Motherboard bulk capacitors (1200 uF)
    esr_bulk: float = 0.3e-3          # Bulk capacitor ESR (0.3 mOhm)
    k_vrm_p: float = 2.5              # VRM active feedback proportional gain
    
    # Package & Socket Stage
    r_pkg: float = 0.10e-3            # Package substrate resistance (0.10 mOhm)
    l_pkg: float = 8.0e-12            # Package loop inductance (8 pH)
    c_pkg: float = 80e-6              # On-package land-side decoupling capacitors (80 uF)
    esr_pkg: float = 0.08e-3          # Package capacitor ESR (0.08 mOhm)
    
    # Die & Interposer Stage (C4 micro-bumps + deep trench capacitors)
    r_die: float = 0.06e-3            # On-die metallization resistance (0.06 mOhm)
    l_die: float = 1.0e-12            # Micro-bump / via loop inductance (1.0 pH)
    c_die: float = 12.0e-6            # On-die intrinsic + deep trench decap (12.0 uF)
    esr_die: float = 0.04e-3          # On-die decap ESR (0.04 mOhm)
    
    # Active Droop Clamping / Charge Boost Circuit
    c_boost: float = 1.2e-6           # Auxiliary switched charge-injection capacitor (1.2 uF)
    r_boost: float = 0.08e-3          # Boost path resistance
    i_boost_current: float = 85.0     # Active localized charge injection current capability (85 A)


class PDNSimulator:
    """
    State-Space Time-Domain Transient Solver for Multi-Stage GPU PDN.
    
    State Vector:
    x = [
        I_vrm   : Current through VRM inductor (A)
        V_bulk  : Voltage across motherboard bulk capacitance (V)
        I_pkg   : Current through package trace inductance (A)
        V_pkg   : Voltage across on-package decoupling capacitors (V)
        I_bump  : Current through C4 microbumps/vias (A)
        V_die   : Voltage across on-die decoupling capacitors (V)
    ]
    """
    def __init__(self, params: PDNParameters = None):
        self.p = params if params is not None else PDNParameters()

    def simulate_transient(
        self,
        time_steps: np.ndarray,
        i_load: np.ndarray,
        boost_active: np.ndarray = None,
        initial_v_nominal: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Solves the system of ordinary differential equations (ODEs) using
        a 4th-order Runge-Kutta (RK4) integration method for maximum numerical stability.
        """
        N = len(time_steps)
        dt = time_steps[1] - time_steps[0]
        
        v_in = initial_v_nominal if initial_v_nominal is not None else self.p.v_nominal
        if boost_active is None:
            boost_active = np.zeros(N, dtype=bool)

        # Preallocate state vectors
        i_vrm  = np.zeros(N)
        v_bulk = np.zeros(N)
        i_pkg  = np.zeros(N)
        v_pkg  = np.zeros(N)
        i_bump = np.zeros(N)
        v_die  = np.zeros(N)

        # Initial steady-state condition (quiescent load)
        i_init = i_load[0]
        v_die_init = v_in - i_init * (self.p.r_vrm + self.p.r_pkg + self.p.r_die)
        
        v_die[0]  = v_die_init
        v_pkg[0]  = v_in - i_init * (self.p.r_vrm + self.p.r_pkg)
        v_bulk[0] = v_in - i_init * self.p.r_vrm
        i_vrm[0]  = i_init
        i_pkg[0]  = i_init
        i_bump[0] = i_init

        # State derivative function: dx/dt = f(x, t, i_load_val, boost_val)
        def derivatives(state, i_l, boost):
            _i_vrm, _v_bulk, _i_pkg, _v_pkg, _i_bump, _v_die = state
            
            # VRM active feedback regulation: senses bulk/package droop and adjusts drive voltage
            v_vrm_drive = v_in + self.p.k_vrm_p * (v_in - _v_bulk)
            v_vrm_drive = np.clip(v_vrm_drive, v_in * 0.9, v_in * 1.35) # Realistic VRM output limits

            # 1. VRM Inductor current derivative
            di_vrm_dt = (v_vrm_drive - _v_bulk - _i_vrm * self.p.r_vrm) / self.p.l_vrm
            
            # 2. Bulk Capacitor voltage derivative
            dv_bulk_dt = (_i_vrm - _i_pkg) / self.p.c_bulk
            
            # 3. Package Inductor current derivative
            # Effective voltage at bulk terminal with ESR
            v_bulk_term = _v_bulk + (_i_vrm - _i_pkg) * self.p.esr_bulk
            v_pkg_term  = _v_pkg + (_i_pkg - _i_bump) * self.p.esr_pkg
            di_pkg_dt   = (v_bulk_term - v_pkg_term - _i_pkg * self.p.r_pkg) / self.p.l_pkg
            
            # 4. Package Capacitor voltage derivative
            dv_pkg_dt = (_i_pkg - _i_bump) / self.p.c_pkg
            
            # 5. C4 Microbump current derivative
            v_die_term = _v_die + (_i_bump - i_l) * self.p.esr_die
            di_bump_dt = (v_pkg_term - v_die_term - _i_bump * self.p.r_die) / self.p.l_die
            
            # 6. Die Decoupling Capacitor voltage derivative
            # Incorporate auxiliary charge injection if boost is pulsing
            i_boost_inj = 0.0
            if boost:
                # Active localized charge-pump injection from pre-charged reservoir
                # Supplies instantaneous current directly to the die rail to counteract L*(di/dt)
                i_boost_inj = self.p.i_boost_current

            dv_die_dt = (_i_bump + i_boost_inj - i_l) / self.p.c_die
            
            return np.array([di_vrm_dt, dv_bulk_dt, di_pkg_dt, dv_pkg_dt, di_bump_dt, dv_die_dt])

        # RK4 Integration Loop
        for n in range(N - 1):
            curr_state = np.array([i_vrm[n], v_bulk[n], i_pkg[n], v_pkg[n], i_bump[n], v_die[n]])
            il_n = i_load[n]
            il_np1 = i_load[n+1]
            il_mid = 0.5 * (il_n + il_np1)
            
            b_n = boost_active[n]
            b_np1 = boost_active[n+1]
            b_mid = b_n or b_np1

            k1 = derivatives(curr_state, il_n, b_n)
            k2 = derivatives(curr_state + 0.5 * dt * k1, il_mid, b_mid)
            k3 = derivatives(curr_state + 0.5 * dt * k2, il_mid, b_mid)
            k4 = derivatives(curr_state + dt * k3, il_np1, b_np1)

            next_state = curr_state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

            i_vrm[n+1]  = next_state[0]
            v_bulk[n+1] = next_state[1]
            i_pkg[n+1]  = next_state[2]
            v_pkg[n+1]  = next_state[3]
            i_bump[n+1] = next_state[4]
            v_die[n+1]  = next_state[5]

        # Calculate instantaneous terminal voltage on the compute transistors (including ESR droop)
        v_transistor = v_die + (i_bump - i_load) * self.p.esr_die

        # Calculate metrics
        v_droop_max = np.max(v_in - v_transistor)
        v_overshoot_max = np.max(v_transistor - v_in)

        return {
            "time": time_steps,
            "v_die": v_transistor,
            "v_capacitor": v_die,
            "i_load": i_load,
            "i_bump": i_bump,
            "i_pkg": i_pkg,
            "v_droop_max": v_droop_max,
            "v_overshoot_max": v_overshoot_max,
            "v_min": np.min(v_transistor),
            "v_max": np.max(v_transistor)
        }
