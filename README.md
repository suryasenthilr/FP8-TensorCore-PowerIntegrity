# VoltTensor: Hardware Power Integrity and Undervolting Architecture for FP8 Tensor Cores

VoltTensor is a digital VLSI hardware design implementing microarchitectural current slew (dI/dt) smoothing and dual-slack arithmetic for deep learning tensor accelerators. The architecture features an on-chip lookahead activity controller, a replica delay-line timing slack sensor, and a 4x4 FP8 (E4M3) systolic array synthesized down to standard logic gates.

---

## Top-Level Hardware Architecture

```text
 +-----------------------------------------------------------------------------------+
 |                                  VOLTTENSOR_TOP                                   |
 |                                                                                   |
 |  Warp Scheduler Interface                                                         |
 |  ────────────────────────                                                         |
 |  gemm_imminent ─────────> +------------------------+                              |
 |  gemm_execute  ─────────> |   proactive_mas_ctrl   |                              |
 |                           |   - FSM Activity Ramp  |────> charge_boost_pulse      |
 |                           |   - 4-cycle S-curve    |────> mas_fsm_state[2:0]      |
 |                           +-----------┬------------+                              |
 |                                       │ row_enable_mask[3:0]                      |
 |  PDN Telemetry Interface              │                                           |
 |  ───────────────────────              │                                           |
 |  sim_droop_code[7:0] ───> +-----------┴------------+                              |
 |                           |  pdn_slack_monitor     |                              |
 |                           |  - Delay Line Sensor   |────> total_droop_events[31:0]|
 |                           +-----------┬------------+                              |
 |                                       │ slack_critical                            |
 |  Systolic Matrix Datapath             │ (slack_degraded)                          |
 |  ────────────────────────             ▼                                           |
 |  a_inputs[31:0]  ───────> +------------------------+                              |
 |  b_inputs[31:0]  ───────> |     systolic_tile      |────> c_outputs[127:0]        |
 |  c_inputs[127:0] ───────> |     4x4 FP8 Array      |────> results_valid[3:0]      |
 |                           |     (16 Dual-Slack PEs)|────> system_error            |
 |                           +------------------------+                              |
 |                                                                                   |
 |  Telemetry Logic                                                                  |
 |  ───────────────                                                                  |
 |  Estimates active logic current ────────────────────────> current_demand_norm[7:0]|
 +-----------------------------------------------------------------------------------+
```

---

## Microarchitectural Timing and Current Ramp

When standard systolic arrays launch dense GEMM operations, activating 100% of processing elements on a single clock cycle produces extreme current slew (dI/dt). VoltTensor uses lookahead scheduling to stagger execution across a 4-cycle S-curve envelope while firing an auxiliary pre-charge pulse:

```text
Clock Cycle:       |  t-1 (Forecast) |  t+0 (25%)   |  t+1 (50%)   |  t+2 (75%)   |  t+3 (100%)  |
-------------------+-----------------+--------------+--------------+--------------+--------------+
gemm_imminent      |      HIGH       |     LOW      |     LOW      |     LOW      |     LOW      |
gemm_execute       |      LOW        |     HIGH     |     HIGH     |     HIGH     |     HIGH     |
mas_fsm_state      |   1 (PRE_BOOST) | 2 (RAMP_UP)  | 2 (RAMP_UP)  | 2 (RAMP_UP)  | 3 (FULL_EXEC)|
charge_boost_pulse |      HIGH       |     LOW      |     LOW      |     LOW      |     LOW      |
row_enable_mask    |      0000       |     0001     |     0011     |     0111     |     1111     |
Current Demand     |   15 (Quiescent)|      70      |     125      |     180      | 235 (Steady) |
```

---

## Hardware Submodule Breakdown

| Module | Source File | Description |
| :--- | :--- | :--- |
| **Dual-Slack MAC** | [`rtl/fp8_mac.sv`](rtl/fp8_mac.sv) | IEEE FP8 E4M3 arithmetic (1 sign, 4 exponent, 3 mantissa). Exponent/sign evaluated with high timing slack. Mantissa array gracefully clamps lower 2 bits to centroid value (`2'b10`) during droop alerts. Accumulates into 32-bit output. |
| **PDN Slack Monitor** | [`rtl/pdn_slack_monitor.sv`](rtl/pdn_slack_monitor.sv) | 32-tap replica delay-line sensor measuring propagation delay across on-chip voltage variations. Generates early warning (`slack_warning`) and critical droop (`slack_critical`) alerts; logs total droop events. |
| **MAS Controller** | [`rtl/proactive_mas_ctrl.sv`](rtl/proactive_mas_ctrl.sv) | 5-state FSM (`IDLE`, `PRE_BOOST`, `RAMPING_UP`, `FULL_EXEC`, `DAMPING`) driving 4-cycle S-curve row-gating (`row_enable_mask[3:0]`) and feedforward charge-boost pulse (`charge_boost_pulse`). |
| **Systolic Array Tile** | [`rtl/systolic_tile.sv`](rtl/systolic_tile.sv) | 4x4 array integrating 16 dual-slack FP8 processing elements with horizontal activation propagation, vertical weight propagation, and column accumulation. |
| **Top-Level Module** | [`rtl/volttensor_top.sv`](rtl/volttensor_top.sv) | Integrates the MAS FSM, delay-line monitor, systolic array tile, and current telemetry estimation logic. |

---

## Top-Level Pinout and Port Specification

| Port Name | Direction | Bit Width | Description |
| :--- | :---: | :---: | :--- |
| `clk` | Input | 1 | Master system clock (calibrated to 1.8 GHz) |
| `rst_n` | Input | 1 | Asynchronous active-low reset |
| `gemm_imminent` | Input | 1 | 4-cycle lookahead forecast from warp scheduler |
| `gemm_execute` | Input | 1 | Warp instruction execution active |
| `sim_droop_code` | Input | 8 | Digital delay tap / ADC voltage sensor input (0 = nominal, 255 = deep droop) |
| `a_inputs` | Input | 32 | Packed row activation inputs (4 rows x 8-bit FP8) |
| `b_inputs` | Input | 32 | Packed column weight inputs (4 cols x 8-bit FP8) |
| `c_inputs` | Input | 128 | Packed column accumulation inputs (4 cols x 32-bit accumulator) |
| `c_outputs` | Output | 128 | Packed column accumulation results (4 cols x 32-bit accumulator) |
| `results_valid` | Output | 4 | Per-column output valid flags |
| `system_error` | Output | 1 | Error flag asserted if exponent or sign logic encounters overflow |
| `charge_boost_pulse` | Output | 1 | Control pulse to trigger auxiliary feedforward charge injection |
| `current_demand_norm` | Output | 8 | Estimated instantaneous switching current telemetry (0 to 255) |
| `mas_fsm_state` | Output | 3 | FSM state telemetry (0: IDLE, 1: PRE_BOOST, 2: RAMPING_UP, 3: FULL_EXEC, 4: DAMPING) |
| `total_droop_events` | Output | 32 | Cumulative counter of detected voltage droop events |

---

## Logic Synthesis with Yosys

The design is synthesized using the open-source Yosys Open SYnthesis Suite (Yosys 0.70+286) with ABC technology mapping targeting generic logic gates.

### Synthesis Overview

| Metric / Artifact | Value / File Path | Description |
| :--- | :--- | :--- |
| **Top-Level Entity** | `volttensor_top` | 4x4 FP8 Tensor Core Tile with MAS & PDN Monitor |
| **Total Standard Cells** | 4,281 gates | Full design cell count |
| **Sequential Registers** | 776 Flip-Flops | 739 DFFE (clock-enabled), 37 DFF |
| **Combinational Gates** | 3,505 gates | Logic gates mapped via ABC |
| **Gate-Level Netlist** | [`syn/volttensor_netlist.v`](syn/volttensor_netlist.v) | Synthesized Verilog netlist |
| **Structural Model** | [`syn/volttensor_synth.json`](syn/volttensor_synth.json) | Structural JSON netlist |
| **Synthesis Log Report** | [`syn/synthesis_report.txt`](syn/synthesis_report.txt) | Complete cell statistics and hierarchy log |

### Gate Utilization Breakdown

| Gate Type | Yosys Cell Primitive | Gate Count | Percentage |
| :--- | :--- | :---: | :---: |
| NAND | `$_NAND_` | 1,075 | 25.1% |
| AND | `$_AND_` | 863 | 20.2% |
| XOR | `$_XOR_` | 688 | 16.1% |
| OR | `$_OR_` | 349 | 8.2% |
| XNOR | `$_XNOR_` | 290 | 6.8% |
| NOR | `$_NOR_` | 178 | 4.2% |
| NOT | `$_NOT_` | 58 | 1.4% |
| MUX | `$_MUX_` | 4 | 0.1% |
| **Total Combinational** | - | **3,505** | **81.9%** |
| **Total Sequential** | `$_DFF_` / `$_DFFE_` | **776** | **18.1%** |
| **Total Design Cells** | - | **4,281** | **100.0%** |

---

## Repository Structure

```
VoltTensor/
├── rtl/                        # Synthesizable SystemVerilog RTL
│   ├── fp8_mac.sv              # Dual-slack FP8 MAC with bit-criticality partitioning
│   ├── pdn_slack_monitor.sv    # Replica delay-line timing slack sensor
│   ├── proactive_mas_ctrl.sv   # Microarchitectural Activity Smoothing FSM
│   ├── systolic_tile.sv        # 4x4 Systolic Tensor Core array
│   ├── volttensor_top.sv       # Top-level integrated GPU compute tile
│   └── tb_volttensor.sv        # Standalone self-checking testbench
├── syn/                        # Yosys Logic Synthesis Flow
│   ├── synth.ys                # Yosys synthesis and ABC technology mapping script
│   ├── run_synth.ps1           # Automated synthesis execution script
│   ├── synthesis_report.txt    # Cell statistics, gate counts, and hierarchy report
│   ├── volttensor_netlist.v    # Synthesized gate-level Verilog netlist
│   └── volttensor_synth.json   # Structural JSON netlist
├── .gitignore                  # Git exclusions for binary and temporary files
└── README.md                   # Project documentation
```

---

## Build, Simulation, and Synthesis Instructions

### 1. Functional Simulation (Icarus Verilog)

Compile and execute the SystemVerilog testbench using Icarus Verilog (`iverilog`) and the VVP runtime engine:

```powershell
iverilog -g2012 -o volttensor_sim.vvp rtl/fp8_mac.sv rtl/pdn_slack_monitor.sv rtl/proactive_mas_ctrl.sv rtl/systolic_tile.sv rtl/volttensor_top.sv rtl/tb_volttensor.sv
vvp volttensor_sim.vvp
```

### Verified Simulation Output Trace

```text
==================================================================
   Starting VoltTensor SystemVerilog Simulation Benchmark
==================================================================
[Time 3000 ns] Reset de-asserted. Compute tile in IDLE state.
[Time 3000 ns] Quiescent current demand: 15

--- Step 1: Microarchitecture Lookahead Forecast ---
[Time 5000 ns] gemm_imminent asserted. MAS state: 1 (PRE_BOOST)
[Time 5000 ns] Charge boost pulse asserted: 1

--- Step 2: Staggered Microarchitectural Activity Smoothing ---
[Time 6.000 ns] Cycle 0: MAS State: 2 | Normalized Current Demand: 70
[Time 6.000 ns] Cycle 1: MAS State: 2 | Normalized Current Demand: 125
[Time 7.000 ns] Cycle 2: MAS State: 2 | Normalized Current Demand: 180
[Time 7.000 ns] Cycle 3: MAS State: 2 | Normalized Current Demand: 235
[Time 8.000 ns] Cycle 4: MAS State: 3 | Normalized Current Demand: 235

--- Step 3: Injecting Severe Transient Voltage Droop ---
[Time 9.000 ns] Sensor detected droop! Total droop events logged: 2
[Time 9.000 ns] Dual-slack MAC protected upper critical bits without system error: system_error = 0

[Time 11.000 ns] GEMM execution concluded. Tile returned to IDLE.
[Time 11.000 ns] Final current demand: 15

==================================================================
   VoltTensor RTL Simulation Completed Successfully (Zero Errors)
==================================================================
rtl/tb_volttensor.sv:128: $finish called at 11100 (1ps)
```

### 2. Logic Synthesis (Yosys)

Run the automated Yosys synthesis script:

```powershell
powershell -ExecutionPolicy Bypass -File syn/run_synth.ps1
```

Or execute Yosys directly from the command line:

```bash
yosys -s syn/synth.ys
```
