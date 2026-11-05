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

The RTL source files are located in `rtl/`:

1. **Dual-Slack FP8 Multiply-Accumulate (`rtl/fp8_mac.sv`)**
   - Implements IEEE FP8 E4M3 arithmetic (1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits).
   - Bifurcated timing paths:
     - **High-Slack Path (Sign and Exponent):** Fast evaluation core with guaranteed timing margin against voltage droop to eliminate exponent underflow and overflow errors.
     - **Soft-Slack Path (Mantissa Array):** Dual-slack multiplier array. Under severe droop conditions flagged by the delay sensor, the lowest 2 mantissa bits gracefully clamp to the centroid value (`2'b10`) to eliminate setup-time violations without pipeline stalls.
   - Accumulates into a 32-bit register.

2. **On-Chip Timing Slack and Droop Monitor (`rtl/pdn_slack_monitor.sv`)**
   - Tunable replica delay-line sensor modeling critical-path propagation.
   - Evaluates real-time timing slack by sampling pulse travel depth across 32 delay taps.
   - Generates early warning (`slack_warning`) and critical droop (`slack_critical`) alerts.
   - Maintains an on-chip cumulative droop event counter (`total_droop_events`).

3. **Microarchitectural Activity Smoothing Controller (`rtl/proactive_mas_ctrl.sv`)**
   - Implements a five-state FSM: `IDLE`, `PRE_BOOST`, `RAMPING_UP`, `FULL_EXEC`, and `DAMPING`.
   - Generates progressive per-row clock gates (`row_enable_mask[3:0]`).
   - Drives feedforward charge injection (`charge_boost_pulse`) 1 cycle prior to compute surges.
   - Includes transient recovery throttling if the PDN monitor detects critical slack collapse.

4. **Systolic Tensor Core Tile (`rtl/systolic_tile.sv`)**
   - 2D grid containing 16 dual-slack FP8 processing elements arranged in a 4x4 topology.
   - Implements horizontal activation propagation, vertical weight propagation, and column-wise accumulation.

5. **Top-Level Integrated Tile (`rtl/volttensor_top.sv`)**
   - Integrates the FSM controller, delay-line sensor, systolic tile, and current estimation logic.

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

The RTL was synthesized using the open-source Yosys Open SYnthesis Suite (Yosys 0.70+286) targeting generic logic gates via ABC technology mapping.

### Synthesis Summary (`volttensor_top`)

- **Total Cell Count:** 4,281 standard logic gates
- **Sequential Elements:** 776 registers (739 DFFE, 37 DFF)
- **Gate Breakdown:**
  - NAND Gates (`$_NAND_`): 1,075
  - AND Gates (`$_AND_`): 863
  - XOR Gates (`$_XOR_`): 688
  - OR Gates (`$_OR_`): 349
  - XNOR Gates (`$_XNOR_`): 290
  - NOR Gates (`$_NOR_`): 178
  - NOT Inverters (`$_NOT_`): 58
  - Multiplexers (`$_MUX_`): 4
- **Synthesized Netlist:** `syn/volttensor_netlist.v`
- **Structural JSON Model:** `syn/volttensor_synth.json`
- **Full Synthesis Report:** `syn/synthesis_report.txt`

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
