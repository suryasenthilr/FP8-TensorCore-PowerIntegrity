# VoltTensor: Hardware Power Integrity and Undervolting Architecture for FP8 Tensor Cores

VoltTensor is a digital VLSI hardware design implementing microarchitectural current slew (dI/dt) smoothing and dual-slack arithmetic for deep learning tensor accelerators. The architecture features an on-chip lookahead activity controller, a replica delay-line timing slack sensor, and a 4x4 FP8 (E4M3) systolic array synthesized down to standard logic gates.

---

## Hardware Architecture

The design is organized into four primary SystemVerilog modules under `rtl/`:

1. **Dual-Slack FP8 Multiply-Accumulate (`rtl/fp8_mac.sv`)**
   - Implements IEEE FP8 E4M3 (1 sign, 4 exponent, 3 mantissa) arithmetic.
   - Bifurcated timing paths: exponent and sign computation logic are protected with high timing margin, while the lower mantissa multiplier array features soft-slack centroid clamping under transient voltage droop.
   - Accumulates into a 32-bit output register.

2. **On-Chip Timing Slack and Droop Monitor (`rtl/pdn_slack_monitor.sv`)**
   - Replica delay-line sensor that measures local timing slack margins in real time.
   - Compares delay tap propagation depth against programmable warning and critical thresholds.
   - Logs droop events and asserts degradation flags when voltage margins tighten.

3. **Microarchitectural Activity Smoothing Controller (`rtl/proactive_mas_ctrl.sv`)**
   - Senses lookahead execution hints from the warp scheduler 4 cycles in advance.
   - Drives a progressive 4-cycle S-curve row-enable envelope (25% -> 50% -> 75% -> 100%) to flatten current slew (dI/dt).
   - Generates an auxiliary charge-boost control pulse to pre-charge local decoupling nodes prior to compute surges.

4. **Systolic Tensor Core Tile (`rtl/systolic_tile.sv` & `rtl/volttensor_top.sv`)**
   - Integrates a 4x4 array of processing elements (16 FP8 MACs).
   - Top-level module (`volttensor_top.sv`) connects the activity smoothing FSM, delay-line sensor, systolic tile, and on-chip current telemetry estimation logic.

---

## Logic Synthesis with Yosys

The design is verified for synthesis using the open-source Yosys Open SYnthesis Suite.

### Synthesis Results (`volttensor_top`)

- **Total Cell Count:** 4,281 standard logic gates
- **Flip-Flops:** 776 registers (739 DFFE, 37 DFF)
- **Logic Gates Breakdown:**
  - NAND Gates: 1,075
  - AND Gates: 863
  - XOR Gates: 688
  - OR Gates: 349
  - XNOR Gates: 290
  - NOR Gates: 178
  - NOT Inverters: 58
  - Multiplexers: 4
- **Synthesized Gate Netlist:** `syn/volttensor_netlist.v`
- **Full Synthesis Report:** `syn/synthesis_report.txt`

---

## Directory Structure

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

## Build and Simulation Instructions

### 1. Functional Simulation (Icarus Verilog)

Compile and run the self-checking SystemVerilog testbench:

```powershell
iverilog -g2012 -o volttensor_sim.vvp rtl/fp8_mac.sv rtl/pdn_slack_monitor.sv rtl/proactive_mas_ctrl.sv rtl/systolic_tile.sv rtl/volttensor_top.sv rtl/tb_volttensor.sv
vvp volttensor_sim.vvp
```

The testbench initializes the tile, exercises the 4-cycle lookahead forecast, steps through the progressive activity ramp, injects a simulated transient droop event, and validates that the dual-slack logic prevents system-level overflow without errors.

### 2. Gate-Level Synthesis (Yosys)

Run the automated Yosys synthesis flow:

```powershell
powershell -ExecutionPolicy Bypass -File syn/run_synth.ps1
```

Or invoke Yosys directly:

```bash
yosys -s syn/synth.ys
```

The synthesized gate-level netlist is written to `syn/volttensor_netlist.v`, and the gate count report is logged to `syn/synthesis_report.txt`.
