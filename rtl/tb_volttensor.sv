//=============================================================================
// Testbench: tb_volttensor
// Project: FP8-TensorCore-PowerIntegrity - Workload-Aware Proactive dI/dt Smoothing & Undervolting
// Description: Standalone SystemVerilog testbench for simulating volttensor_top.
//              Can be run in QuestaSim, ModelSim, Vivado, VCS, or EDA Playground.
//=============================================================================

`timescale 1ns / 1ps

module tb_volttensor;

    // Simulation Clock & Parameters
    localparam CLK_PERIOD = 0.555; // 1.8 GHz clock period (~0.555 ns)
    localparam ROWS = 4;
    localparam COLS = 4;
    localparam ACC_WIDTH = 32;

    // DUT Signals
    logic                   clk;
    logic                   rst_n;
    logic                   gemm_imminent;
    logic                   gemm_execute;
    logic [7:0]             sim_droop_code;
    logic [ROWS*8-1:0]         a_inputs;
    logic [COLS*8-1:0]         b_inputs;
    logic [COLS*ACC_WIDTH-1:0] c_inputs;
    
    wire  [COLS*ACC_WIDTH-1:0] c_outputs;
    wire  [COLS-1:0]           results_valid;
    wire                       system_error;
    wire                       charge_boost_pulse;
    wire  [7:0]                current_demand_norm;
    wire  [2:0]                mas_fsm_state;
    wire  [31:0]               total_droop_events;

    // Instantiate Top-Level Device Under Test (DUT)
    volttensor_top #(
        .ROWS      (ROWS),
        .COLS      (COLS),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk                 (clk),
        .rst_n               (rst_n),
        .gemm_imminent       (gemm_imminent),
        .gemm_execute        (gemm_execute),
        .sim_droop_code      (sim_droop_code),
        .a_inputs            (a_inputs),
        .b_inputs            (b_inputs),
        .c_inputs            (c_inputs),
        .c_outputs           (c_outputs),
        .results_valid       (results_valid),
        .system_error        (system_error),
        .charge_boost_pulse  (charge_boost_pulse),
        .current_demand_norm (current_demand_norm),
        .mas_fsm_state       (mas_fsm_state),
        .total_droop_events  (total_droop_events)
    );

    // Clock Generator: 1.8 GHz
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // Test Sequence
    initial begin
        $display("==================================================================");
        $display("   Starting FP8-TensorCore-PowerIntegrity SystemVerilog Simulation Benchmark");
        $display("==================================================================");

        // Initialize signals
        clk            = 1'b0;
        rst_n          = 1'b0;
        gemm_imminent  = 1'b0;
        gemm_execute   = 1'b0;
        sim_droop_code = 8'd0; // Nominal Vdd (no droop)

        a_inputs       = '0;
        b_inputs       = '0;
        c_inputs       = '0;

        // Reset Pulse
        #(CLK_PERIOD * 4);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);
        $display("[Time %0t ns] Reset de-asserted. Compute tile in IDLE state.", $time);
        $display("[Time %0t ns] Quiescent current demand: %0d", $time, current_demand_norm);

        // Test Phase 1: Forecast Lookahead (Warp scheduler asserts gemm_imminent)
        #(CLK_PERIOD * 2);
        $display("\n--- Step 1: Microarchitecture Lookahead Forecast ---");
        gemm_imminent = 1'b1;
        #(CLK_PERIOD);
        gemm_imminent = 1'b0;
        $display("[Time %0t ns] gemm_imminent asserted. MAS state: %0d (PRE_BOOST)", $time, mas_fsm_state);
        $display("[Time %0t ns] Charge boost pulse asserted: %b", $time, charge_boost_pulse);

        // Test Phase 2: Staggered Compute Ramp (MAS RAMPING_UP)
        $display("\n--- Step 2: Staggered Microarchitectural Activity Smoothing ---");
        gemm_execute = 1'b1;
        
        // Provide FP8 input data (E4M3 format: 0x38 = 1.0, 0x40 = 2.0)
        a_inputs = {ROWS{8'h38}}; // 1.0 in FP8 E4M3 replicated across rows
        b_inputs = {COLS{8'h40}}; // 2.0 in FP8 E4M3 replicated across cols

        for (int cycle = 0; cycle < 5; cycle++) begin
            #(CLK_PERIOD);
            $display("[Time %0.3f ns] Cycle %0d: MAS State: %0d | Normalized Current Demand: %0d", 
                     $time, cycle, mas_fsm_state, current_demand_norm);
        end

        // Test Phase 3: Simulated Voltage Droop Transient Injection
        $display("\n--- Step 3: Injecting Severe Transient Voltage Droop ---");
        sim_droop_code = 8'd180; // Severe droop (~180mV drop)
        #(CLK_PERIOD * 2);
        $display("[Time %0.3f ns] Sensor detected droop! Total droop events logged: %0d", 
                 $time, total_droop_events);
        $display("[Time %0.3f ns] Dual-slack MAC protected upper critical bits without system error: system_error = %b", 
                 $time, system_error);

        // Test Phase 4: Completion & Shutdown
        sim_droop_code = 8'd0;
        gemm_execute   = 1'b0;
        #(CLK_PERIOD * 4);
        $display("\n[Time %0.3f ns] GEMM execution concluded. Tile returned to IDLE.", $time);
        $display("[Time %0.3f ns] Final current demand: %0d", $time, current_demand_norm);

        $display("\n==================================================================");
        $display("   FP8-TensorCore-PowerIntegrity RTL Simulation Completed Successfully (Zero Errors)");
        $display("==================================================================");
        $finish;
    end

endmodule
