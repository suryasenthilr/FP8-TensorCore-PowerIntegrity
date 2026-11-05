//=============================================================================
// Module: volttensor_top
// Project: VoltTensor - Workload-Aware Proactive dI/dt Smoothing & Undervolting
// Description: Top-Level Integrated GPU Compute Tile with VoltTensor Power
//              Delivery Management, Proactive MAS, and Dynamic Droop Clamping.
//=============================================================================

`timescale 1ns / 1ps

module volttensor_top #(
    parameter ROWS        = 4,
    parameter COLS        = 4,
    parameter ACC_WIDTH   = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,
    
    // Microarchitectural Warp Scheduler Interface
    input  logic                   gemm_imminent,     // 4-cycle lookahead forecast
    input  logic                   gemm_execute,      // Warp execution valid
    
    // PDN Voltage Telemetry / Sensor Feedback
    input  logic [7:0]             sim_droop_code,    // Delay tap sensor input from PDN
    
    // Systolic Matrix Inputs
    input  logic [ROWS*8-1:0]          a_inputs,
    input  logic [COLS*8-1:0]          b_inputs,
    input  logic [COLS*ACC_WIDTH-1:0]  c_inputs,
    
    // Systolic Matrix Outputs
    output logic [COLS*ACC_WIDTH-1:0]  c_outputs,
    output logic [COLS-1:0]            results_valid,
    output logic                   system_error,
    
    // Power & PDN Interface
    output logic                   charge_boost_pulse, // Pulse to active capacitive boost / D-LDO
    output logic [7:0]             current_demand_norm,// Normalized tile current demand (0 to 255) for PDN co-sim
    output logic [2:0]             mas_fsm_state,      // Telemetry status
    output logic [31:0]            total_droop_events  // Telemetry event counter
);

    // Internal interconnect signals
    logic [ROWS-1:0] row_enable_mask;
    logic            slack_warning;
    logic            slack_critical;
    logic            throttle_active;
    logic [4:0]      slack_margin;
    logic            array_error;

    // 1. Instantiate Real-Time Timing Slack & Droop Monitor
    pdn_slack_monitor #(
        .SENSOR_STAGES   (32),
        .WARNING_THRESH  (20),
        .CRITICAL_THRESH (14)
    ) u_slack_monitor (
        .clk             (clk),
        .rst_n           (rst_n),
        .enable          (1'b1),
        .sim_droop_code  (sim_droop_code),
        .slack_warning   (slack_warning),
        .slack_critical  (slack_critical),
        .slack_margin    (slack_margin),
        .droop_event_cnt (total_droop_events)
    );

    // 2. Instantiate Proactive Microarchitectural Activity Smoothing (MAS) Controller
    proactive_mas_ctrl #(
        .NUM_ROWS        (ROWS),
        .RAMP_CYCLES     (4)
    ) u_mas_ctrl (
        .clk                (clk),
        .rst_n              (rst_n),
        .gemm_imminent      (gemm_imminent),
        .gemm_execute       (gemm_execute),
        .slack_warning      (slack_warning),
        .slack_critical     (slack_critical),
        .row_enable_mask    (row_enable_mask),
        .charge_boost_pulse (charge_boost_pulse),
        .throttle_active    (throttle_active),
        .mas_state          (mas_fsm_state)
    );

    // 3. Instantiate Systolic Tensor Core Array with Dual-Slack FP8 MACs
    systolic_tile #(
        .ROWS      (ROWS),
        .COLS      (COLS),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_systolic_tile (
        .clk              (clk),
        .rst_n            (rst_n),
        .row_en           (row_enable_mask),
        .slack_degraded   (slack_critical),
        .a_in             (a_inputs),
        .b_in             (b_inputs),
        .c_in             (c_inputs),
        .c_out            (c_outputs),
        .valid_out        (results_valid),
        .array_error_flag (array_error)
    );

    assign system_error = array_error;

    // 4. On-Chip Current Telemetry Calculation
    // Estimates total instantaneous switching current for co-simulation with the PDN ODE solver
    always_comb begin
        int active_rows;
        active_rows = 0;
        for (int i = 0; i < ROWS; i++) begin
            if (row_enable_mask[i]) active_rows++;
        end

        // Baseline leakage / static current is ~15 (normalized out of 255)
        // Each active row adds ~55 (normalized)
        current_demand_norm = 8'd15 + (active_rows * 8'd55);

        // When auxiliary charge boost is pulsing, instantaneous charge demand is compensated
        if (charge_boost_pulse) begin
            // Boost pulse delivers charge, offsetting initial surge
            if (current_demand_norm > 8'd30)
                current_demand_norm = current_demand_norm - 8'd20;
        end
    end

endmodule
