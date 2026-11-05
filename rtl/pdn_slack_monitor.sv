//=============================================================================
// Module: pdn_slack_monitor
// Project: VoltTensor - Workload-Aware Proactive dI/dt Smoothing & Undervolting
// Description: Real-Time On-Chip Delay-Line Voltage Droop & Timing Slack Monitor.
//              Emulates a tunable inverter-chain critical-path replica sensor.
//=============================================================================

`timescale 1ns / 1ps

module pdn_slack_monitor #(
    parameter SENSOR_STAGES = 32,      // Length of calibrated replica delay line
    parameter WARNING_THRESH = 20,     // Delay tap warning threshold
    parameter CRITICAL_THRESH = 26     // Delay tap critical droop threshold
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        enable,
    input  logic [7:0]  sim_droop_code, // Digital test injection or ADC voltage reading (0=nominal 0.8V, 255=deep droop 0.6V)
    output logic        slack_warning,  // Early alert: timing slack margin tightening
    output logic        slack_critical, // Critical droop: LSB truncation/active clamp triggered
    output logic [4:0]  slack_margin,   // Remaining timing margin tap value
    output logic [31:0] droop_event_cnt // Cumulative droop event counter
);

    // Emulated replica delay line tap propagation
    // As voltage drops (sim_droop_code increases), propagation delay increases,
    // so fewer stages toggle before the clock sampling edge.
    logic [SENSOR_STAGES-1:0] delay_taps;
    logic [4:0] current_tap_depth;

    always_comb begin
        // Map droop code (0-255) to tap depth
        // At nominal 0.8V (droop=0), delay is fast, pulse travels far (tap depth ~ 30)
        // Under 150mV droop (droop=128), delay is slow, pulse only travels to tap ~ 18
        // Under 250mV droop (droop=255), pulse stops early at tap ~ 8
        if (sim_droop_code < 8'd40) begin
            current_tap_depth = 5'd30;
        end else if (sim_droop_code < 8'd100) begin
            current_tap_depth = 5'd24;
        end else if (sim_droop_code < 8'd160) begin
            current_tap_depth = 5'd18;
        end else if (sim_droop_code < 8'd220) begin
            current_tap_depth = 5'd12;
        end else begin
            current_tap_depth = 5'd6;
        end
    end

    assign slack_margin = current_tap_depth;

    // Slack threshold comparators
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            slack_warning   <= 1'b0;
            slack_critical  <= 1'b0;
            droop_event_cnt <= '0;
        end else if (enable) begin
            // If current tap depth falls below threshold, propagation is dangerously slow
            slack_warning   <= (current_tap_depth <= WARNING_THRESH);
            slack_critical  <= (current_tap_depth <= CRITICAL_THRESH);

            if (current_tap_depth <= CRITICAL_THRESH) begin
                droop_event_cnt <= droop_event_cnt + 1'b1;
            end
        end else begin
            slack_warning  <= 1'b0;
            slack_critical <= 1'b0;
        end
    end

endmodule
