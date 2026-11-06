//=============================================================================
// Module: proactive_mas_ctrl
// Project: FP8-TensorCore-PowerIntegrity - Workload-Aware Proactive dI/dt Smoothing & Undervolting
// Description: Microarchitectural Activity Smoothing (MAS) Controller.
//              Generates proactive ramp profiles and coordinates auxiliary charge
//              injection to suppress inductive L*(di/dt) voltage droop.
//=============================================================================

`timescale 1ns / 1ps

module proactive_mas_ctrl #(
    parameter NUM_ROWS      = 4,       // Systolic array rows
    parameter RAMP_CYCLES   = 4        // Staggered ramp-up cycle duration
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   gemm_imminent,     // 4-cycle lookahead hint from warp scheduler/instruction buffer
    input  logic                   gemm_execute,      // Execution phase active
    input  logic                   slack_warning,     // From PDN delay sensor
    input  logic                   slack_critical,    // From PDN delay sensor
    output logic [NUM_ROWS-1:0]    row_enable_mask,   // Per-row progressive clock gate enables
    output logic                   charge_boost_pulse,// Proactive charge pump / D-LDO boost trigger
    output logic                   throttle_active,   // Active droop damping asserted
    output logic [2:0]             mas_state          // FSM state for telemetry
);

    // MAS FSM States
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        PRE_BOOST   = 3'b001,  // Fire charge boost pulse 1 cycle ahead
        RAMPING_UP  = 3'b010,  // Staggered row activation to flatten di/dt
        FULL_EXEC   = 3'b011,  // Steady-state 100% activation
        DAMPING     = 3'b100   // Emergency transient damping upon critical slack warning
    } state_t;

    state_t state_reg, state_next;
    logic [2:0] ramp_counter;

    // FSM State Transition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg    <= IDLE;
            ramp_counter <= '0;
        end else begin
            state_reg <= state_next;
            if (state_reg == RAMPING_UP) begin
                ramp_counter <= ramp_counter + 1'b1;
            end else begin
                ramp_counter <= '0;
            end
        end
    end

    // Next-State Logic
    always_comb begin
        state_next = state_reg;
        case (state_reg)
            IDLE: begin
                if (gemm_imminent)
                    state_next = PRE_BOOST;
                else if (gemm_execute)
                    state_next = RAMPING_UP;
            end

            PRE_BOOST: begin
                state_next = RAMPING_UP;
            end

            RAMPING_UP: begin
                if (slack_critical)
                    state_next = DAMPING;
                else if (ramp_counter >= (RAMP_CYCLES - 1))
                    state_next = FULL_EXEC;
            end

            FULL_EXEC: begin
                if (!gemm_execute)
                    state_next = IDLE;
                else if (slack_critical)
                    state_next = DAMPING;
            end

            DAMPING: begin
                // Hold damping for 1 cycle to let PDN oscillation settle
                state_next = FULL_EXEC;
            end

            default: state_next = IDLE;
        endcase
    end

    // Output Generation
    always_comb begin
        mas_state          = state_reg;
        charge_boost_pulse = 1'b0;
        throttle_active    = 1'b0;
        row_enable_mask    = '0;

        case (state_reg)
            IDLE: begin
                row_enable_mask    = '0;
                charge_boost_pulse = 1'b0;
            end

            PRE_BOOST: begin
                row_enable_mask    = '0;
                charge_boost_pulse = 1'b1; // Pulse local auxiliary charge pump before current surge
            end

            RAMPING_UP: begin
                // Staggered activation of systolic array rows:
                // Cycle 0: Row 0 only (25% di/dt)
                // Cycle 1: Rows 0,1   (50% di/dt)
                // Cycle 2: Rows 0,1,2 (75% di/dt)
                // Cycle 3: All rows   (100%)
                case (ramp_counter)
                    3'd0:    row_enable_mask = 4'b0001;
                    3'd1:    row_enable_mask = 4'b0011;
                    3'd2:    row_enable_mask = 4'b0111;
                    default: row_enable_mask = 4'b1111;
                endcase
            end

            FULL_EXEC: begin
                row_enable_mask = {NUM_ROWS{1'b1}}; // All rows enabled
            end

            DAMPING: begin
                // Throttle upper 2 rows for 1 clock cycle to halt resonant LC overshoot
                row_enable_mask = 4'b0011;
                throttle_active = 1'b1;
            end

            default: begin
                row_enable_mask = '0;
            end
        endcase
    end

endmodule
