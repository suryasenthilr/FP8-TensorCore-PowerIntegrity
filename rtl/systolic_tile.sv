//=============================================================================
// Module: systolic_tile
// Project: FP8-TensorCore-PowerIntegrity - Workload-Aware Proactive dI/dt Smoothing & Undervolting
// Description: Parametric 4x4 Systolic Tensor Core Array for FP8 GEMM.
//              Integrates dual-slack FP8 MACs with per-row activity gating.
//=============================================================================

`timescale 1ns / 1ps

module systolic_tile #(
    parameter ROWS      = 4,
    parameter COLS      = 4,
    parameter ACC_WIDTH = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [ROWS-1:0]        row_en,            // From MAS controller: fine-grained per-row activity gate
    input  logic                   slack_degraded,    // From PDN delay monitor: droop slack violation flag
    input  logic [ROWS*8-1:0]          a_in,              // Flattened row activations (ROWS x 8-bit)
    input  logic [COLS*8-1:0]          b_in,              // Flattened col weights (COLS x 8-bit)
    input  logic [COLS*ACC_WIDTH-1:0]  c_in,              // Flattened acc inputs (COLS x ACC_WIDTH)
    output logic [COLS*ACC_WIDTH-1:0]  c_out,             // Flattened acc outputs
    output logic [COLS-1:0]            valid_out,
    output logic                       array_error_flag   // Asserted if any PE encountered MSB overflow
);

    // Internal interconnect registers for systolic dataflow
    logic [7:0] a_reg [ROWS-1:0][COLS:0];
    logic [7:0] b_reg [ROWS:0][COLS-1:0];
    logic [ACC_WIDTH-1:0] acc_reg [ROWS:0][COLS-1:0];
    logic pe_error [ROWS-1:0][COLS-1:0];
    logic pe_valid [ROWS-1:0][COLS-1:0];

    // Connect top boundary inputs
    genvar c;
    generate
        for (c = 0; c < COLS; c++) begin : gen_col_inputs
            assign b_reg[0][c]   = b_in[(c+1)*8-1 : c*8];
            assign acc_reg[0][c] = c_in[(c+1)*ACC_WIDTH-1 : c*ACC_WIDTH];
            assign c_out[(c+1)*ACC_WIDTH-1 : c*ACC_WIDTH] = acc_reg[ROWS][c];
            assign valid_out[c]  = pe_valid[ROWS-1][c];
        end
    endgenerate

    // Connect left boundary inputs
    genvar r;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row_inputs
            assign a_reg[r][0] = a_in[(r+1)*8-1 : r*8];
        end
    endgenerate

    // Instantiate 2D Grid of Dual-Slack FP8 PEs
    generate
        for (r = 0; r < ROWS; r++) begin : gen_grid_rows
            for (c = 0; c < COLS; c++) begin : gen_grid_cols
                
                // Pipeline registers for data propagation
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        a_reg[r][c+1] <= '0;
                        b_reg[r+1][c] <= '0;
                    end else if (row_en[r]) begin
                        a_reg[r][c+1] <= a_reg[r][c];
                        b_reg[r+1][c] <= b_reg[r][c];
                    end
                end

                // Dual-Slack FP8 MAC unit
                fp8_mac #(
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_pe_mac (
                    .clk            (clk),
                    .rst_n          (rst_n),
                    .enable         (row_en[r]),
                    .slack_degraded (slack_degraded),
                    .a_fp8          (a_reg[r][c]),
                    .b_fp8          (b_reg[r][c]),
                    .c_acc_in       (acc_reg[r][c]),
                    .acc_out        (acc_reg[r+1][c]),
                    .valid_out      (pe_valid[r][c]),
                    .error_flag     (pe_error[r][c])
                );

            end
        end
    endgenerate

    // Aggregate error flags across the systolic array via flattened reduction
    logic [ROWS*COLS-1:0] pe_error_flat;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_err_rows
            for (c = 0; c < COLS; c++) begin : gen_err_cols
                assign pe_error_flat[r*COLS + c] = pe_error[r][c];
            end
        end
    endgenerate

    assign array_error_flag = |pe_error_flat;

endmodule
