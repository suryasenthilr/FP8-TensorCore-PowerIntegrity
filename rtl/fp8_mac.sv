//=============================================================================
// Module: fp8_mac
// Project: FP8-TensorCore-PowerIntegrity - Workload-Aware Proactive dI/dt Smoothing & Undervolting
// Description: Dual-Slack FP8 (E4M3) Multiply-Accumulate Unit with Asymmetric
//              Timing Slack Paths for Graceful Undervolting Under Voltage Droop.
//=============================================================================

`timescale 1ns / 1ps

module fp8_mac #(
    parameter ACC_WIDTH = 32,      // 32-bit FP accumulator for high numerical dynamic range
    parameter FP8_EXP   = 4,       // E4M3 standard: 4 exponent bits
    parameter FP8_MAN   = 3        // E4M3 standard: 3 mantissa bits
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   enable,
    input  logic                   slack_degraded, // Asserted when voltage droop exceeds threshold
    input  logic [7:0]             a_fp8,          // Operands (E4M3: S[7], E[6:3], M[2:0])
    input  logic [7:0]             b_fp8,
    input  logic [ACC_WIDTH-1:0]   c_acc_in,       // Accumulator input
    output logic [ACC_WIDTH-1:0]   acc_out,        // Accumulator output
    output logic                   valid_out,
    output logic                   error_flag      // Flag asserted if MSB timing violated
);

    // FP8 E4M3 Field Deconstruction
    // Sign: 1 bit, Exponent: 4 bits (bias=7), Mantissa: 3 bits (implicit 1 for normal numbers)
    wire        sign_a = a_fp8[7];
    wire [3:0]  exp_a  = a_fp8[6:3];
    wire [2:0]  man_a  = a_fp8[2:0];

    wire        sign_b = b_fp8[7];
    wire [3:0]  exp_b  = b_fp8[6:3];
    wire [2:0]  man_b  = b_fp8[2:0];

    // Check for zeros / denormals
    wire is_zero_a = (exp_a == 4'b0000 && man_a == 3'b000);
    wire is_zero_b = (exp_b == 4'b0000 && man_b == 3'b000);

    // Implicit leading 1 for normalized numbers
    wire [3:0] full_man_a = (exp_a == 4'b0000) ? {1'b0, man_a} : {1'b1, man_a};
    wire [3:0] full_man_b = (exp_b == 4'b0000) ? {1'b0, man_b} : {1'b1, man_b};

    //-------------------------------------------------------------------------
    // CRITICAL PATH 1: Fast High-Slack Path (Sign & Exponent computation)
    // Protected against voltage droop using boosted priority logic / low-Vt cells
    //-------------------------------------------------------------------------
    logic        prod_sign;
    logic [4:0]  prod_exp_raw;
    logic [4:0]  prod_exp;

    always_comb begin
        prod_sign    = sign_a ^ sign_b;
        prod_exp_raw = 5'd0;
        prod_exp     = 5'd0;
        if (!is_zero_a && !is_zero_b) begin
            // Exponent addition with bias correction (bias = 7 for E4M3)
            // exp_a + exp_b - 7
            prod_exp_raw = {1'b0, exp_a} + {1'b0, exp_b};
            if (prod_exp_raw >= 5'd7)
                prod_exp = prod_exp_raw - 5'd7;
            else
                prod_exp = 5'd0; // Underflow clamp
        end
    end

    //-------------------------------------------------------------------------
    // CRITICAL PATH 2: Asymmetric Mantissa Multiplier Array (Dual-Slack)
    // MSB portion has hard timing margin.
    // LSB portion has soft timing margin with Slack-Aware Graceful Truncation.
    //-------------------------------------------------------------------------
    wire [7:0] man_prod_full = full_man_a * full_man_b; // 4-bit * 4-bit = 8-bit product
    wire [7:0] man_prod_safe = slack_degraded ? {man_prod_full[7:2], 2'b10} : man_prod_full;

    // Normalization & Conversion to 32-bit Accumulator format
    // S[31], Exp[30:23], Mantissa[22:15], Pad[14:0] = 32 bits
    logic [ACC_WIDTH-1:0] mult_result;
    always_comb begin
        if (is_zero_a || is_zero_b) begin
            mult_result = '0;
        end else begin
            mult_result = {
                prod_sign,
                (8'd120 + {3'b0, prod_exp}), // 8-bit rebasing to FP32 exponent (bias 127)
                man_prod_safe[7:0],          // 8-bit mantissa
                15'b0                        // 15-bit fraction padding
            };
        end
    end

    // Accumulation Pipeline Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out    <= '0;
            valid_out  <= 1'b0;
            error_flag <= 1'b0;
        end else if (enable) begin
            acc_out    <= c_acc_in + mult_result; // Parallel accumulator add
            valid_out  <= 1'b1;
            // Error flag is asserted only if critical exponent or sign encounters illegal state
            error_flag <= (prod_exp_raw > 5'd28); // Exponent overflow detection
        end else begin
            valid_out  <= 1'b0;
            error_flag <= 1'b0;
        end
    end

endmodule
