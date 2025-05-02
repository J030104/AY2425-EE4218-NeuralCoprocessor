`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2025 01:37:47
// Design Name: 
// Module Name: matrix_multiply
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
// NOT THIS ONE
module matrix_multiply #(
    parameter DATA_WIDTH = 8,
    parameter A_DEPTH_BITS = 9,   // e.g., log2 of total A matrix entries (64*7), (64*2)
    parameter A_entries = 448,    // 64*7
    parameter M = 7,              // M = # of columns of A = # of rows of B
    parameter B_DEPTH_BITS = 4,   // e.g., log2 of B vector length        (8*2),  (3*1)
    parameter RES_DEPTH_BITS = 7  //                                      (64*2), (64*1)
)(
    input  wire                   clk,
    input  wire                   Start,
    output reg                    Done,
    input wire                    Sigmoid_en,
    // Interface to A (matrix) memory
    output reg                    A_read_en,
    output reg [A_DEPTH_BITS-1:0] A_read_address,
    input  wire [DATA_WIDTH-1:0]  A_read_data_out,
    // Interface to B (vector) memory
    output reg                    B_read_en,
    output reg [B_DEPTH_BITS-1:0] B_read_address,
    input  wire [DATA_WIDTH-1:0]  B_read_data_out,
    // Interface to result (optional, can be unused if capturing differently)
    output reg                    RES_write_en,
    output reg [RES_DEPTH_BITS-1:0] RES_write_address,
    output reg [DATA_WIDTH-1:0]   RES_write_data_in
);

    // Internal derived constants
    localparam B_LEN    = 2 ** B_DEPTH_BITS;                // number of elements in B vector
    localparam NUM_ROWS = (2 ** A_DEPTH_BITS) / B_LEN;      // number of rows in A (and outputs)
    
    localparam [1:0] S_IDLE     = 2'b00, S_CALC = 2'b01, S_STORE = 2'b11;
    reg [1:0] state = S_IDLE;
    
    // Accumulator for dot-product and loop counter
    reg [15:0] acc;              // 16-bit accumulator for sums (supports up to 8-bit*8-bit*256)
    reg [B_DEPTH_BITS:0] idx;    // counter for elements in a row (goes 0 to B_LEN)
    reg [A_DEPTH_BITS-1:0] base_addr; // base address of current row in A
    reg [A_DEPTH_BITS-1:0] out_index; // index of current output (row index)

    LUTenable = 1'b0;

    // Default outputs
    always @(posedge clk) begin
        if (Start) begin
            // Initialize for first computation
            state          <= S_CALC;
            A_read_en      <= 1'b1;
            B_read_en      <= 1'b1;
            A_read_address <= 0;  // start at 0
            B_read_address <= {B_DEPTH_BITS{1'b0}};  // start at 0
            base_addr      <= 0;
            out_index      <= 0;
            idx            <= 0;
            acc            <= 16'd0;
            Done           <= 1'b0;
            RES_write_en   <= 1'b0;
        end else begin
            // Default signals each cycle
            A_read_en    <= 1'b0;
            B_read_en    <= 1'b0;
            RES_write_en <= 1'b0;
            Done         <= 1'b0;
            case (state)
            S_CALC: begin
                // Perform multiply-accumulate
                acc <= acc + (A_read_data_out * B_read_data_out);
                // Advance to next element in row
                if (idx < B_LEN - 1) begin
                    idx <= idx + 1;
                    A_read_en <= 1'b1;
                    B_read_en <= 1'b1;
                    A_read_address <= base_addr + (idx + 1);
                    B_read_address <= idx + 1;
                end else begin
                    // Reached end of row
                    state <= S_STORE;
                    // Prepare to store result of this row
                    RES_write_en <= 1'b1;
                    RES_write_address <= out_index;
                    // Output the high 8 bits of acc (>>8) as result (lower bits are fractional part)
                    RES_write_data_in <= acc[15:8];
                end
            end
            S_STORE: begin
                // One output written. Prepare next row if any remain.
                if (out_index < NUM_ROWS - 1) begin
                    // Move to next row
                    out_index  <= out_index + 1;
                    base_addr  <= base_addr + B_LEN;
                    // Reset for next accumulation
                    idx        <= 0;
                    acc        <= 16'd0;
                    // Issue reads for next row's first element
                    A_read_en  <= 1'b1;
                    B_read_en  <= 1'b1;
                    A_read_address <= base_addr + B_LEN;  // base_addr was old row; add B_LEN to jump to next row start
                    B_read_address <= 0;
                    state      <= S_CALC;
                end else begin
                    // All rows processed
                    Done  <= 1'b1;
                    state <= S_IDLE;
                end
            end
            default: state <= S_IDLE;
            endcase
        end // not Start
    end // always
endmodule
