`timescale 1ns / 1ps

// 64x7 and 8x2 => 64x2
// 64x2 and 3*1 => 64x1

module neural_network_v1_0 #(
    parameter DATA_WIDTH = 8,
    parameter N = 64,             // 64
    parameter FEATURES = 7,       // M = # of columns of A = # of rows of B
    parameter HIDDEN = 2,
    parameter A_DEPTH_BITS = 9,   // e.g., log2 of total A matrix entries (64*7)
    parameter B_DEPTH_BITS = 4,   // e.g., log2 of B matrix entries        (8*2)
    parameter C_DEPTH_BITS = 2,   // e.g., log2 of C matrix entries        (3*1)
    parameter RES_DEPTH_BITS = 7  //                                      (64*1)
) (
    input  wire                   clk,
    input  wire                   Start,
    output reg                    Done,
    // Interface to the Sigmoid function
    output reg                    Sigmoid_en_1,
    output reg [DATA_WIDTH-1:0]   Sigmoid_lookup_1,
    input wire [DATA_WIDTH-1:0]   Sigmoid_result_1,
    output reg                    Sigmoid_en_2,
    output reg [DATA_WIDTH-1:0]   Sigmoid_lookup_2,
    input wire [DATA_WIDTH-1:0]   Sigmoid_result_2,
    // Interface to A (matrix) memory (X matrix)
    output reg                    A_read_en,
    output reg [A_DEPTH_BITS-1:0] A_read_address,
    input  wire [DATA_WIDTH-1:0]  A_read_data_out,
    // Interface to B (matrix) memory (w_hid matrix)
    output reg                    B_read_en_1,
    output reg [B_DEPTH_BITS-1:0] B_read_address_1,
    input  wire [DATA_WIDTH-1:0]  B_read_data_out_1,
    output reg                    B_read_en_2,
    output reg [B_DEPTH_BITS-1:0] B_read_address_2,
    input  wire [DATA_WIDTH-1:0]  B_read_data_out_2,
    // Interface to C (matrix) memory (w_out matrix)
    output reg                    C_read_en_1,
    output reg [C_DEPTH_BITS-1:0] C_read_address_1,
    input  wire [DATA_WIDTH-1:0]  C_read_data_out_1,
    output reg                    C_read_en_2,
    output reg [C_DEPTH_BITS-1:0] C_read_address_2,
    input  wire [DATA_WIDTH-1:0]  C_read_data_out_2,
    // Interface to result
    output reg                      RES_write_en,
    output reg [RES_DEPTH_BITS-1:0] RES_write_address,
    output reg [DATA_WIDTH-1:0]     RES_write_data_in
);

    localparam W_HID_COUNT    = (FEATURES + 1) * HIDDEN;     // total hidden layer weights (including bias for each neuron)
    localparam W_OUT_COUNT    = HIDDEN + 1;                  // total output layer weights (including bias)
    
    localparam Idle            = 8'b1000_0000;
    localparam Initialize      = 8'b0100_0000;
    localparam Wait_Initialize = 8'b0010_0000;
    localparam Compute_Hidden  = 8'b0001_0000;
    localparam Wait_Hidden     = 8'b0000_1000;
    localparam Compute_Output  = 8'b0000_0100;
    localparam Wait_Output     = 8'b0000_0010;
    localparam Store           = 8'b0000_0001;

    reg [7:0] state = Idle;

    reg [5:0] row;   // 0~63 (64 rows, 64 big loops)
    reg [2:0] col;   // 0~6  (at most 7 columns in the first stage)
    reg lookup_finished;
    reg [15:0] acc [0:HIDDEN-1]; // Two more bits to ensure no overflow occurs

    reg [1:0] cnt;
    always @(posedge clk) begin
        if (Start) begin
            A_read_en    <= 1'b1;
            B_read_en_1    <= 1'b1; B_read_en_2    <= 1'b1;
            C_read_en_1    <= 1'b1; C_read_en_2    <= 1'b1;
            A_read_address   <= 0; 
            B_read_address_1 <= 0; B_read_address_2 <= 1;
            C_read_address_1 <= 0; C_read_address_2 <= 2;

            Sigmoid_en_1 <= 1'b1; Sigmoid_en_2 <= 1'b1;
            RES_write_en <= 1'b0;
            
            Done         <= 1'b0;
            state        <= Initialize;
            cnt          <= 0;
            acc[0]       <= 0;
            acc[1]       <= 0;
        end else begin
            // Default signals each cycle
            Done         <= 1'b0;
            RES_write_en <= 1'b0;

            case (state)
                // Will enter before Start signal and stay in here until Start signal
                Idle: begin
                    A_read_address   <= 0; 
                    B_read_address_1 <= 0; B_read_address_2 <= 1;
                    C_read_address_1 <= 0; C_read_address_2 <= 2;
                    row              <= 0;
                    col              <= 0;
                    lookup_finished  <= 0;
                    acc[0]           <= 0;
                    acc[1]           <= 0;
                end

                Initialize: begin
                    // Bias term for each neuron
                    // acc[0]           <= {B_read_data_out_1, 8'b0};
                    // acc[1]           <= {B_read_data_out_2, 8'b0};
                    acc[0]           <= (B_read_data_out_1 << 8);
                    acc[1]           <= (B_read_data_out_2 << 8);
                    B_read_address_1 <= B_read_address_1 + 2;
                    B_read_address_2 <= B_read_address_2 + 2;

                    state <= Wait_Initialize;
                end
                
                Wait_Initialize: begin
                    state <= Compute_Hidden;
                end
                
                Compute_Hidden: begin
                    acc[0] <= acc[0] + A_read_data_out * B_read_data_out_1;
                    acc[1] <= acc[1] + A_read_data_out * B_read_data_out_2;
                    if (col == FEATURES - 1) begin
                        col <= 0;
                        state            <= Wait_Hidden;
                    end else begin
                        col              <= col + 1;
                        A_read_address   <= A_read_address + 1;
                        B_read_address_1 <= B_read_address_1 + 2;
                        B_read_address_2 <= B_read_address_2 + 2;
                        state            <= Compute_Hidden; // Keep accumulating
                    end
                end

                Wait_Hidden: begin
                    lookup_finished  <= 1'b0;
                    C_read_address_1 <= 0;
                    state <= Compute_Output;
                end
                
                Compute_Output: begin
                    // Sigmoid Activation function
                    if (lookup_finished == 0) begin    
                        Sigmoid_lookup_1 <= acc[0][15:8];
                        Sigmoid_lookup_2 <= acc[1][15:8]; // Overflow should be taken care of 
                        // acc[0]           <= {C_read_data_out_1, 8'b0};
                        acc[0]           <= (C_read_data_out_1 << 8);
                        acc[1]           <= 0;
                        C_read_address_1 <= 1;
                        C_read_address_2 <= 2;
                        lookup_finished  <= 1'b1;
                        cnt              <= 0;
                        state            <= Compute_Output; 
                    end else if (cnt == 1) begin
                    // end else begin
                        // Timing issue?
                        acc[0] <= acc[0] + C_read_data_out_1 * Sigmoid_result_1
                                        + C_read_data_out_2 * Sigmoid_result_2;
                        cnt <= 0;
                        state <= Wait_Output;
                    end else begin
                        // WAIT cycle for ROM and C_read_data to be valid
                        cnt <= cnt + 1;
                        state <= Compute_Output;
                    end 
                end

                Wait_Output: begin
                    // acc[0] <= (acc[0] >> 8);
                    RES_write_address <= row;
                    state <= Store;
                end
                
                Store: begin
                    RES_write_en      <= 1'b1;
                    RES_write_data_in <= (acc[0][15:8] > 8'b1000_0000) ? 1'b1 : 1'b0;
                    
                    if (row == N - 1) begin
                        Done <= 1'b1;
                        state <= Idle;
                    end else begin
                        acc[0]            <= 0;
                        acc[1]            <= 0;
                        row               <= row + 1;
                        A_read_address    <= A_read_address + 1;
                        B_read_address_1  <= 0;
                        B_read_address_2  <= 1;
                        state             <= Initialize;
                    end
                end
                
                default: state <= Idle;
            endcase
        end // not Start
    end // always
endmodule