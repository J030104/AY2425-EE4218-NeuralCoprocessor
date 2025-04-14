`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2025 01:37:47
// Design Name: 
// Module Name: MLP_IP
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

module MLP_IP #(
    // Parameter definitions for input and network sizes
    parameter N_BITS        = 6,  // Address bits for N input samples (2^N_BITS = N = 64)
    parameter FEATURES_BITS = 3,  // Address bits for number of features (2^FEATURES_BITS = FEATURES) (7 features but 3 bits required)
    parameter HIDDEN_BITS   = 1,  // Address bits for number of hidden neurons (2^HIDDEN_BITS = HIDDEN = 2)
    parameter TDATA_WIDTH   = 32 // AXI Stream data width (bits)
) (
    ACLK,
    ARESETN,
    S_AXIS_TREADY,
    S_AXIS_TDATA,
    S_AXIS_TLAST,
    S_AXIS_TVALID,
    M_AXIS_TVALID,
    M_AXIS_TDATA,
    M_AXIS_TLAST,
    M_AXIS_TREADY
);
    input wire                     ACLK;
    input wire                     ARESETN;    // Active-low reset
    // AXI-Stream Slave (input) interface
    output reg                     S_AXIS_TREADY;
    input wire [TDATA_WIDTH-1:0]   S_AXIS_TDATA;
    input wire                     S_AXIS_TLAST;
    input wire                     S_AXIS_TVALID;
    // AXI-Stream Master (output) interface
    output reg                     M_AXIS_TVALID;
    output reg  [TDATA_WIDTH-1:0]  M_AXIS_TDATA;
    output reg                     M_AXIS_TLAST;
    input wire                     M_AXIS_TREADY;

    localparam DATA_WIDTH = 8; // Bit-width of data values (e.g., 8-bit unsigned)
    
    localparam LUT_SIZE   = 256;   // Sigmoid function LUT entries (256 entries for 8-bit input)

    // Derived constants
    localparam N                  = 2 ** N_BITS;
    localparam FEATURES           = 2 ** FEATURES_BITS - 1;      // number of features per sample (bias is handled separately)
    localparam HIDDEN             = 2 ** HIDDEN_BITS;            // number of hidden layer neurons
    localparam W_HID_COUNT        = (FEATURES + 1) * HIDDEN;     // total hidden layer weights (including bias for each neuron)
    localparam W_OUT_COUNT        = HIDDEN + 1;                  // total output layer weights (including bias)
    localparam TOTAL_INPUT_COUNT  = N * FEATURES + W_HID_COUNT + W_OUT_COUNT;
    localparam TOTAL_OUTPUT_COUNT = N;

    // State encoding (one-hot FSM for clarity). In fact this can be reduced to 4 bits
    localparam ST_IDLE        = 4'b1000;
    localparam ST_READ_INPUTS = 4'b0100;
    localparam ST_COMPUTE     = 4'b0010;   
    localparam ST_WRITE_OUT   = 4'b0001;
    reg [3:0] state;

    reg [1:0] RES_fetch;  // signals that we are currently fetching from RES;

    // BRAM interface signals
    // Input data memory (X_RAM): stores N * FEATURES values (2^N_BITS * 2^FEATURES_BITS = 2^(N_BITS + FEATURES_BITS))
    reg                       X_write_en;
    reg  [N_BITS+FEATURES_BITS-1:0] X_write_addr;
    reg  [DATA_WIDTH-1:0]     X_write_data;
    wire                       X_read_en;
    wire  [N_BITS+FEATURES_BITS-1:0] X_read_addr;
    wire [DATA_WIDTH-1:0]     X_read_data;
    // Hidden weights memory (W_hid_RAM): stores W_HID_COUNT values
    reg                       Wh_write_en;
    reg  [$clog2(W_HID_COUNT)-1:0]    Wh_write_addr;
    reg  [DATA_WIDTH-1:0]     Wh_write_data;
    wire                      Wh_read_en_1;
    wire [$clog2(W_HID_COUNT)-1:0]    Wh_read_addr_1;
    wire [DATA_WIDTH-1:0]     Wh_read_data_1;
    wire                      Wh_read_en_2;
    wire [$clog2(W_HID_COUNT)-1:0]    Wh_read_addr_2;
    wire [DATA_WIDTH-1:0]     Wh_read_data_2;
    // Output weights memory (W_out_RAM): stores W_OUT_COUNT values
   (* keep = "true" *) reg                       Wo_write_en;
   (* keep = "true" *) reg  [$clog2(W_OUT_COUNT)-1:0]    Wo_write_addr;
   (* keep = "true" *) reg  [DATA_WIDTH-1:0]     Wo_write_data;
   (* keep = "true" *) wire                      Wo_read_en_1;
   (* keep = "true" *) wire [$clog2(W_OUT_COUNT)-1:0]    Wo_read_addr_1;
   (* keep = "true" *) wire [DATA_WIDTH-1:0]     Wo_read_data_1;
   (* keep = "true" *) wire                      Wo_read_en_2;
   (* keep = "true" *) wire [$clog2(W_OUT_COUNT)-1:0]    Wo_read_addr_2;
   (* keep = "true" *) wire [DATA_WIDTH-1:0]     Wo_read_data_2;
    // Result memory (RES_RAM): stores N results (predictions)
    wire                     RES_write_en;
    wire  [N_BITS-1:0]       RES_write_addr;
    wire  [DATA_WIDTH-1:0]   RES_write_data;
    reg                     RES_read_en;
    reg  [N_BITS-1:0]       RES_read_addr;
    wire [DATA_WIDTH-1:0]   RES_read_data;
    // Sigmoid LUT ROM
    wire sigmoid_en_1;
    wire sigmoid_en_2;
    wire [$clog2(LUT_SIZE)-1:0]   sigmoid_in_1;
    wire [$clog2(LUT_SIZE)-1:0]   sigmoid_in_2;
    wire [DATA_WIDTH-1:0]        sigmoid_out_1;
    wire [DATA_WIDTH-1:0]        sigmoid_out_2;

    // Counters for reading input stream data
    reg [$clog2(N*FEATURES)-1:0] count_in_X;   // counter for X values
    reg [$clog2(W_HID_COUNT)-1:0]  count_in_Wh;  // counter for hidden weights
    reg [$clog2(W_OUT_COUNT)-1:0]  count_in_Wo;  // counter for output weights

    // Control signals between FSM and computation modules
    reg Start; 
    wire Done;

    // Instantiate internal BRAMs using the RAM modules
    memory_RAM #(.width(DATA_WIDTH), .depth_bits(N_BITS + FEATURES_BITS)) X_RAM (
        .clk(ACLK),
        .write_en(X_write_en), .write_address(X_write_addr), .write_data_in(X_write_data),
        .read_en(X_read_en),   .read_address(X_read_addr),   .read_data_out(X_read_data)
    );
    memory_RAM_d_r #(.width(DATA_WIDTH), .depth_bits($clog2(W_HID_COUNT))) Wh_RAM (
        .clk(ACLK),
        .write_en(Wh_write_en),     .write_address(Wh_write_addr),     .write_data_in(Wh_write_data),
        .read_en_1(Wh_read_en_1),   .read_address_1(Wh_read_addr_1),   .read_data_out_1(Wh_read_data_1),
        .read_en_2(Wh_read_en_2),   .read_address_2(Wh_read_addr_2),   .read_data_out_2(Wh_read_data_2)
    );
    memory_RAM_d_r #(.width(DATA_WIDTH), .depth_bits($clog2(W_OUT_COUNT))) Wo_RAM (
        .clk(ACLK),
        .write_en(Wo_write_en),     .write_address(Wo_write_addr),     .write_data_in(Wo_write_data),
        .read_en_1(Wo_read_en_1),   .read_address_1(Wo_read_addr_1),   .read_data_out_1(Wo_read_data_1),
        .read_en_2(Wo_read_en_2),   .read_address_2(Wo_read_addr_2),   .read_data_out_2(Wo_read_data_2)
    );
    memory_RAM #(.width(DATA_WIDTH), .depth_bits(N_BITS)) RES_RAM (
        .clk(ACLK),
        .write_en(RES_write_en), .write_address(RES_write_addr), .write_data_in(RES_write_data),
        .read_en(RES_read_en),   .read_address(RES_read_addr),   .read_data_out(RES_read_data)
    );

    ROM256 SigmoidLUT (
        .clk(ACLK),
        .enable_1(sigmoid_en_1),
        .enable_2(sigmoid_en_2),
        .address_1(sigmoid_in_1),
        .address_2(sigmoid_in_2),
        .data_1(sigmoid_out_1),
        .data_2(sigmoid_out_2)
    );

    // Instantiate matrix multiplication modules for hidden and output layers
    neural_network_v1_0 #(
        .DATA_WIDTH(DATA_WIDTH),
        .N(N),
        .FEATURES(FEATURES),
        .HIDDEN(HIDDEN),
        .A_DEPTH_BITS(N_BITS+FEATURES_BITS), // X
        .B_DEPTH_BITS($clog2(W_HID_COUNT)), // w_hid
        .C_DEPTH_BITS($clog2(W_OUT_COUNT)), // w_out
        .RES_DEPTH_BITS(N_BITS)
    ) nn_v1_0 (
        .clk(ACLK), .Start(Start), .Done(Done),
        .Sigmoid_en_1(sigmoid_en_1), .Sigmoid_lookup_1(sigmoid_in_1), .Sigmoid_result_1(sigmoid_out_1),
        .Sigmoid_en_2(sigmoid_en_2), .Sigmoid_lookup_2(sigmoid_in_2), .Sigmoid_result_2(sigmoid_out_2),
        .A_read_en(X_read_en), .A_read_address(X_read_addr), .A_read_data_out(X_read_data),
        .B_read_en_1(Wh_read_en_1), .B_read_address_1(Wh_read_addr_1), .B_read_data_out_1(Wh_read_data_1),
        .B_read_en_2(Wh_read_en_2), .B_read_address_2(Wh_read_addr_2), .B_read_data_out_2(Wh_read_data_2),
        .C_read_en_1(Wo_read_en_1), .C_read_address_1(Wo_read_addr_1), .C_read_data_out_1(Wo_read_data_1),
        .C_read_en_2(Wo_read_en_2), .C_read_address_2(Wo_read_addr_2), .C_read_data_out_2(Wo_read_data_2),
        .RES_write_en(RES_write_en), .RES_write_address(RES_write_addr), .RES_write_data_in(RES_write_data)
    );

    // // Register for sigmoid LUT (256 entries)
    // reg [7:0] sigmoid_LUT [0:255];
    // initial begin
    //     // Initialize the sigmoid lookup table (same values as in HLS code)
    //     sigmoid_LUT[  0] = 8'd12;  sigmoid_LUT[  1] = 8'd12;  // ... (omitted for brevity)
    //     // [Initialize all values 0..255 accordingly]
    //     sigmoid_LUT[252] = 8'd243; sigmoid_LUT[253] = 8'd243; sigmoid_LUT[254] = 8'd243; sigmoid_LUT[255] = 8'd243;
    // end

    reg [1:0] read_state = 2'b00;

    // Active-low synchronous reset and FSM
    always @(posedge ACLK) begin
        if (!ARESETN) begin
            // Reset: initialize FSM and control signals
            state         <= ST_IDLE;
            S_AXIS_TREADY <= 1'b0;
            M_AXIS_TVALID <= 1'b0;
            M_AXIS_TLAST  <= 1'b0;
            
            Start <= 0;
            X_write_en <= 0; Wh_write_en <= 0; Wo_write_en <= 0;
            count_in_X <= 0; count_in_Wh <= 0; count_in_Wo <= 0;
            
        end else begin
            // Default de-assert memory write enables each cycle (except when writing)
            X_write_en <= 0; Wh_write_en <= 0; Wo_write_en <= 0;
            X_write_addr <= X_write_addr;
            Wh_write_addr <= Wh_write_addr;
            Wo_write_addr <= Wo_write_addr;
            X_write_data <= 0;
            Wh_write_data <= 0;
            Wo_write_data <= 0;
            // Default control signals
            Start <= 0;

            case (state)
                //-------------------------------------
                ST_IDLE: begin
                    // Wait for input stream to start
                    M_AXIS_TVALID <= 1'b0;
                    M_AXIS_TLAST  <= 1'b0;
                    count_in_X    <= 0;
                    count_in_Wh    <= 0;
                    count_in_Wo    <= 0;
                    if (S_AXIS_TVALID == 1) begin
                        state         <= ST_READ_INPUTS;
                        S_AXIS_TREADY <= 1'b1;
                        // Start receiving data
                    end
                end
                //-------------------------------------
                ST_READ_INPUTS: begin
                    // Accept all TOTAL_INPUT_COUNT words from S_AXIS
                    if (S_AXIS_TVALID && S_AXIS_TREADY) begin
                        // Latch incoming data byte (LSB of TDATA)
                        // Determine which memory to write based on counters
//                        if (count_in_X < N * FEATURES) begin
//                            // Write to X_RAM
//                            X_write_en   <= 1'b1;
//                            X_write_addr <= count_in_X;
//                            X_write_data <= S_AXIS_TDATA[7:0];
//                            count_in_X   <= count_in_X + 1;
//                        end else if (count_in_Wh < W_HID_COUNT) begin
//                            // Write to Wh_RAM (hidden weights)
//                            Wh_write_en   <= 1'b1;
//                            Wh_write_addr <= count_in_Wh;
//                            Wh_write_data <= S_AXIS_TDATA[7:0];
//                            count_in_Wh   <= count_in_Wh + 1;
//                        end else if (count_in_Wo < W_OUT_COUNT) begin
//                            // Write to Wo_RAM (output weights)
//                            Wo_write_en   <= 1'b1;
//                            Wo_write_addr <= count_in_Wo;
//                            Wo_write_data <= S_AXIS_TDATA[7:0];
//                            count_in_Wo   <= count_in_Wo + 1;
//                        end
                        case (read_state)
                            2'b00: begin
                                // Write to X_RAM
                                X_write_en   <= 1'b1;
                                X_write_addr <= count_in_X;
                                X_write_data <= S_AXIS_TDATA[7:0];
                                if (count_in_X == N * FEATURES - 1) begin
                                    read_state <= read_state + 1;
                                end else begin
                                    count_in_X   <= count_in_X + 1;
                                end
                            
                            end
                            
                            2'b01: begin
                                // Write to Wh_RAM (hidden weights)
                                Wh_write_en   <= 1'b1;
                                Wh_write_addr <= count_in_Wh;
                                Wh_write_data <= S_AXIS_TDATA[7:0];
                                if (count_in_Wh == W_HID_COUNT - 1) begin
                                    read_state <= read_state + 1;
                                end else begin
                                    count_in_Wh <= count_in_Wh + 1;
                                end
                            end
                            
                            2'b10: begin
                                // Write to Wo_RAM (output weight
                                Wo_write_en   <= 1'b1;           
                                Wo_write_addr <= count_in_Wo;    
                                Wo_write_data <= S_AXIS_TDATA[7:0];
                                if (count_in_Wo == N * FEATURES - 1) begin
                                    read_state <= read_state + 1;
                                end else begin
                                    count_in_Wo <= count_in_Wo + 1;
                                end
                            end
                        endcase
    
                        // ? Should check if the TLAST can cause error
                        // Check if we reached the end of all inputs
                        // When the counters reach their limits, we know the transmission is complete.
//                        if ((count_in_X == N * FEATURES) && (count_in_Wh == W_HID_COUNT) && (count_in_Wo == W_OUT_COUNT) && S_AXIS_TLAST) begin
                        if (read_state == 2'b11) begin
                            S_AXIS_TREADY <= 1'b0;
                            Start <= 1'b1;
                            state <= ST_COMPUTE;
                        end
                    end
                end
                //-------------------------------------
                ST_COMPUTE: begin
                    if (Done == 1) begin
                        M_AXIS_TVALID <= 0;
                        RES_read_en   <= 1;
                        RES_read_addr <= 0;
                        RES_fetch     <= 1;
                        state         <= ST_WRITE_OUT;
                    end
                end
                //-------------------------------------
                ST_WRITE_OUT: begin
                    if (RES_fetch == 0) begin
                        M_AXIS_TVALID <= 0;
                        // Fetch element from result RAM
                        RES_read_en <= 1;
                        RES_fetch <= 1;
                    end else if (RES_fetch == 1) begin  // Waiting cycle
                        RES_fetch <= 2;
                    end else begin
                        RES_fetch <= 0;  // Signal that we need to fetch a new element
    
                        M_AXIS_TVALID <= 1;
                        M_AXIS_TDATA[7:0] <= RES_read_data;
    
                        if (M_AXIS_TREADY == 1) begin
                            if (RES_read_addr == TOTAL_OUTPUT_COUNT - 1) begin
                                state   <= ST_IDLE;
                                M_AXIS_TLAST    <= 1;
                                // M_AXIS_TLAST, though optional in AXIS, is necessary in practice as AXI Stream FIFO and AXI DMA expects it.
                            end else begin
                                RES_read_addr <= RES_read_addr + 1;
                            end
                        end
                    end
    
                    // // Stream out the results stored in RES_RAM
                    // if (!M_AXIS_TVALID || (M_AXIS_TVALID && M_AXIS_TREADY)) begin
                    //     // If ready or just starting, put next result on the bus
                    //     M_AXIS_TDATA[7:0] <= RES_read_data;
                    //     M_AXIS_TVALID <= 1'b1;
                    //     // Mark last transfer when reaching final sample
                    //     if (RES_read_addr == N-1) begin
                    //         M_AXIS_TLAST <= 1'b1;
                    //     end else begin
                    //         M_AXIS_TLAST <= 1'b0;
                    //     end
                    //     // Move to next result
                    //     RES_read_en   <= 1'b1;
                    //     RES_read_addr <= RES_read_addr + 1;
                    // end
                    // // After transmitting the last result, go back to IDLE
                    // if (M_AXIS_TVALID && M_AXIS_TLAST && M_AXIS_TREADY) begin
                    //     M_AXIS_TVALID <= 1'b0;
                    //     M_AXIS_TLAST  <= 1'b0;
                    //     state         <= ST_IDLE;
                    // end
                end
                //-------------------------------------
            endcase
        end // reset else
    end // always
endmodule
