`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2025 14:48:38
// Design Name: 
// Module Name: tb_MLP_IP
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

module tb_MLP_IP;
    // Parameters (match the DUT)
    localparam N = 64;
    localparam FEATURES = 7;
    localparam HIDDEN = 2;
    // AXI Stream data width
    localparam TDATA_WIDTH = 32;
    // Clock period
    localparam CLK_PERIOD = 50;

    // DUT signals
    reg                  ACLK;
    reg                  ARESETN;
    wire                 S_AXIS_TREADY;
    reg  [TDATA_WIDTH-1:0] S_AXIS_TDATA;
    reg                  S_AXIS_TLAST;
    reg                  S_AXIS_TVALID;
    wire                 M_AXIS_TVALID;
    wire [TDATA_WIDTH-1:0] M_AXIS_TDATA;
    wire                 M_AXIS_TLAST;
    reg                  M_AXIS_TREADY;

    // Instantiate the Device Under Test (DUT)
    MLP_IP #(
        .N_BITS(6), .FEATURES_BITS(3), .HIDDEN_BITS(1), .TDATA_WIDTH(TDATA_WIDTH)
    ) dut (
        .ACLK(ACLK), .ARESETN(ARESETN),
        .S_AXIS_TREADY(S_AXIS_TREADY),
        .S_AXIS_TDATA(S_AXIS_TDATA),
        .S_AXIS_TLAST(S_AXIS_TLAST),
        .S_AXIS_TVALID(S_AXIS_TVALID),
        .M_AXIS_TVALID(M_AXIS_TVALID),
        .M_AXIS_TDATA(M_AXIS_TDATA),
        .M_AXIS_TLAST(M_AXIS_TLAST),
        .M_AXIS_TREADY(M_AXIS_TREADY)
    );

    // Clock generation
    initial begin
        ACLK = 1'b0;
        forever #(CLK_PERIOD/2) ACLK = ~ACLK;
    end

    // Task to drive a single AXI-Stream word
    task drive_axis_input;
        input [7:0] data;
        input bit last;
        begin
            @(posedge ACLK);
            S_AXIS_TDATA <= {24'd0, data};  // data on lower byte
            S_AXIS_TLAST <= last;
            S_AXIS_TVALID <= 1'b1;
            // Wait until DUT asserts TREADY and a posedge occurs
            wait(S_AXIS_TREADY);
            @(posedge ACLK);
            S_AXIS_TVALID <= 1'b0; // remove valid after one beat
        end
    endtask

    // Stimulus data (from HLS testbench)
    reg [7:0] X     [0:N*FEATURES-1];
    reg [7:0] W_hid [0:(FEATURES+1)*HIDDEN-1];
    reg [7:0] W_out [0:HIDDEN];
    reg [7:0] expected_labels [0:N-1];
    integer i, j, idx;
    bit success = 1'b1;

    initial begin
        // Initialize input feature matrix X (64x7) and weights with known test data
        // (Here we load partial data for brevity; in a real test, load all values or read from file)
        $readmemh("X_data.mem", X);         // Contains 448 values for X
        $readmemh("W_hid_data.mem", W_hid); // Contains 16 values for hidden weights (8x2)
        $readmemh("W_out_data.mem", W_out); // Contains 3 values for output weights (3x1)
        $readmemh("labels.mem", expected_labels); // Contains 64 expected output labels
    end

    // Reset sequence
    initial begin
        // Start in reset
        ARESETN = 1'b0;
        S_AXIS_TDATA = 32'd0;
        S_AXIS_TVALID = 1'b0;
        S_AXIS_TLAST  = 1'b0;
        M_AXIS_TREADY = 1'b1;  // ready to accept outputs
        #(CLK_PERIOD * 5);      // hold reset for a few cycles
        
        ARESETN = 1'b1;
        @(posedge ACLK);       // wait for a clock edge after reset deassertion
        #1;
        // Drive input stream
        
        // Send X matrix (N*FEATURES bytes)
        idx = 0;
        for (i = 0; i < N; i = i + 1) begin
            for (j = 0; j < FEATURES; j = j + 1) begin
                // (We don't assert TLAST here because more data follows after X)
                automatic bit last_flag = 0;
                drive_axis_input(X[idx], last_flag);
                idx = idx + 1;
            end
        end
        
        // Send hidden layer weights (including biases)
        for (i = 0; i < FEATURES + 1; i = i + 1) begin
            for (j = 0; j < HIDDEN; j = j + 1) begin
                // (We don't assert TLAST here because more data follows after X)
                automatic bit last_flag = 1'b0;
                drive_axis_input(W_hid[i * HIDDEN + j], last_flag);
            end
        end
        
        // Send output layer weights (including bias)
        for (i = 0; i < HIDDEN + 1; i = i + 1) begin
            automatic bit last_flag = (i == HIDDEN) ? 1'b1 : 1'b0;  // last word of input stream
            drive_axis_input(W_out[i], last_flag);
        end

        // Now wait for outputs and collect them
        idx = 0;
        while (idx < N) begin
            @(posedge ACLK);
            if (M_AXIS_TVALID) begin
                // Capture output
                reg [7:0] out_byte;
                out_byte = M_AXIS_TDATA[7:0];
                $display("Output[%0d] = %0d", idx, out_byte);
                // Check against expected label
                if (out_byte !== expected_labels[idx]) begin
                    $display("Mismatch at sample %0d: expected %0d, got %0d", idx, expected_labels[idx], out_byte);
                    success = 0;
                end
                idx = idx + 1;
            end
        end
        // All outputs received, end simulation
        $display("Test completed");
        $finish;
    end
endmodule
