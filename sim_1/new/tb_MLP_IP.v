`timescale 1ns / 1ps

/*
----------------------------------------------------------------------------------
--  (c) Rajesh C Panicker, NUS
--  Description : Self-checking testbench for Matrix Multiplication AXI Stream Coprocessor.
--  License terms :
--  You are free to use this code as long as you
--      (i) DO NOT post a modified version of this on any public repository;
--      (ii) use it only for educational purposes;
--      (iii) accept the responsibility to ensure that your implementation does not violate any intellectual property of any entity.
--      (iv) accept that the program is provided "as is" without warranty of any kind or assurance regarding its suitability for any particular purpose;
--      (v) send an email to rajesh.panicker@ieee.org briefly mentioning its use (except when used for the course EE4218 at the National University of Singapore);
--      (vi) retain this notice in this file or any files derived from this.
----------------------------------------------------------------------------------
*/

module tb_MLP_IP ();

    // verilog_format: off
    reg                          ACLK = 0;    // Synchronous clock
    reg                          ARESETN;     // System reset, active low
    // slave in interface
    wire                         S_AXIS_TREADY;  // Ready to accept data in
    reg      [31 : 0]            S_AXIS_TDATA;   // Data in
    reg                          S_AXIS_TLAST;   // Optional data in qualifier
    reg                          S_AXIS_TVALID;  // Data in is valid
    // master out interface
    wire                         M_AXIS_TVALID;  // Data out is valid
    wire     [31 : 0]            M_AXIS_TDATA;   // Data out
    wire                         M_AXIS_TLAST;   // Optional data out qualifier
    reg                          M_AXIS_TREADY;  // Connected slave device is ready to accept data out
    // verilog_format: on

    // Parameters (match the DUT)
    localparam N = 64;
    localparam FEATURES = 7;
    localparam HIDDEN = 2;
    // AXI Stream data width
    localparam TDATA_WIDTH = 32;
    // AXI Stream data width actually used
    localparam width = 8;  // width of an input vector

    localparam X_ENTRIES = N * FEATURES;
    localparam W_HID_ENTRIES = (FEATURES + 1) * HIDDEN;
    localparam W_OUT_ENTRIES = HIDDEN + 1;
    localparam TOTAL_OUTPUT_COUNT = N;

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

    reg [7:0] X     [0:N*FEATURES-1];
    reg [7:0] W_hid [0:(FEATURES+1)*HIDDEN-1];
    reg [7:0] W_out [0:HIDDEN];
    reg [7:0] results [0:N-1];
    reg [7:0] expected_labels [0:N-1];
    // reg [8:0] i;
    integer i;

    reg success = 1'b1;
    reg M_AXIS_TLAST_prev = 1'b0;

    always @(posedge ACLK) M_AXIS_TLAST_prev <= M_AXIS_TLAST;

    always #50 ACLK = ~ACLK;

    initial begin
        $dumpfile("tb_mlp_ip_v1_0.vcd");
        // $dumpvars(0, tb_mlp_ip_v1_0);

        $display("Loading Memory.");
        $readmemh("X_data.mem", X);         // Contains 448 values for X
        $readmemh("W_hid_data.mem", W_hid); // Contains 16 values for hidden weights (8x2)
        $readmemh("W_out_data.mem", W_out); // Contains 3 values for output weights (3x1)
        $readmemh("labels.mem", expected_labels); // Contains 64 expected output labels
        #25  // to make inputs and capture from testbench not aligned with clock edges
        
        ARESETN = 1'b0;  // apply reset (active low)
        S_AXIS_TVALID = 1'b0;  // no valid data placed on the S_AXIS_TDATA yet
        S_AXIS_TLAST = 1'b0;    // not required unless we are dealing with an unknown number of inputs. Ignored by the coprocessor. We will be asserting it correctly anyway
        M_AXIS_TREADY = 1'b0;  // not ready to receive data from the co-processor yet.

        #100             // hold reset for 100 ns.
        ARESETN = 1'b1;  // release reset

        // Input X
        i = 0;
        S_AXIS_TVALID = 1'b1; // data is ready at the input of the coprocessor.
        while (i < X_ENTRIES) begin
            if(S_AXIS_TREADY) begin  // S_AXIS_TREADY is asserted by the coprocessor in response to S_AXIS_TVALID
                S_AXIS_TDATA = X[i]; // set the next data ready
                i = i + 1;
            end
            #100;           // wait for one clock cycle before for co-processor to capture data (if S_AXIS_TREADY was set) 
                            // or before checking S_AXIS_TREADY again (if S_AXIS_TREADY was not set)
        end

        // Input w_hid
        i = 0;
        S_AXIS_TVALID = 1'b1; // data is ready at the input of the coprocessor.
        while (i < W_HID_ENTRIES) begin
            if(S_AXIS_TREADY) begin  // S_AXIS_TREADY is asserted by the coprocessor in response to S_AXIS_TVALID
                S_AXIS_TDATA = W_hid[i]; // set the next data ready
                i = i + 1;
            end
            #100;           // wait for one clock cycle before for co-processor to capture data (if S_AXIS_TREADY was set) 
                            // or before checking S_AXIS_TREADY again (if S_AXIS_TREADY was not set)
        end

        // Input w_out
        i = 0;
        S_AXIS_TVALID = 1'b1; // data is ready at the input of the coprocessor.
        while (i < W_OUT_ENTRIES) begin
            if(S_AXIS_TREADY) begin  // S_AXIS_TREADY is asserted by the coprocessor in response to S_AXIS_TVALID
                S_AXIS_TDATA = W_out[i]; // set the next data ready
                if (i == W_OUT_ENTRIES - 1) S_AXIS_TLAST = 1'b1;
                else S_AXIS_TLAST = 1'b0;
                i = i + 1;
            end
            #100;           // wait for one clock cycle before for co-processor to capture data (if S_AXIS_TREADY was set) 
                            // or before checking S_AXIS_TREADY again (if S_AXIS_TREADY was not set)
        end
        S_AXIS_TVALID = 1'b0;  // we no longer give any data to the co-processor
        S_AXIS_TLAST = 1'b0;

        // Output RES
        i = 0;
        M_AXIS_TREADY = 1'b1;
        while (M_AXIS_TLAST | ~M_AXIS_TLAST_prev) begin
            if (M_AXIS_TVALID) begin
                results[i] = M_AXIS_TDATA;
                i = i + 1;
            end
            #100;
        end
        M_AXIS_TREADY = 1'b0;

        // checking correctness of results
        for (
            i = 0;
            i < TOTAL_OUTPUT_COUNT;
            i = i + 1
        ) begin
            success = success & (results[i] == expected_labels[i]);
            if (success) $display("Test Passed.");
            else $display("Test Failed. i = %d, %h != %h",
                          i,
                          results[i],
                          expected_labels[i]);
        end

        $finish;
    end

endmodule
