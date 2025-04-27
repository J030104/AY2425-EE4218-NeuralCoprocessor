`timescale 1ns / 1ps

module tb_nn_v1_0;
  // Parameter definitions
  parameter DATA_WIDTH   = 8;
  parameter N            = 64;
  parameter FEATURES     = 7;
  parameter HIDDEN       = 2;
  parameter A_DEPTH_BITS = 9;  // Addresses for 64*7 inputs
  parameter B_DEPTH_BITS = 4;  // Addresses for 16 hidden weights
  parameter C_DEPTH_BITS = 2;  // Addresses for 3 output weights
  parameter RES_DEPTH_BITS = 6; // Addresses for 64 outputs

  // Clock and control signals
  reg clk;
  reg Start;
  wire Done;

  // DUT <-> Memory interface signals
  wire                 A_read_en;
  wire [A_DEPTH_BITS-1:0] A_read_address;
  wire [DATA_WIDTH-1:0]   A_read_data_out;

  wire                 B_read_en_1, B_read_en_2;
  wire [B_DEPTH_BITS-1:0] B_read_address_1, B_read_address_2;
  wire [DATA_WIDTH-1:0]   B_read_data_out_1, B_read_data_out_2;

  wire                 C_read_en_1, C_read_en_2;
  wire [C_DEPTH_BITS-1:0] C_read_address_1, C_read_address_2;
  wire [DATA_WIDTH-1:0]   C_read_data_out_1, C_read_data_out_2;

  wire                 RES_write_en;
  wire [RES_DEPTH_BITS-1:0] RES_write_address;
  wire [DATA_WIDTH-1:0]     RES_write_data_in;

  // Sigmoid LUT interface (dual-port ROM)
  wire                 Sigmoid_en_1, Sigmoid_en_2;
  wire [DATA_WIDTH-1:0] Sigmoid_lookup_1, Sigmoid_lookup_2;
  wire [DATA_WIDTH-1:0] Sigmoid_result_1, Sigmoid_result_2;

  // Instantiate the Neural Network DUT
  neural_network_v1_0 #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N),
    .FEATURES(FEATURES),
    .HIDDEN(HIDDEN),
    .A_DEPTH_BITS(A_DEPTH_BITS),
    .B_DEPTH_BITS(B_DEPTH_BITS),
    .C_DEPTH_BITS(C_DEPTH_BITS),
    .RES_DEPTH_BITS(RES_DEPTH_BITS)
  ) dut (
    .clk(clk),
    .Start(Start),
    .Done(Done),
    // Sigmoid interface
    .Sigmoid_en_1(Sigmoid_en_1), .Sigmoid_lookup_1(Sigmoid_lookup_1), .Sigmoid_result_1(Sigmoid_result_1),
    .Sigmoid_en_2(Sigmoid_en_2), .Sigmoid_lookup_2(Sigmoid_lookup_2), .Sigmoid_result_2(Sigmoid_result_2),
    // Input matrix A memory interface
    .A_read_en(A_read_en), .A_read_address(A_read_address), .A_read_data_out(A_read_data_out),
    // Hidden weights B memory interface (two ports for two neurons)
    .B_read_en_1(B_read_en_1), .B_read_address_1(B_read_address_1), .B_read_data_out_1(B_read_data_out_1),
    .B_read_en_2(B_read_en_2), .B_read_address_2(B_read_address_2), .B_read_data_out_2(B_read_data_out_2),
    // Output weights C memory interface (two ports for two weights at once)
    .C_read_en_1(C_read_en_1), .C_read_address_1(C_read_address_1), .C_read_data_out_1(C_read_data_out_1),
    .C_read_en_2(C_read_en_2), .C_read_address_2(C_read_address_2), .C_read_data_out_2(C_read_data_out_2),
    // Result memory interface
    .RES_write_en(RES_write_en), .RES_write_address(RES_write_address), .RES_write_data_in(RES_write_data_in)
  );

  // Instantiate memory for input matrix X (single-port RAM)
  memory_RAM #(.width(DATA_WIDTH), .depth_bits(A_DEPTH_BITS)) X_mem (
    .clk(clk),
    .write_en(1'b0), .write_address({A_DEPTH_BITS{1'b0}}), .write_data_in({DATA_WIDTH{1'b0}}), 
    .read_en(A_read_en), .read_address(A_read_address), .read_data_out(A_read_data_out)
  );

  // Instantiate memory for hidden layer weights (dual-read RAM)
  memory_RAM_d_r #(.width(DATA_WIDTH), .depth_bits(B_DEPTH_BITS)) Wh_mem (
    .clk(clk),
    .write_en(1'b0), .write_address({B_DEPTH_BITS{1'b0}}), .write_data_in({DATA_WIDTH{1'b0}}),
    .read_en_1(B_read_en_1), .read_address_1(B_read_address_1), .read_data_out_1(B_read_data_out_1),
    .read_en_2(B_read_en_2), .read_address_2(B_read_address_2), .read_data_out_2(B_read_data_out_2)
  );

  // Instantiate memory for output layer weights (dual-read RAM)
  memory_RAM_d_r #(.width(DATA_WIDTH), .depth_bits(C_DEPTH_BITS)) Wo_mem (
    .clk(clk),
    .write_en(1'b0), .write_address({C_DEPTH_BITS{1'b0}}), .write_data_in({DATA_WIDTH{1'b0}}),
    .read_en_1(C_read_en_1), .read_address_1(C_read_address_1), .read_data_out_1(C_read_data_out_1),
    .read_en_2(C_read_en_2), .read_address_2(C_read_address_2), .read_data_out_2(C_read_data_out_2)
  );

  // Instantiate memory for result output (single-port RAM)
  memory_RAM #(.width(DATA_WIDTH), .depth_bits(RES_DEPTH_BITS)) RES_mem (
    .clk(clk),
    .write_en(RES_write_en), .write_address(RES_write_address), .write_data_in(RES_write_data_in),
    .read_en(1'b0), .read_address({RES_DEPTH_BITS{1'b0}}), .read_data_out()  // not reading asynchronously in this TB
  );

  // Instantiate dual-port ROM for Sigmoid LUT (using memory_RAM_d_r as ROM)
  // memory_RAM_d_r #(.width(DATA_WIDTH), .depth_bits(8)) SigmoidROM (
  //   .clk(clk),
  //   .write_en(1'b0), .write_address(8'd0), .write_data_in({DATA_WIDTH{1'b0}}),
  //   .read_en_1(Sigmoid_en_1), .read_address_1(Sigmoid_lookup_1), .read_data_out_1(Sigmoid_result_1),
  //   .read_en_2(Sigmoid_en_2), .read_address_2(Sigmoid_lookup_2), .read_data_out_2(Sigmoid_result_2)
  // );

  ROM256 SigmoidROM (
    .clk(clk),
    .enable_1(Sigmoid_en_1),
    .enable_2(Sigmoid_en_2),
    .address_1(Sigmoid_lookup_1),
    .address_2(Sigmoid_lookup_2),
    .data_1(Sigmoid_result_1),
    .data_2(Sigmoid_result_2)
  );

  // Testbench variables for expected labels
  reg [DATA_WIDTH-1:0] labels_mem [0:N-1];
  // Clock generation: 100MHz clock (10ns period as example)
  initial clk = 0;
  always #5 clk = ~clk;

  // Load memory contents at start of simulation
  initial begin
    $readmemh("X_data.mem",     X_mem.RAM);        // Load input features (64x7 values)
    $readmemh("W_hid_data.mem", Wh_mem.RAM);       // Load hidden layer weights (16 values)
    $readmemh("W_out_data.mem", Wo_mem.RAM);       // Load output layer weights (3 values)
    $readmemh("ROM256.dat",     SigmoidROM.ROM);   // Load sigmoid lookup table (256 values)
    $readmemh("labels.mem",     labels_mem);       // Load expected labels for comparison
  end

  // Drive the Start signal to trigger the computation after initialization
  initial begin
     repeat (5) @(posedge clk);  // Wait 5 clock cycles
    Start = 1'b0;
    @(posedge clk);
    Start = 1'b1;               // Raise Start BEFORE edge
    @(posedge clk);             // Keep it high for one cycle
    Start = 1'b0;               // Done
  end

  // Monitor results: print output when a result is written, comparing with expected label
  always @(posedge clk) begin
    if (RES_write_en) begin
      $display("Output for sample %0d = %0d (expected %0d)",
               RES_write_address, RES_write_data_in, labels_mem[RES_write_address]);
    end
    if (Done) begin
      $display("All %0d samples processed. Simulation done.", N);
      $finish;
    end
  end

endmodule
