#include "hls_stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"

#define N 64
#define FEATURES 7
#define HIDDEN 2
#define WEIGHT_INPUT_SIZE (FEATURES + 1)  // +1 bias
#define WEIGHT_OUTPUT_SIZE (HIDDEN + 1)   // +1 bias
#define TOTAL_INPUT (N * FEATURES + WEIGHT_INPUT_SIZE * HIDDEN + WEIGHT_OUTPUT_SIZE)
#define TOTAL_OUTPUT (N)

typedef ap_axis<32,0,0,0> AXIS;
typedef ap_uint<8> u8;
typedef ap_uint<16> u16;

u16 fixed_mul(u16 a, u16 b) {
    return a * b;
}

u8 sigmoid(u8 x) {
#pragma HLS inline
    static const u8 sigmoid_table[256] = {
        12,12,12,12,13,13,13,14,14,14,15,15,15,16,16,16,17,17,18,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,26,26,27,27,28,28,29,30,30,31,32,32,33,34,34,35,36,36,37,38,39,39,40,41,42,43,44,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,66,67,68,69,70,72,73,74,75,76,78,79,80,82,83,84,86,87,88,90,91,92,94,95,97,98,99,101,102,104,105,107,108,110,111,113,114,116,117,119,120,122,123,125,126,128,129,130,132,133,135,136,138,139,141,142,144,145,147,148,150,151,153,154,156,157,158,160,161,163,164,165,167,168,169,171,172,173,175,176,177,179,180,181,182,183,185,186,187,188,189,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,211,212,213,214,215,216,216,217,218,219,219,220,221,221,222,223,223,224,225,225,226,227,227,228,228,229,229,230,231,231,232,232,233,233,234,234,234,235,235,236,236,237,237,237,238,238,239,239,239,240,240,240,241,241,241,242,242,242,243,243,243
    };
    return sigmoid_table[x];
}

void myMLP_HLS(hls::stream<AXIS>& S_AXIS, hls::stream<AXIS>& M_AXIS) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS

    u8 X[N][FEATURES];
    u8 w_hid[WEIGHT_INPUT_SIZE][HIDDEN];
    u8 w_out[WEIGHT_OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=w_hid dim=2 complete
#pragma HLS ARRAY_PARTITION variable=w_out complete

    // Read X
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < FEATURES; j++) {
#pragma HLS PIPELINE
            AXIS t = S_AXIS.read();
            X[i][j] = t.data;
        }
    }

    // Reading w_hid
    for (int i = 0; i < WEIGHT_INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN; j++) {
#pragma HLS PIPELINE
            AXIS t = S_AXIS.read();
            w_hid[i][j] = t.data;
        }
    }

    // Reading w_out
    for (int i = 0; i < WEIGHT_OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE
        AXIS t = S_AXIS.read();
        w_out[i] = t.data;
    }

    // Compute and write
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE
        u8 hidden[HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
#pragma HLS UNROLL
            u16 acc = fixed_mul(256, w_hid[0][j]); // bias
            for (int k = 0; k < FEATURES; k++) {
                acc += fixed_mul(X[i][k], w_hid[k + 1][j]);
            }
            hidden[j] = sigmoid(acc >> 8);
        }

        u16 out = fixed_mul(256, w_out[0]);
        for (int j = 0; j < HIDDEN; j++) {
#pragma HLS UNROLL
            out += fixed_mul(hidden[j], w_out[j + 1]);
        }
        AXIS o;
        o.data = (out >> 8) > 128 ? 1 : 0;
        o.last = (i == N - 1);
        M_AXIS.write(o);
    }
}
