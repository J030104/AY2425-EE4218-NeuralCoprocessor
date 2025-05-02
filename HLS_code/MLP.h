#ifndef MLP_H
#define MLP_H

#include "ap_int.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

#define TIMEOUT_VALUE 1 << 20 // timeout for FIFO reception

#define N 64
#define FEATURES 7
#define HIDDEN 2
#define OUTPUT 1
#define WEIGHT_INPUT_SIZE (FEATURES + 1)  // +1 for bias
#define WEIGHT_OUTPUT_SIZE (HIDDEN + 1)   // +1 for bias

extern u8 X[N][FEATURES];

extern const u8 w_hid[WEIGHT_INPUT_SIZE][HIDDEN];
extern const u8 w_out[WEIGHT_OUTPUT_SIZE];

extern const u8 labels[N];
extern u8 result_soft[N];
extern u8 result_HLS[N];

extern const u8 sigmoid_table[256];

u8 sigmoid(u16 x, const u8 *sigmoid_table);

u16 fixed_mul(u16 a, u16 b);

extern int time0, time1;

u8 mlp_predict(u8 input[FEATURES], const u8 w_hid[WEIGHT_INPUT_SIZE][HIDDEN], const u8 w_out[WEIGHT_OUTPUT_SIZE], const u8 *sigmoid_table);
void myMLP_software(u8 X[N][FEATURES], u8 result[N], const u8 w_hid[WEIGHT_INPUT_SIZE][HIDDEN], const u8 w_out[WEIGHT_OUTPUT_SIZE], const u8 *sigmoid_table);

#endif // MLP_H
