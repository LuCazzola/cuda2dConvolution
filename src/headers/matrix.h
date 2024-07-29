#ifndef MATRIX_H
#define MATRIX_H

#include "common.h"

// to get float and int limit values
#include <float.h>
#include <limits.h>

// 'MATRIX_ELEM_DTYPE' defined in the makefile and imported via -D flag
typedef MATRIX_ELEM_DTYPE matrix_element;
typedef matrix_element* matrix;

// fill the input matrix with random "matrix_element" datatype
void fill_matrix (matrix mat, const int ROWS, const int COLS);
// print content of the input matrix
void print_matrix(matrix mat, const int ROWS, const int COLS);
// returns a copy of the input matrix
matrix matrix_copy(matrix src, const int ROWS, const int COLS);

// compute the total point-wise error in transposing A into B
//      NOTE : when comparing very big matrices this implementation is not
//      numerically stable resulting in wrong outputs. It's tested to work
//      well when size is <= 2^28, otherwise it's unstable
void print_transpose_error(matrix A, matrix B, const int SIZE);

#endif