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
void fill_matrix_random (matrix mat, const int BUFFER_SIZE);

// print content of the input matrix
void print_matrix_2d(matrix mat, const int A, const int B);
void print_matrix_3d(matrix mat, const int A, const int B, const int C);

#endif