#include "headers/matrix.h"
#include "headers/opt_parser.h"

// initialize the matrix with random values
void fill_matrix (matrix mat, const int ROWS, const int COLS){

    char datatype [] = {VALUE(MATRIX_ELEM_DTYPE)};
    bool is_int = strcmp(datatype, "int") == 0 ? true : false;

    srand(time(NULL));
    // initialize main matrix
    for(unsigned int i = 0; i < ROWS; i++){
        for (unsigned int j = 0; j < COLS; j++){
            mat[i*ROWS + j] = is_int ? (int)(rand() - rand()) : (float)((float)(rand()) / (float)(rand())-(float)(rand()) / (float)(rand()));
        }
    }
}

// print the specified matrix
void print_matrix(matrix mat, const int ROWS, const int COLS){
    char datatype [] = {VALUE(MATRIX_ELEM_DTYPE)};
    bool is_int = strcmp(datatype, "int") == 0 ? true : false;

    for(unsigned int i = 0; i < ROWS; i++){
        for (unsigned int j = 0; j < COLS; j++){
           is_int ? printf("%d, ", (int)mat[i*ROWS + j]) : printf("%f, ", (float)mat[i*ROWS + j]);
        }
        printf("\n");
    }
    printf("\n");
}

matrix matrix_copy(matrix src, const int ROWS, const int COLS){
    matrix copy = (matrix) malloc(ROWS * COLS * sizeof(matrix_element));

    for(unsigned int i = 0; i < ROWS; i++){
        for (unsigned int j = 0; j < COLS; j++){
            copy[i*ROWS + j] = src[i*ROWS + j];
        }
    }

    return copy;
}

// NOTE : when comparing very big matrices this implementation is not numerically stable resulting in wrong outputs.
//        it's tested to work well when size is <= 2^28, otherwise it's unstable
void print_transpose_error(matrix A, matrix B, const int SIZE){
    float error = 0;
    
    for(unsigned int i = 0; i < SIZE; i++){
        for (unsigned int j = 0; j < SIZE; j++){
            error += A[i*SIZE + j] - B[j*SIZE + i];
        }
    }
    printf("\n=== Transpose Error === :\n= %f\n\n", error);
}