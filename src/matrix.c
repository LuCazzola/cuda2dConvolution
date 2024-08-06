#include "headers/matrix.h"
#include "headers/opt_parser.h"

// initialize the matrix with random values
void fill_matrix_random (matrix mat, const int BUFFER_SIZE){

    char datatype [] = {VALUE(MATRIX_ELEM_DTYPE)};
    bool is_int = strcmp(datatype, "int") == 0 ? true : false;

    srand(time(NULL));
    // initialize main matrix
    for(unsigned int i = 0; i < BUFFER_SIZE; i++){
        mat[i] = is_int ? (int)(rand() - rand()) : (float)((float)(rand()) / (float)(rand())-(float)(rand()) / (float)(rand()));
    }
}

// print the specified matrix
void print_matrix_2d(matrix mat, const int A, const int B){
    char datatype [] = {VALUE(MATRIX_ELEM_DTYPE)};
    bool is_int = strcmp(datatype, "int") == 0 ? true : false;

    for(unsigned int i = 0; i < A; i++){
        for (unsigned int j = 0; j < B; j++){
           is_int ? printf("%d, ", (int)mat[i*B + j]) : printf("%f, ", (float)mat[i*B + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// print the specified matrix
void print_matrix_3d(matrix mat, const int A, const int B, const int C){
    char datatype [] = {VALUE(MATRIX_ELEM_DTYPE)};
    bool is_int = strcmp(datatype, "int") == 0 ? true : false;

    for(unsigned int i = 0; i < A; i++){
        for (unsigned int j = 0; j < B; j++){
            for (unsigned int k = 0; k < C; k++){
                is_int ? printf("%d, ", (int)mat[i*B*C + j*C + k]) : printf("%f, ", (float)mat[i*B*C + j*C + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}