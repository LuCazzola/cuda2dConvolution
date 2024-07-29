#include <stdio.h>
#include <stdlib.h>

#define IMAGE_DIM 5

void generate_image(int dim, int* image){
    // Generate image for testing, use loaded image later
    for(int i = 0; i < dim * dim; i++){
        image[i] = rand() % 100;
    }
}

void apply_convolution(int image_dim, int* image, int filter_dim, float* filter, int* output){
    if(filter_dim % 2 == 0){ // Only allow odd filter size
        perror("Method 'apply_convolution' takes only filters with odd dimensions. Got even dimension.");
        exit(EXIT_FAILURE);
    }
    
    float sum = 0.0;
    int filter_center = filter_dim / 2;

    for(int i = 0; i < image_dim; i++){
        for(int j = 0; j < image_dim; j++){
            sum = 0.0;
            for(int k = 0; k < filter_dim; k++){
                for(int l = 0; l < filter_dim; l++){
                    int patch_i = i-filter_center+k;
                    int patch_j = j-filter_center+l;
                    if(patch_i < 0 || patch_i >= image_dim || patch_j < 0 || patch_j >= image_dim){
                        sum += 0;
                    }
                    else {
                        sum += filter[k*filter_dim+l] * (float) image[patch_i*image_dim+patch_j];
                    }
                }
            }
            output[i*image_dim+j] = (int) sum;
        }
    }
}

void print_matrix(int dim, int* mat){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            printf("%2d ", mat[i*dim+j]);
        }
        printf("\n");
    }
}

void print_filter(int dim, float* mat){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            printf("%2.2f ", mat[i*dim+j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv){
    int* image = (int *) malloc(IMAGE_DIM * IMAGE_DIM * sizeof(int));
    int* output = (int *) malloc(IMAGE_DIM * IMAGE_DIM * sizeof(int));
    generate_image(IMAGE_DIM, image);

    printf("Image:\n");
    print_matrix(IMAGE_DIM, image);
    printf("\n");

    int filter_dim = 3;
    float filter[] = {1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0};

    printf("Filter:\n");
    print_filter(filter_dim, filter);
    
    apply_convolution(IMAGE_DIM, image, filter_dim, filter, output);
    printf("\n");
    printf("Output:\n");
    print_matrix(IMAGE_DIM, output);
    
    free(image);
    free(output);
}