#include "headers/pngUtils.h"

// allocate memory for a 'PngImage' struct object 
PngImage* make_img(png_structp png_ptr, png_infop info_ptr, int padding) {
    PngImage* img = (PngImage*)malloc(sizeof(PngImage));
    img->PAD = padding;
    img->W = png_get_image_width(png_ptr, info_ptr);
    img->H = png_get_image_height(png_ptr, info_ptr);
    img->color_type = png_get_color_type(png_ptr, info_ptr);
    img->C = get_num_channels(img->color_type);

    img->val = (matrix)malloc(sizeof(matrix_element) * (img->W + 2*img->PAD) * (img->H + 2*img->PAD) * img->C);
    return img;
}

// free memory of a 'PngImage' struct object 
void del_img(PngImage* image) {
    if (image) {
        free(image->val);
        free(image);
    }
}

// given a 'color_type' from png.h library outputs the corresponding number of channels
unsigned int get_num_channels(png_byte color_type) {
    switch (color_type) {
        case PNG_COLOR_TYPE_GRAY:
            return 1;
        case PNG_COLOR_TYPE_RGB:
            return 3;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            return 4;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            return 2;
        default:
            return 0; // Unknown or Unsupported color type
    }
}

// outputs a 'PngImage' object according to the specified '.png' file content
PngImage* read_png(char *file_name, int padding) {
    FILE *fp = fopen(file_name, "rb");

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);

    // initialize custom image type
    PngImage* img = make_img(png_ptr, info_ptr, padding);
    for (unsigned int i = 0; i < img->H; i++) {
        for (unsigned int j = 0; j < img->W; j++) {
            for (unsigned int c = 0; c < img->C; c++) {
                img->val[((i+img->PAD) * (img->W+2*img->PAD) * img->C) + ((j+img->PAD) * img->C) + c] = (matrix_element)row_pointers[i][(j * img->C) + c];
            }
        }
    }

    // free memory
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    
    return img;
}


// saves a .png file according to the specified 'PngImage' object
void write_png(char *file_name, PngImage *img) {

    if (img->C == 0){
        perror("Unknown or unsupported color type for write_png()\n");
        return;
    }

    // Open file for writing (binary mode)
    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        perror("Failed to open file for writing");
        return;
    }

    // Initialize components
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, img->W, img->H, 8, img->color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    // Convert float array to png_byte array
    png_bytepp row_pointers = (png_bytepp) malloc(sizeof(png_bytep) * img->H);

    for (unsigned int i = 0; i < img->H; i++) {
        row_pointers[i] = (png_bytep) malloc(sizeof(png_byte) * img->W * 3);
        for (unsigned int j = 0; j < img->W; j++) {
            for (unsigned int c = 0; c < img->C; c++){
                row_pointers[i][j*img->C + c] = img->val[((i+img->PAD) * (img->W+2*img->PAD) * img->C) + ((j+img->PAD) * img->C) + c];
            }
        }
    }

    // Write image data
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    free(row_pointers);

    // free memory
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}