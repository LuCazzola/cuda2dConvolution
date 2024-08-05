#ifndef PNGUTILS_H
#define PNGUTILS_H

#include "matrix.h"
#include "common.h"
#include <png.h> // requires loading "module load libpng/1.6.39-GCCcore-12.3.0" on Marzola cluster

typedef struct {
    int W;        // image width in pixels
    int H;        // image height in pixels
    int C;        // number of channels
    int PAD;      // padding
    png_byte color_type;   // Color type information
    float* val;            // values of the image
} PngImage;

// allocate memory for a 'PngImage' struct object 
PngImage* make_img(png_structp png_ptr, png_infop info_ptr, int padding);
// free memory for a 'PngImage' struct object 
void del_img(PngImage* image);

// given a 'color_type' from png.h library outputs the corresponding number of channels
unsigned int get_num_channels(png_byte color_type);

// outputs a 'PngImage' object according to the specified '.png' file content
PngImage* read_png(char *file_name, int padding);
// saves a .png file according to the specified 'PngImage' object
void write_png(char *file_name, PngImage *img);

#endif