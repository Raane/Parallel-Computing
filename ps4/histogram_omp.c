#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;


int main(int argc, char** argv){

  if(argc != 3){
    printf("Useage: %s image n_threads\n", argv[0]);
    exit(-1);
  }
  int n_threads = atoi(argv[2]);

  unsigned char* image = read_bmp(argv[1]);
  unsigned char* output_image = malloc(sizeof(unsigned char) * image_size);

  int* histogram = (int*)calloc(sizeof(int), color_depth);
  int** histograms = (int**)malloc(sizeof(int*)*n_threads);
  // This for loop have very few iterations so I will not parallelize it (the overhead is greater than the gain)
  for(int i=0;i<n_threads;i++) {
    histograms[i] = (int*)calloc(sizeof(int), color_depth);
  }
  //For each pixel, increment the correct cell in the local histogram
  #pragma omp parallel for num_threads(n_threads)
  for(int i = 0; i < image_size; i++){
    histograms[omp_get_thread_num()][image[i]]++;
  }
  //For each cell in the local histogram, add it to the global histogram
  //For the color depths used in these example images, paralellizing this is not that usefull,
  //For images with larger color depth however, it's quite usefull.
  #pragma omp parallel for num_threads(n_threads)
  for(int i=0;i<color_depth;i++) {
    for(int j=0;j<n_threads;j++) {
      histogram[i] += histograms[j][i];
    }
  }


  float* transfer_function = (float*)calloc(sizeof(float), color_depth);
  // The inner loop can not be parallelized, as it would cause race conditions. The outer loop 
  // iterations can be parallelized without and hazards.
  #pragma omp parallel for
  for(int i = 0; i < color_depth; i++){
    for(int j = 0; j < i+1; j++){
      transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
    }
  }

  // Each loop iteration work on a different array position, so this can be parallelized without any race conditions.
  #pragma omp parallel for
  for(int i = 0; i < image_size; i++){
    output_image[i] = transfer_function[image[i]];
  }

  write_bmp(output_image, image_width, image_height);

  // A little code snippet to compare the result to a correct.bmp image.
  // This is here just for testing.
  unsigned char* fasit = read_bmp("correct.bmp");
  int no_errors = 1;
  for(int i=0;i<image_size;i++) {
    if(!(output_image[i] == fasit[i]
        || output_image[i]+1 == fasit[i]
        || output_image[i]-1 == fasit[i])) {
      no_errors = 0;
    }
  }
  printf("Correct: %d\n", no_errors);
}
