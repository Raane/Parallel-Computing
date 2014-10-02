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
  unsigned char* fasit = read_bmp("fasit.bmp");
  unsigned char* output_image = malloc(sizeof(unsigned char) * image_size);

  int* histogram = (int*)calloc(sizeof(int), color_depth);
  int** histograms = (int**)malloc(sizeof(int*)*n_threads);
  for(int i=0;i<n_threads;i++) {
    histograms[i] = (int*)calloc(sizeof(int), color_depth);
  }
  #pragma omp parallel for
  for(int i = 0; i < image_size; i++){
    //histograms[omp_get_thread_num()][image[i]]++;
    histograms[1][image[i]]++;
    //printf("Thread: %d, num%d\n", omp_get_thread_num(), omp_get_num_threads());
  }
  /*for(int i=0;i<color_depth;i++) {
    for(int j=0;j<omp_get_num_threads();j++) {
      histogram[i] += histograms[j][i];
    }
  }*/


  float* transfer_function = (float*)calloc(sizeof(float), color_depth);
  #pragma omp parallel for
  for(int i = 0; i < color_depth; i++){
    for(int j = 0; j < i+1; j++){
      transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
    }
  }


  for(int i = 0; i < image_size; i++){
    output_image[i] = transfer_function[image[i]];
  }

  write_bmp(output_image, image_width, image_height);

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
