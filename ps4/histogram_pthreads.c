#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;

typedef struct{
  int n_threads;
  int thread_n;
  int** histograms;
  unsigned char* image;
}histogram_count_args;

typedef struct{
  int n_threads;
  int thread_n;
  int* histogram;
  int** histograms;
}histogram_sum_args;

typedef struct{
  int n_threads;
  int thread_n;
  float* transfer_function;
  int* histogram;
}transfer_args;

void *threaded_histogram_count(void* a) {
  histogram_count_args* b = (histogram_count_args*)a;
  for(int i = image_size * (b->thread_n) / (b->n_threads); i <image_size * (b->thread_n+1) / (b->n_threads); i++){
    (b->histograms)[b->thread_n][(b->image)[i]]++;
  }
  pthread_exit(NULL);
}

void *threaded_histogram_sum(void* a) {
  histogram_sum_args* b = (histogram_sum_args*)a;
  for(int i = color_depth * (b->thread_n) / (b->n_threads); i <color_depth * (b->thread_n+1) / (b->n_threads); i++){
    for(int j=0;j<b->n_threads;j++) {
      b->histogram[i] += b->histograms[j][i];
    }
  }
  pthread_exit(NULL);
}

void *threaded_transfer(void* a) {
  transfer_args* b = (transfer_args*)a;
  for(int i = color_depth * (b->thread_n) / (b->n_threads); i <color_depth * (b->thread_n+1) / (b->n_threads); i++){
    for(int j = 0; j < i+1; j++){
      (b->transfer_function)[i] += color_depth*((float)(b->histogram[j]))/(image_size);
    }
  }
  pthread_exit(NULL);
}

int main(int argc, char** argv){

  if(argc != 3){
    printf("Useage: %s image n_threads\n", argv[0]);
    exit(-1);
  }
  int n_threads = atoi(argv[2]);
  pthread_t threads[n_threads];

  unsigned char* image = read_bmp(argv[1]);
  unsigned char* output_image = malloc(sizeof(unsigned char) * image_size);


  int* histogram = (int*)calloc(sizeof(int), color_depth);
  int** histograms = (int**)malloc(sizeof(int*)*n_threads);
  // This for loop have very few iterations so I will not parallelize it (the overhead is greater than the gain)
  for(int i=0;i<n_threads;i++) {
    histograms[i] = (int*)calloc(sizeof(int), color_depth);
  }
  //For each pixel, increment the correct cell in the local histogram
  for(int thread = 0; thread<n_threads; thread++) {
    histogram_count_args* a = malloc(sizeof(histogram_count_args));
    a->n_threads = n_threads;
    a->thread_n = thread;
    a->histograms = histograms;
    a->image = image;
    pthread_create(&threads[thread], NULL, threaded_histogram_count, (void *)a);
  }
  for(int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL); // Wait for all threads to finish
  /*for(int i = 0; i < image_size; i++){
    histogram[image[i]]++;
  }*/
  //For each cell in the local histogram, add it to the global histogram
  for(int thread = 0; thread<n_threads; thread++) {
    histogram_sum_args* a = malloc(sizeof(histogram_sum_args));
    a->n_threads = n_threads;
    a->thread_n = thread;
    a->histogram = histogram;
    a->histograms = histograms;
    pthread_create(&threads[thread], NULL, threaded_histogram_sum, (void *)a);
  }
  for(int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL); // Wait for all threads to finish
  for(int i=0;i<color_depth;i++) {
  }
  
  
  
  float* transfer_function = (float*)calloc(sizeof(float), color_depth);
  for(int thread = 0; thread<n_threads; thread++) {
    transfer_args* a = malloc(sizeof(transfer_args));
    a->n_threads = n_threads;
    a->thread_n = thread;
    a->transfer_function = transfer_function;
    a->histogram = histogram;

    pthread_create(&threads[thread], NULL, threaded_transfer, (void *)a);
  }



  for(int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL); // Wait for all threads to finish

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
