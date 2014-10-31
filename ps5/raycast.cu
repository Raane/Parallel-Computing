#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "bmp.h"

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512

texture<int, cudaTextureType3D, cudaReadModeElementType> data_texture;
texture<int, cudaTextureType3D, cudaReadModeElementType> region_texture;


// Stack for the serial region growing
typedef struct{
  int size;
  int buffer_size;
  int3* pixels;
} stack_t;

stack_t* new_stack(){
  stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
  stack->size = 0;
  stack->buffer_size = 1024;
  stack->pixels = (int3*)malloc(sizeof(int3)*1024);

  return stack;
}

void push(stack_t* stack, int3 p){
  if(stack->size == stack->buffer_size){
    stack->buffer_size *= 2;
    int3* temp = stack->pixels;
    stack->pixels = (int3*)malloc(sizeof(int3)*stack->buffer_size);
    memcpy(stack->pixels, temp, sizeof(int3)*stack->buffer_size/2);
    free(temp);

  }
  stack->pixels[stack->size] = p;
  stack->size += 1;
}

int3 pop(stack_t* stack){
  stack->size -= 1;
  return stack->pixels[stack->size];
}

// float3 utilities
__device__ __host__ float3 cross(float3 a, float3 b){
  float3 c;
  c.x = a.y*b.z - a.z*b.y;
  c.y = a.z*b.x - a.x*b.z;
  c.z = a.x*b.y - a.y*b.x;

  return c;
}

__device__ __host__ float3 normalize(float3 v){
  float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  v.x /= l;
  v.y /= l;
  v.z /= l;

  return v;
}

__device__ __host__ float3 add(float3 a, float3 b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;

  return a;
}

__device__ __host__ float3 scale(float3 a, float b){
  a.x *= b;
  a.y *= b;
  a.z *= b;

  return a;
}


// Prints CUDA device properties
void print_properties(){
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Device count: %d\n", deviceCount);

  cudaDeviceProp p;
  cudaSetDevice(0);
  cudaGetDeviceProperties (&p, 0);
  printf("Compute capability: %d.%d\n", p.major, p.minor);
  printf("Name: %s\n" , p.name);
  printf("\n\n");
}


// Fills data with values
unsigned char func(int x, int y, int z){
  unsigned char value = rand() % 20;

  int x1 = 300;
  int y1 = 400;
  int z1 = 100;
  float dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

  if(dist < 100){
    value  = 30;
  }

  x1 = 100;
  y1 = 200;
  z1 = 400;
  dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

  if(dist < 50){
    value = 50;
  }

  if(x > 200 && x < 300 && y > 300 && y < 500 && z > 200 && z < 300){
    value = 45;
  }
  if(x > 0 && x < 100 && y > 250 && y < 400 && z > 250 && z < 400){
    value =35;
  }
  return value;
}

unsigned char* create_data(){
  unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_DIM*DATA_DIM*DATA_DIM);

  for(int i = 0; i < DATA_DIM; i++){
    for(int j = 0; j < DATA_DIM; j++){
      for(int k = 0; k < DATA_DIM; k++){
        data[i*DATA_DIM*DATA_DIM + j*DATA_DIM+ k]= func(k,j,i);
      }
    }
  }

  return data;
}

// Checks if position is inside the volume (float3 and int3 versions)
__device__ __host__ int inside(float3 pos){
  int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
  int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
  int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

  return x && y && z;
}

__device__ __host__ int inside(int3 pos){
  int x = (pos.x >= 0 && pos.x < DATA_DIM);
  int y = (pos.y >= 0 && pos.y < DATA_DIM);
  int z = (pos.z >= 0 && pos.z < DATA_DIM);

  return x && y && z;
}

// Indexing function (note the argument order)
__device__ __host__ int index(int z, int y, int x){
  return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Trilinear interpolation
__device__ __host__ float value_at(float3 pos, unsigned char* data){
  if(!inside(pos)){
    return 0;
  }

  int x = floor(pos.x);
  int y = floor(pos.y);
  int z = floor(pos.z);

  int x_u = ceil(pos.x);
  int y_u = ceil(pos.y);
  int z_u = ceil(pos.z);

  float rx = pos.x - x;
  float ry = pos.y - y;
  float rz = pos.z - z;

  float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
  float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
  float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
  float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];

  float b0 = ry*a0 + (1-ry)*a1;
  float b1 = ry*a2 + (1-ry)*a3;

  float c0 = rz*b0 + (1-rz)*b1;


  return c0;
}


// Serial ray casting
unsigned char* raycast_serial(unsigned char* data, unsigned char* region){
  unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

  // Camera/eye position, and direction of viewing. These can be changed to look
  // at the volume from different angles.
  float3 camera = {.x=1000,.y=1000,.z=1000};
  float3 forward = {.x=-1, .y=-1, .z=-1};
  float3 z_axis = {.x=0, .y=0, .z = 1};

  // Finding vectors aligned with the axis of the image
  float3 right = cross(forward, z_axis);
  float3 up = cross(right, forward);

  // Creating unity lenght vectors
  forward = normalize(forward);
  right = normalize(right);
  up = normalize(up);

  float fov = 3.14/4;
  float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
  float step_size = 0.5;

  // For each pixel
  for(int y = -(IMAGE_DIM/2); y < (IMAGE_DIM/2); y++){
    for(int x = -(IMAGE_DIM/2); x < (IMAGE_DIM/2); x++){

      // Find the ray for this pixel
      float3 screen_center = add(camera, forward);
      float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
      ray = add(ray, scale(camera, -1));
      ray = normalize(ray);
      float3 pos = camera;

      // Move along the ray, we stop if the color becomes completely white,
      // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
      // to be big enough to let rays pass through the entire volume)
      int i = 0;
      float color = 0;
      while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at(pos, region);                  // Check if we're in the region
        color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
      }

      // Write final color to image
      image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
    }
  }

  return image;
}


// Check if two values are similar, threshold can be changed.
__device__ __host__ int similar(unsigned char* data, int3 a, int3 b){
  unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
  unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

  int i = abs(va-vb) < 1;
  return i;
}


// Serial region growing, same algorithm as in assignment 2
unsigned char* grow_region_serial(unsigned char* data){
  unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);

  stack_t* stack = new_stack();

  int3 seed = {.x=50, .y=300, .z=300};
  push(stack, seed);
  region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 1;

  int dx[6] = {-1,1,0,0,0,0};
  int dy[6] = {0,0,-1,1,0,0};
  int dz[6] = {0,0,0,0,-1,1};

  while(stack->size > 0){
    int3 pixel = pop(stack);
    for(int n = 0; n < 6; n++){
      int3 candidate = pixel;
      candidate.x += dx[n];
      candidate.y += dy[n];
      candidate.z += dz[n];

      if(!inside(candidate)){
        continue;
      }

      if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
        continue;
      }

      if(similar(data, pixel, candidate)){
        push(stack, candidate);
        region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 1;
      }
    }
  }

  return region;
}

// This kernel will run one time per pixel. Each thread initialize the same variables and run it's own
// iteration of the for loop from the serial implementation we got from the TA.
__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){

  // Camera/eye position, and direction of viewing. These can be changed to look
  // at the volume from different angles.
  float3 camera = {.x=1000,.y=1000,.z=1000};
  float3 forward = {.x=-1, .y=-1, .z=-1};
  float3 z_axis = {.x=0, .y=0, .z = 1};

  // Finding vectors aligned with the axis of the image
  float3 right = cross(forward, z_axis);
  float3 up = cross(right, forward);

  // Creating unity lenght vectors
  forward = normalize(forward);
  right = normalize(right);
  up = normalize(up);

  float fov = 3.14/4;
  float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
  float step_size = 0.5;

  int x = -IMAGE_DIM/2 + blockIdx.x * blockDim.x + threadIdx.x;
  int y = -IMAGE_DIM/2 + blockIdx.y * blockDim.y + threadIdx.y;

  // Find the ray for this pixel
  float3 screen_center = add(camera, forward);
  float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
  ray = add(ray, scale(camera, -1));
  ray = normalize(ray);
  float3 pos = camera;

  // Move along the ray, we stop if the color becomes completely white,
  // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
  // to be big enough to let rays pass through the entire volume)
  int i = 0;
  float color = 0;
  while(color < 255 && i < 5000){
    i++;
    pos = add(pos, scale(ray, step_size));          // Update position
    int r = value_at(pos, region);                  // Check if we're in the region
    color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
  }
  // Write final color to image
  image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
}

// Trilinear interpolation from the region texture
__device__ float value_at_region(float3 pos){
  if(!inside(pos)){
    return 0;
  }

  int x = floor(pos.x);
  int y = floor(pos.y);
  int z = floor(pos.z);

  int x_u = ceil(pos.x);
  int y_u = ceil(pos.y);
  int z_u = ceil(pos.z);

  float rx = pos.x - x;
  float ry = pos.y - y;
  float rz = pos.z - z;

  float a0 = rx*tex3D(region_texture,z,y,x) + (1-rx)*tex3D(region_texture,z,y,x_u);
  float a1 = rx*tex3D(region_texture,z,y_u,x) + (1-rx)*tex3D(region_texture,z,y_u,x_u);
  float a2 = rx*tex3D(region_texture,z_u,y,x) + (1-rx)*tex3D(region_texture,z_u,y,x_u);
  float a3 = rx*tex3D(region_texture,z_u,y_u,x) + (1-rx)*tex3D(region_texture,z_u,y_u,x_u);

  float b0 = ry*a0 + (1-ry)*a1;
  float b1 = ry*a2 + (1-ry)*a3;

  float c0 = rz*b0 + (1-rz)*b1;


  return c0;
}
// Trilinear interpolation from the data texture
__device__ float value_at_data(float3 pos){
  if(!inside(pos)){
    return 0;
  }

  int x = floor(pos.x);
  int y = floor(pos.y);
  int z = floor(pos.z);

  int x_u = ceil(pos.x);
  int y_u = ceil(pos.y);
  int z_u = ceil(pos.z);

  float rx = pos.x - x;
  float ry = pos.y - y;
  float rz = pos.z - z;

  float a0 = rx*tex3D(data_texture,z,y,x) + (1-rx)*tex3D(data_texture,z,y,x_u);
  float a1 = rx*tex3D(data_texture,z,y_u,x) + (1-rx)*tex3D(data_texture,z,y_u,x_u);
  float a2 = rx*tex3D(data_texture,z_u,y,x) + (1-rx)*tex3D(data_texture,z_u,y,x_u);
  float a3 = rx*tex3D(data_texture,z_u,y_u,x) + (1-rx)*tex3D(data_texture,z_u,y_u,x_u);

  float b0 = ry*a0 + (1-ry)*a1;
  float b1 = ry*a2 + (1-ry)*a3;

  float c0 = rz*b0 + (1-rz)*b1;


  return c0;
}


// This kernel will run one time per pixel. Each thread initialize the same variables and run it's own
// iteration of the for loop from the serial implementation we got from the TA.
// All data from data and region is read from a 3D texture.
// TODO: Find the bug that make this output a few random cubes.
__global__ void raycast_kernel_texture(unsigned char* image){
  // Camera/eye position, and direction of viewing. These can be changed to look
  // at the volume from different angles.
  float3 camera = {.x=1000,.y=1000,.z=1000};
  float3 forward = {.x=-1, .y=-1, .z=-1};
  float3 z_axis = {.x=0, .y=0, .z = 1};

  // Finding vectors aligned with the axis of the image
  float3 right = cross(forward, z_axis);
  float3 up = cross(right, forward);

  // Creating unity lenght vectors
  forward = normalize(forward);
  right = normalize(right);
  up = normalize(up);

  float fov = 3.14/4;
  float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
  float step_size = 0.5;

  int x = -IMAGE_DIM/2 + blockIdx.x * blockDim.x + threadIdx.x;
  int y = -IMAGE_DIM/2 + blockIdx.y * blockDim.y + threadIdx.y;

  // Find the ray for this pixel
  float3 screen_center = add(camera, forward);
  float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
  ray = add(ray, scale(camera, -1));
  ray = normalize(ray);
  float3 pos = camera;

  // Move along the ray, we stop if the color becomes completely white,
  // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
  // to be big enough to let rays pass through the entire volume)
  int i = 0;
  float color = 0;
  while(color < 255 && i < 5000){
    i++;
    pos = add(pos, scale(ray, step_size));          // Update position
    int r = value_at_region(pos);             // Check if we're in the region
    color += value_at_data(pos)*(0.01 + r) ;  // Update the color based on data value, and if we're in the region

  }
  // Write final color to image
  image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;

}

// This function loads data and region into the memory of the gpu. After the data is loaded
// it run a thread on the gpu for each pixel in the image. After all threads are done
// the finished image is loaded from the gpu to the main memory.
unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
  unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

  unsigned char *data_device;
  unsigned char *region_device;
  unsigned char* image_device;
  cudaMalloc( (void**)&data_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMalloc( (void**)&region_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMalloc( (void**)&image_device,DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMemcpy( data_device, data, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy( region_device, region, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy( image_device, image, DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);

  dim3 dimBlock( 32, 32 );
  dim3 dimGrid( 16, 16 );

  raycast_kernel<<<dimGrid, dimBlock>>>(data_device, image_device, region_device);

  cudaMemcpy( image, image_device, DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(data_device);
  cudaFree(region_device);
  cudaFree(image_device);

  return image;
}


// This function loads data and region into the texture memory of the gpu. After the data is loaded
// it run a thread on the gpu for each pixel in the image. After all threads are done
// the finished image is loaded from the gpu to the main memory.
// TODO: Find the bug that make this output a few random cubes.
unsigned char* raycast_gpu_texture(unsigned char* data, unsigned char* region){
  unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

  unsigned char* image_device;
  cudaMalloc( (void**)&image_device,DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMemcpy( image_device, image, DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);

  data_texture.filterMode=cudaFilterModePoint;
  data_texture.addressMode[0]=cudaAddressModeWrap; 
  data_texture.addressMode[1]=cudaAddressModeWrap; 
  data_texture.addressMode[2]=cudaAddressModeClamp; 
  region_texture.filterMode=cudaFilterModePoint;
  region_texture.addressMode[0]=cudaAddressModeWrap; 
  region_texture.addressMode[1]=cudaAddressModeWrap; 
  region_texture.addressMode[2]=cudaAddressModeClamp; 

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaExtent extent = make_cudaExtent(512,512,512);

  cudaArray* data_array;
  cudaArray* region_array;

  cudaMalloc3DArray(&data_array, &channelDesc, extent);
  cudaMalloc3DArray(&region_array, &channelDesc, extent);

  cudaMemcpy3DParms copyDataParams    = {0};
  cudaMemcpy3DParms copyRegionParams  = {0};
  copyDataParams.srcPtr     = make_cudaPitchedPtr(data, 512*sizeof(unsigned char), 512, 512);
  copyRegionParams.srcPtr   = make_cudaPitchedPtr(region, 512*sizeof(unsigned char), 512, 512);
  copyDataParams.dstArray   = data_array;
  copyRegionParams.dstArray = data_array;
  copyDataParams.extent     = extent;
  copyRegionParams.extent   = extent;
  copyDataParams.kind       = cudaMemcpyHostToDevice;
  copyRegionParams.kind     = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&copyDataParams);
  cudaMemcpy3D(&copyRegionParams);

  cudaBindTextureToArray(data_texture, data_array, channelDesc);
  cudaBindTextureToArray(region_texture, region_array, channelDesc);

  dim3 dimBlock( 32, 32 );
  dim3 dimGrid( 16, 16 );

  raycast_kernel_texture<<<dimGrid, dimBlock>>>(image_device);

  cudaMemcpy( image, image_device, DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(image_device);

  return image;
}

// Each thread launched with this kernel will check if it is in a border pixel. If it is
// it will run 40 iterations of the serial grow algorithm from the TA.
__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int dx[6] = {-1,1,0,0,0,0};
  int dy[6] = {0,0,-1,1,0,0};
  int dz[6] = {0,0,0,0,-1,1};

  int3 pixel = {.x=x, .y=y, .z=z};
  for(int i=0;i<40;i++) {
    if(region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x]==2){
      region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x] = 1;
      for(int n = 0; n < 6; n++){
        int3 candidate = pixel;
        candidate.x += dx[n];
        candidate.y += dy[n];
        candidate.z += dz[n];

        if(!inside(candidate)){
          continue;
        }

        if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
          continue;
        }

        if(similar(data, pixel, candidate)){
          region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 2;
          finished[0]=0;
        }
      }
    }
  }
}


// Each thread launched with this kernel will check if it is in a region border pixel. If it is
// it will run 40 iterations of the serial grow algorithm from the TA. Each block of the gpu will
// use a shared memory during the 40 iterations and syncronize with eachother when they are all done.
// TODO: This currently doesn't do anything. It should be fixed.
__global__ void region_grow_kernel_shared(unsigned char* data, unsigned char* region, int* finished){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int dx[6] = {-1,1,0,0,0,0};
  int dy[6] = {0,0,-1,1,0,0};
  int dz[6] = {0,0,0,0,-1,1};

  int3 pixel = {.x=x, .y=y, .z=z};


  __shared__ unsigned char shared_data[DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char)];
  __shared__ unsigned char shared_region[DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char)];
  shared_data[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x] = data[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x];
  shared_region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x] = region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x];


  __syncthreads();
  if(threadIdx.x!=0&&threadIdx.y!=0&&threadIdx.z!=0
      &&threadIdx.x!=blockDim.x-1&&threadIdx.y!=blockDim.y-1&&threadIdx.z!=blockDim.z-1) {
    for(int i=0;i<40;i++) {
      if(shared_region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x]==2){
        shared_region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x] = 1;
        for(int n = 0; n < 6; n++){
          int3 candidate = pixel;
          candidate.x += dx[n];
          candidate.y += dy[n];
          candidate.z += dz[n];

          if(!inside(candidate)){
            continue;
          }

          if(shared_region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
            continue;
          }

          if(similar(shared_data, pixel, candidate)){
            shared_region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 2;
            finished[0]=0;
          }
        }
      }
    }
  }
  region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x] = shared_region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x];
}

// This function creates an empty region and upload it and the data to the gpu.
// When the data is ready it will launch a thread on the gpu for each point in
// the region. The threads will run 40 iterations each before terminating.
// The cpu will then check if there was any extended borders during the execution,
// and run 40 new iterations until there is no updates.
unsigned char* grow_region_gpu(unsigned char* data){
  unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);
  int* finished = (int*)malloc(sizeof(int));
    finished[0] = 0;

  int3 seed = {.x=50, .y=300, .z=300};
  region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 2;

  unsigned char *data_device;
  unsigned char *region_device;
  int *finished_device;
  cudaMalloc( (void**)&data_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMalloc( (void**)&region_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMalloc( (void**)&finished_device, sizeof(int));
  cudaMemcpy( data_device, data, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy( region_device, region, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);

  dim3 dimBlock( 8, 8, 8 );
  dim3 dimGrid( 64, 64, 64 );

  while(finished[0] == 0){
    finished[0] = 1;
    cudaMemcpy( finished_device, finished, sizeof(int), cudaMemcpyHostToDevice);
    region_grow_kernel<<<dimGrid, dimBlock>>>(data_device, region_device, finished_device);
    cudaMemcpy( finished, finished_device, sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy( region, region_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(data_device);
  cudaFree(region_device);
  cudaFree(finished_device);
  return region;
}


// This function creates an empty region and upload it and the data to the gpu.
// When the data is ready it will launch a thread on the gpu for each point in
// the region. The threads will run 40 iterations each before terminating.
// The cpu will then check if there was any extended borders during the execution,
// and run 40 new iterations until there is no updates.
// This is identical to the non shared memory version, but it will run a different
// kernel.
// TODO: This currently doesn't do anything. It should be fixed.
unsigned char* grow_region_gpu_shared(unsigned char* data){
  unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);
  int* finished = (int*)malloc(sizeof(int));
    finished[0] = 0;

  int3 seed = {.x=50, .y=300, .z=300};
  region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 2;

  unsigned char *data_device;
  unsigned char *region_device;
  int *finished_device;
  cudaMalloc( (void**)&data_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMalloc( (void**)&region_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char));
  cudaMalloc( (void**)&finished_device, sizeof(int));
  cudaMemcpy( data_device, data, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy( region_device, region, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice);

  dim3 dimBlock( 8, 8, 8 );
  dim3 dimGrid( 64, 64, 64 );

  while(finished[0] == 0){
    finished[0] = 1;
    cudaMemcpy( finished_device, finished, sizeof(int), cudaMemcpyHostToDevice);
    region_grow_kernel_shared<<<dimGrid, dimBlock>>>(data_device, region_device, finished_device);
    cudaMemcpy( finished, finished_device, sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy( region, region_device, DATA_DIM*DATA_DIM*DATA_DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(data_device);
  cudaFree(region_device);
  cudaFree(finished_device);
  printf("seed: %i\n", region[50 *DATA_DIM*DATA_DIM + 300*DATA_DIM + 300]);
  return region;
}


void print_time(struct timeval start, struct timeval end){
  long int ms = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
  double s = ms/1e6;
  printf("Time : %f s\n", s);
}

int main(int argc, char** argv){

  print_properties();

  struct timeval start, end;

  unsigned char* data = create_data();
  unsigned char* region;
  unsigned char* image;
  int all = 1;
  if(all) {
    gettimeofday(&start, NULL);
    region = grow_region_gpu_shared(data);
    gettimeofday(&end, NULL);
    printf("grow_region_gpu_shared:\n");
    print_time(start, end);

    gettimeofday(&start, NULL);
    region = grow_region_gpu(data);
    gettimeofday(&end, NULL);
    printf("grow_region_gpu:\n");
    print_time(start, end);

    gettimeofday(&start, NULL);
    region = grow_region_serial(data);
    gettimeofday(&end, NULL);
    printf("grow_region_serial:\n");
    print_time(start, end);


    gettimeofday(&start, NULL);
    image = raycast_gpu_texture(data, region);
    gettimeofday(&end, NULL);
    printf("raycast_gpu_texture:\n");
    print_time(start, end);

    gettimeofday(&start, NULL);
    image = raycast_gpu(data, region);
    gettimeofday(&end, NULL);
    printf("raycast_gpu:\n");
    print_time(start, end);

    gettimeofday(&start, NULL);
    image = raycast_serial(data, region);
    gettimeofday(&end, NULL);
    printf("raycast_serial:\n");
    print_time(start, end);
} else {
    gettimeofday(&start, NULL);
    region = grow_region_gpu(data);
    gettimeofday(&end, NULL);
    printf("grow_region_gpu:\n");
    print_time(start, end);

    gettimeofday(&start, NULL);
    image = raycast_gpu(data, region);
    gettimeofday(&end, NULL);
    printf("raycast_gpu:\n");
    print_time(start, end);
  }
  write_bmp(image, IMAGE_DIM, IMAGE_DIM);
}

