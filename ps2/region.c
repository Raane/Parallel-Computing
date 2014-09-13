#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include "bmp.h"

typedef struct{
  int x;
  int y;
} pixel_t;

typedef struct{
  int size;
  int buffer_size;
  pixel_t* pixels;
} stack_t;


// Global variables
int rank,                       // MPI rank
    size,                       // Number of MPI processes
    dims[2],                    // Dimensions of MPI grid
    coords[2],                  // Coordinate of this rank in MPI grid
    periods[2] = {0,0},         // Periodicity of grid
    north,south,east,west,      // Four neighbouring MPI ranks
    image_size[2] = {512,512},  // Hard coded image size
    local_image_size[2];        // Size of local part of image (not including border)


MPI_Comm cart_comm;             // Cartesian communicator
MPI_Status status;              // MPI status


// MPI datatypes, you may have to add more.
MPI_Datatype border_row_t,
             border_col_t;


unsigned char *image,           // Entire image, only on rank 0
              *region,          // Region bitmap. 1 if in region, 0 elsewise
              *local_image,     // Local part of image
              *local_region;    // Local part of region bitmap


// Create new pixel stack
stack_t* new_stack(){
  stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
  stack->size = 0;
  stack->buffer_size = 1024;
  stack->pixels = (pixel_t*)malloc(sizeof(pixel_t*)*1024);
}


// Push on pixel stack
void push(stack_t* stack, pixel_t p){
  if(stack->size == stack->buffer_size){
    stack->buffer_size *= 2;
    stack->pixels = realloc(stack->pixels, sizeof(pixel_t)*stack->buffer_size);
  }
  stack->pixels[stack->size] = p;
  stack->size += 1;
}


// Pop from pixel stack
pixel_t pop(stack_t* stack){
  stack->size -= 1;
  return stack->pixels[stack->size];
}


// Check if two pixels are similar. The hardcoded threshold can be changed.
// More advanced similarity checks could have been used.
int similar(unsigned char* im, pixel_t p, pixel_t q){
  int a = im[p.x +  p.y * image_size[1]];
  int b = im[q.x +  q.y * image_size[1]];
  int diff = abs(a-b);
  return diff < 2;
}


// Create and commit MPI datatypes
void create_types(){

}


// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(){
  for(int dest=0;dest<size;dest++) {
    if(dest==0) {
      for(int row=0;row<image_size[1];row++) {
        memcpy(
          (local_image + (local_image_size[0] + 2) * ( row + 1 ) + 1),
          (image + (coords[1] * local_image_size[1] + row) * image_size[0] + coords[0] * local_image_size[0]),
          local_image_size[0]
          );
      }
    } else {
      for(int row=0;row<image_size[1];row++) {
        MPI_Send(
            (image + (coords[1] * local_image_size[1] + row) * image_size[0] + coords[0] * local_image_size[0]), 
            local_image_size[0], MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
      }
    }
  }
}

void send_image_border(int direction) {
  int recipient;
  switch(direction) {
    case 0: recipient = north; break;
    case 1: recipient = east; break;
    case 2: recipient = south; break;
    case 3: recipient = west; break;
    default: recipient = -1;
  }
  if(recipient!=-2) {
    //printf("Rank %d sending to rank %d\n", rank, recipient);
    if(direction%2==0) {
      int row = direction==0?0:local_image_size[1];
      MPI_Send(
          (local_image + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
    } else {
      for(int row=1;row<local_image_size[1]-1;row++) {
        MPI_Send(
            (local_image + row * (local_image_size[0] + 2)), 
            1, MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
        MPI_Send(
            (local_image + row * (local_image_size[0] + 2) + local_image_size[0]+2), 
            1, MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
      }
    }
  }
}

void receive_image_border(int direction) {
  int sender;
  switch(direction) {
    case 0: sender = north; break;
    case 1: sender = east; break;
    case 2: sender = south; break;
    case 3: sender = west; break;
    default: sender = -1;
  }
  if(sender!=-2) {
    //printf("Rank %d receiving from rank %d\n", rank, sender);
    if(direction%2==0) {
      int row = direction==0?0:local_image_size[1];
      MPI_Recv(
          (local_image + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
    } else {
      for(int row=1;row<local_image_size[1]-1;row++) {
        MPI_Recv(
            (local_image + row * (local_image_size[0] + 2)), 
            1, MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(
            (local_image + row * (local_image_size[0] + 2) + local_image_size[0]+2), 
            1, MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
      }
      
    }
  }
}


void distribute_image_border(){
  for(int direction=0;direction<4;direction++) {
    if((rank+(rank/dims[0]))%2==0) {
      send_image_border(direction);
      receive_image_border(direction);
    } else {
      receive_image_border((direction+2)%4);
      send_image_border((direction+2)%4);
    }
  }
}

void send_region_border(int direction, stack_t* stack) {
  int recipient;
  switch(direction) {
    case 0: recipient = north; break;
    case 1: recipient = east; break;
    case 2: recipient = south; break;
    case 3: recipient = west; break;
    default: recipient = -1;
  }
  if(recipient!=-2) {
    if(direction%2==0) {
      int row = direction==0?0:local_image_size[1];
      MPI_Send(
          (local_region + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
    } else {
      int col = direction==3?0:local_image_size[0]+2;
      for(int row=1;row<local_image_size[1]-1;row++) {
        MPI_Send(
            (local_region + row * (local_image_size[0] + 2) + col), 
            1, MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
      }
    }
  }
}

void receive_region_border(int direction, stack_t* stack) {
  int sender;
  switch(direction) {
    case 0: sender = north; break;
    case 1: sender = east; break;
    case 2: sender = south; break;
    case 3: sender = west; break;
    default: sender = -1;
  }
  if(sender!=-2) {
    if(direction%2==0) {
      int row = direction==0?0:local_image_size[1];
      unsigned char* received_border = malloc(local_image_size[0] * sizeof(unsigned char));
      MPI_Recv(&received_border, local_image_size[0], MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
      for(int i=0;i<local_image_size[0];i++) {
        if(received_border[i] == 1) {
          pixel_t pixel;
          pixel.x = i + 1;
          pixel.y = row;;
          push(stack, pixel);
        }
      }
      free(received_border);
    } else {
      for(int row=1;row<local_image_size[1]-1;row++) {
        int col = direction==3?0:local_image_size[0]+2;
        unsigned char* received_border = malloc(sizeof(unsigned char));
        MPI_Recv(&received_border, 1, MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
        if(received_border[0] == 1) {
          pixel_t pixel;
          pixel.x = col;
          pixel.y = row;;
          push(stack, pixel);
        }
        free(received_border);
      }
    }
  }
}

// Exchange borders with neighbour ranks
void exchange(stack_t* stack){
  for(int direction=0;direction<4;direction++) {
    if((rank+(rank/dims[0]))%2==0) {
      send_region_border(direction, stack);
      receive_region_border(direction, stack);
    } else {
      receive_region_border((direction+2)%4, stack);
      send_region_border((direction+2)%4, stack);
    }
  }
}


// Gather region bitmap from all ranks to rank 0, from local_region to region
void gather_region(){
  if(rank==0) {
    for(int sender=1;sender<size;sender++) {
      for(int row=1;row<local_image_size[1]+1;row++) {
        int sender_x = sender%dims[0];
        int sender_y = sender/dims[0];
        MPI_Recv(
            (region + (sender_y * local_image_size[1] + row) * image_size[0] + sender_x * local_image_size[0]),
            local_image_size[0], MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
      }
    }
    for(int row=0;row<local_image_size[1];row++) {
      memcpy(
          (region + (coords[1] * local_image_size[1] + row) * image_size[0] + coords[0] * local_image_size[0]),
          (local_region + (local_image_size[0] + 2) * ( row + 1 ) + 1),
          local_image_size[0]
          );
    }
  } else {
    for(int row=1;row<local_image_size[1]+1;row++) {
      MPI_Send(
          (local_region + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
}

// Determine if all ranks are finished. You may have to add arguments.
// You dont have to have this check as a seperate function
int finished(){

}

// All but rank 0 will use this to receive the image
void receive_image(){
  for(int row=0;row<image_size[1];row++) {
    MPI_Recv(
        (local_image + (local_image_size[0] + 2) * ( row + 1 ) + 1),
        local_image_size[0], MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  }
}



// Check if pixel is inside local image
int inside(pixel_t p){
  return (p.x >= 0 && p.x < local_image_size[1] && p.y >= 0 && p.y < local_image_size[0]);
}


// Adding seeds in corners.
void add_seeds(stack_t* stack){
  int seeds [8];
  seeds[0] = 5;
  seeds[1] = 5;
  seeds[2] = image_size[1]-5;
  seeds[3] = 5;
  seeds[4] = image_size[1]-5;
  seeds[5] = image_size[0]-5;
  seeds[6] = 5;
  seeds[7] = image_size[0]-5;

  for(int i = 0; i < 4; i++){
    pixel_t seed;
    seed.x = seeds[i*2] - coords[1]*local_image_size[1];
    seed.y = seeds[i*2+1] - coords[0]*local_image_size[0];

    if(inside(seed)){
      push(stack, seed);
    }
  }
}


// Region growing, serial implementation
void grow_region(){

  stack_t* stack = new_stack();
  add_seeds(stack);

  while(stack->size > 0){
    pixel_t pixel = pop(stack);

    region[pixel.y * image_size[1] + pixel.x] = 1;


    int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
    for(int c = 0; c < 4; c++){
      pixel_t candidate;
      candidate.x = pixel.x + dx[c];
      candidate.y = pixel.y + dy[c];

      if(!inside(candidate)){
        continue;
      }


      if(region[candidate.y * image_size[1] + candidate.x]){
        continue;
      }

      if(similar(image, pixel, candidate)){
        region[candidate.x + candidate.y * image_size[1]] = 1;
        push(stack,candidate);
      }
    }
  }
}


// MPI initialization, setting up cartesian communicator
void init_mpi(int argc, char** argv){
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm );
  MPI_Cart_coords( cart_comm, rank, 2, coords );

 /* printf("dim 0: %d, ", dims[0]);
  printf("dim 1: %d, ", dims[0]);
  printf("coor 0: %d, ", coords[0]);
  printf("coor 1: %d\n", coords[1]);*/

  MPI_Cart_shift( cart_comm, 0, 1, &north, &south );
  MPI_Cart_shift( cart_comm, 1, 1, &west, &east );
}


void load_and_allocate_images(int argc, char** argv){

  if(argc != 2){
    printf("Useage: region file");
    exit(-1);
  }

  if(rank == 0){
    image = read_bmp(argv[1]);
    region = (unsigned char*)calloc(sizeof(unsigned char),image_size[0]*image_size[1]);
  }

  local_image_size[0] = image_size[0]/dims[0];
  local_image_size[1] = image_size[1]/dims[1];

  int lsize = local_image_size[0]*local_image_size[1];
  int lsize_border = (local_image_size[0] + 2)*(local_image_size[1] + 2);
  local_image = (unsigned char*)malloc(sizeof(unsigned char)*lsize_border);
  local_region = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);
}

void load_and_allocate_local_images() {
  local_image_size[0] = image_size[0]/dims[0];
  local_image_size[1] = image_size[1]/dims[1];

  int lsize = local_image_size[0]*local_image_size[1];
  int lsize_border = (local_image_size[0] + 2)*(local_image_size[1] + 2);
  local_image = (unsigned char*)malloc(sizeof(unsigned char)*lsize_border);
  local_region = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);
}


void write_image(){
  if(rank==0){
    for(int i = 0; i < image_size[0]*image_size[1]; i++){

      image[i] *= (region[i] == 0);
    }
    write_bmp(image, image_size[0], image_size[1]);
  }
}


int main(int argc, char** argv){

  init_mpi(argc, argv);
  if(rank==0) {

    load_and_allocate_images(argc, argv);

    create_types();

    printf("Starting dirtribute image\n");

    distribute_image();
    
    printf("Starting dirtribute image border\n");

    distribute_image_border();

    printf("Starting growing region\n");

    grow_region();

    printf("Starting gather\n");

    gather_region();

    printf("Finalize\n");

    MPI_Finalize();

    write_image();
  } else {
    load_and_allocate_local_images(argc, argv);
    receive_image();
    distribute_image_border();
    MPI_Finalize();
  }
    //printf("rank(%d): n,e,s,w: %d %d %d %d\n", rank, north, east, south, west);

  exit(0);
}
