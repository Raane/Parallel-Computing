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
  int a = im[p.x +  p.y * (local_image_size[1]+2)];
  int b = im[q.x +  q.y * (local_image_size[1]+2)];
  int diff = abs(a-b);
  return diff < 2;
}


// Create and commit MPI datatypes
void create_types(){
  /* 
   * I never got this working, but I really should have a type for sending the whole local
   * image as a single send, and vertical region borders as a single send.
   */
}


// Send image from rank 0 to all ranks, from image to local_image
// This does not send the border of the images, as these can be distributed paralell between the neighbours.
void distribute_image(){
  for(int dest=0;dest<size;dest++) { // Rank 0 will send the image to each of the other processes.
    if(dest==0) {
      for(int row=0;row<image_size[1];row++) {
        // Rank 0 itself do not need to send the data, and can simply copy them from memory.
        memcpy(
            (local_image + (local_image_size[0] + 2) * ( row + 1 ) + 1),
            (image + (coords[1] * local_image_size[1] + row) * image_size[0] + coords[0] * local_image_size[0]),
            local_image_size[0]
            );
      }
    } else {
      for(int row=0;row<image_size[1];row++) {
        // Send each row of the local image as a MPI_Sends, this can be done more efficiently with a cutsom MPI type, but I never got it working.
        MPI_Send(
            (image + (coords[1] * local_image_size[1] + row) * image_size[0] + coords[0] * local_image_size[0]), 
            local_image_size[0], MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
      }
    }
  }
}

// All but rank 0 will use this to receive the local image
void receive_image(){
  for(int row=0;row<image_size[1];row++) {
    MPI_Recv(
        (local_image + (local_image_size[0] + 2) * ( row + 1 ) + 1),
        local_image_size[0], MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  }
}



// Sends the border of the local image to one of the 4 neighbours of this node.
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
    if(direction%2==0) {
      // Horizontal borders
      int row = direction==0?0:local_image_size[1];
      MPI_Send(
          (local_image + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
    } else {
      // Vertical borders
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

// Receive the border of the local image from one of the 4 neightbours of this node.
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
    if(direction%2==0) {
      // Horizontal borders.
      int row = direction==0?0:local_image_size[1];
      MPI_Recv(
          (local_image + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
    } else {
      // Vertical borders
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

// Distribute the local image border to each neighbour of this node.
// This is done by dividing the mesh of nodes in a checkerboard pattern and let half send and half receive at any time.
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

// Send the border of the local region to a neighbour
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
      // Horizontal borders.
      int row = direction==0?0:local_image_size[1];
      MPI_Send(
          (&local_region + row * (local_image_size[0] + 2) + 1), 
          local_image_size[0], MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
    } else {
      // Vertical borders.
      int col = direction==3?0:local_image_size[0]+2;
      for(int row=1;row<local_image_size[1]-1;row++) {
        MPI_Send(
            (&local_region + row * (local_image_size[0] + 2) + col), 
            1, MPI_UNSIGNED_CHAR, recipient, 0, MPI_COMM_WORLD);
      }
    }
  }
}

// Receive the border of the local region from a neighbour
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
      // Horizontal borders.
      int row = direction==2?0:local_image_size[1];
      unsigned char* received_border = malloc(local_image_size[0] * sizeof(unsigned char));
      MPI_Recv(received_border, local_image_size[0], MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
      for(int i=0;i<local_image_size[0];i++) {
        if(received_border[i] == 1) {
          if(local_region[i + 1 + row * (local_image_size[0]+2)] != 1) {
            pixel_t pixel;
            pixel.x = i + 1;
            pixel.y = row;;
            push(stack, pixel);
          }
        }
      }
      free(received_border);
    } else {
      // Vertical borders.
      for(int row=1;row<local_image_size[1]-1;row++) {
        int col = direction==1?0:local_image_size[0]+2;
        unsigned char* received_border = malloc(sizeof(unsigned char));
        MPI_Recv(received_border, 1, MPI_UNSIGNED_CHAR, sender, 0, MPI_COMM_WORLD, &status);
        if(received_border[0] == 1) {
          if(local_region[col + row * (local_image_size[0]+2)] != 1) {
            pixel_t pixel;
            pixel.x = col;
            pixel.y = row;;
            push(stack, pixel);
          }
        }
        free(received_border);
      }
    }
  }
}

// Exchange borders with neighbour ranks
// This is done by dividing the mesh of nodes in a checkerboard pattern and let half send and half receive at any time.
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

void clear_local_region() {
  int lsize = local_image_size[0]*local_image_size[1];
  int lsize_border = (local_image_size[0] + 2)*(local_image_size[1] + 2);
  local_region = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);
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

// Determine if all ranks are finished.
// This is done by summing the stacksize across all nodes using reduce all.
int finished(stack_t* stack){
  int global_sum;
  MPI_Allreduce(&(stack->size), &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return global_sum == 0;
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
    seed.x = seeds[i*2] - coords[1]*(local_image_size[1]+2);
    seed.y = seeds[i*2+1] - coords[0]*(local_image_size[0]+2);

    if(inside(seed)){
      push(stack, seed);
    }
  }
}


// Region growing, paralell implementation.
void grow_region(stack_t* stack) {

  add_seeds(stack);

  while(stack->size > 0){
    pixel_t pixel = pop(stack);

    local_region[pixel.y * (local_image_size[1]+2) + pixel.x] = 1;

    int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
    for(int c = 0; c < 4; c++){
      pixel_t candidate;
      candidate.x = pixel.x + dx[c];
      candidate.y = pixel.y + dy[c];

      if(!inside(candidate)){
        continue;
      }


      if(local_region[candidate.y * (local_image_size[1]+2) + candidate.x]){
        continue;
      }

      if(similar(local_image, pixel, candidate)){
        local_region[candidate.x + candidate.y * (local_image_size[1] + 2)] = 1;
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

void write_image(){
  if(rank==0){
    for(int i = 0; i < image_size[0]*image_size[1]; i++){
      image[i] *= (region[i] == 0);
    }
    write_bmp(image, image_size[0], image_size[1]);
  }
}

int main(int argc, char** argv){

  // Initializing everything and distributing data.
  init_mpi(argc, argv);
  stack_t* stack = new_stack();
  load_and_allocate_images(argc, argv);
  if(rank==0) {
    distribute_image();
    distribute_image_border();
  } else {
    receive_image();
    distribute_image_border();
  }

  clear_local_region();

  // Doing the actual parallel image processing.
  do{
    grow_region(stack);
    exchange(stack);
  } while(!finished(stack));

  // Gathering data, writing to file and finalizing.
  if(rank==0) {
    gather_region();
    MPI_Finalize();
    write_image();
  } else {
    gather_region();
    MPI_Finalize();
  }

  exit(0);
}
