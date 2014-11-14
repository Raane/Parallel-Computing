// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 64

int ps5_inside(int3 pos){
  int x = (pos.x >= 0 && pos.x < DATA_DIM);
  int y = (pos.y >= 0 && pos.y < DATA_DIM);
  int z = (pos.z >= 0 && pos.z < DATA_DIM);

  return x && y && z;
}

// Check if two values are similar, threshold can be changed.
int ps5_similar(unsigned char* data, int3 a, int3 b){
  unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
  unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

  int i = abs(va-vb) < 1;
  return i;
}


__kernel void region(__global unsigned char *data, __global int* finished, __global unsigned char* region) {
  int id = get_global_id(0);
  int x = id%DATA_DIM;
  int y = id/DATA_DIM;
  int z = id/(DATA_DIM*DATA_DIM);

  int dx[6] = {-1,1,0,0,0,0};
  int dy[6] = {0,0,-1,1,0,0};
  int dz[6] = {0,0,0,0,-1,1};

  int3 pixel;
  pixel.x=x;
  pixel.y=y;
  pixel.z=z;
  for(int i=0;i<40;i++) {
    if(region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x]==2){
      region[pixel.z * DATA_DIM*DATA_DIM + pixel.y*DATA_DIM + pixel.x] = 1;
      for(int n = 0; n < 6; n++){
        int3 candidate = pixel;
        candidate.x += dx[n];
        candidate.y += dy[n];
        candidate.z += dz[n];

        if(!ps5_inside(candidate)){
          continue;
        }

        if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
          continue;
        }

        if(ps5_similar(*data, pixel, candidate)){
          region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 2;
          finished[0]=0;
        }
      }
    }
  }
}
