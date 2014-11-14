// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 64
// float3 utilities
float3 ps5_cross(float3 a, float3 b){
  float3 c;
  c.x = a.y*b.z - a.z*b.y;
  c.y = a.z*b.x - a.x*b.z;
  c.z = a.x*b.y - a.y*b.x;

  return c;
}

float3 ps5_normalize(float3 v){
  float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  v.x /= l;
  v.y /= l;
  v.z /= l;

  return v;
}

float3 ps5_add(float3 a, float3 b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;

  return a;
}

float3 ps5_scale(float3 a, float b){
  a.x *= b;
  a.y *= b;
  a.z *= b;

  return a;
}

// Indexing function (note the argument order)
int ps5_index(int z, int y, int x){
  return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Checks if position is inside the volume (float3 and int3 versions)
int ps5_inside(float3 pos){
  int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
  int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
  int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

  return x && y && z;
}

// Trilinear interpolation
float ps5_value_at(float3 pos, unsigned char* data){
  if(!ps5_inside(pos)){
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

  float a0 = rx*data[ps5_index(z,y,x)] + (1-rx)*data[ps5_index(z,y,x_u)];
  float a1 = rx*data[ps5_index(z,y_u,x)] + (1-rx)*data[ps5_index(z,y_u,x_u)];
  float a2 = rx*data[ps5_index(z_u,y,x)] + (1-rx)*data[ps5_index(z_u,y,x_u)];
  float a3 = rx*data[ps5_index(z_u,y_u,x)] + (1-rx)*data[ps5_index(z_u,y_u,x_u)];

  float b0 = ry*a0 + (1-ry)*a1;
  float b1 = ry*a2 + (1-ry)*a3;

  float c0 = rz*b0 + (1-rz)*b1;


  return c0;
}



__kernel void raycast(__global unsigned char* data, __global unsigned char* region, __global unsigned char* image) {
  int id = get_global_id(0);

  // Camera/eye position, and direction of viewing. These can be changed to look
  // at the volume from different angles.
  float3 camera;
  camera.x=1000;
  camera.y=1000;
  camera.z=1000;
  float3 forward;
  forward.x=-1;
  forward.y=-1;
  forward.z=-1;
  float3 z_axis;
  z_axis.x=0;
  z_axis.y=0;
  z_axis.z=1;

  // Finding vectors aligned with the axis of the image
  float3 right = ps5_cross(forward, z_axis);
  float3 up = ps5_cross(right, forward);

  // Creating unity lenght vectors
  forward = ps5_normalize(forward);
  right = ps5_normalize(right);
  up = ps5_normalize(up);

  float fov = 3.14/4;
  float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
  float step_size = 0.5;

  int x = -IMAGE_DIM/2 + id%IMAGE_DIM;
  int y = -IMAGE_DIM/2 + id/IMAGE_DIM;

  // Find the ray for this pixel
  float3 screen_center = ps5_add(camera, forward);
  float3 ray = ps5_add(ps5_add(screen_center, ps5_scale(right, x*pixel_width)), ps5_scale(up, y*pixel_width));
  ray = ps5_add(ray, ps5_scale(camera, -1));
  ray = ps5_normalize(ray);
  float3 pos = camera;

  // Move along the ray, we stop if the color becomes completely white,
  // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
  // to be big enough to let rays pass through the entire volume)
  int i = 0;
  float color = 0;
  while(color < 255 && i < 5000){
    i++;
    pos = ps5_add(pos, ps5_scale(ray, step_size));          // Update position
    int r = ps5_value_at(pos, *region);                  // Check if we're in the region
    color += ps5_value_at(pos, *data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
  }
  // Write final color to image
  image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
  //image[id] = 255;
}
