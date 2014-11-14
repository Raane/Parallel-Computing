__kernel void region(__global unsigned char *data, __global int* finished, __global unsigned char* region) {
	int id = get_global_id(0);
	region[id] = data[id];
}
