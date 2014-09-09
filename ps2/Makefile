all : region

run : region
	mpirun -n 4 region pic1.bmp

region : region.c
	mpicc -std=c99 region.c bmp.c -o region -lm
	

