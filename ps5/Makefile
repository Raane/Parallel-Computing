run: raycast
	./raycast
	
raycast : bmp.cpp raycast.cu
	nvcc bmp.cpp raycast.cu -arch compute_20 -lcudart -lm -o raycast
