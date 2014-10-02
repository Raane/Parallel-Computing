all : histogram_serial histogram_omp histogram_pthreads

histogram_serial : histogram_serial.c
	gcc -std=c99 -Wall histogram_serial.c bmp.c -o histogram_serial
	./histogram_serial out.bmp 8
	diff out.bmp fasit.bmp
    
histogram_omp : histogram_omp.c
	gcc -std=c99 -Wall histogram_omp.c bmp.c -o histogram_omp -fopenmp
	./histogram_omp out.bmp 8
	diff out.bmp fasit.bmp
    
histogram_pthreads : histogram_pthreads.c
	gcc -std=c99 -Wall histogram_pthreads.c bmp.c -o histogram_pthreads -pthread
	./histogram_pthreads out.bmp 8
	diff out.bmp fasit.bmp
	
clean :
	rm histogram_omp histogram_serial histogram_pthreads
