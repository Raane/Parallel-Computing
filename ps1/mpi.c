#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv){
  int size, rank, in;
  MPI_Status status;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0){
    for(int i = 1; i < size; i++){
      MPI_Send(&i, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
    }
  }
  else{
    MPI_Recv(&in, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    printf("Rank %d recieved %d\n",rank, in);
  }
  MPI_Finalize();
}
