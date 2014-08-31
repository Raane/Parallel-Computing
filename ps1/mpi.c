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
    int message=0;
    MPI_Send(&message, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    printf("Rank %d sent %d\n",rank, message);
    MPI_Recv(&in, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
    printf("Rank %d recieved %d\n",rank, in);
  }
  else if(rank == size-1) {
    MPI_Recv(&in, 1, MPI_INT, size-2, 1, MPI_COMM_WORLD, &status);
    printf("Rank %d recieved %d\n",rank, in);
    int message=rank;
    MPI_Send(&message, 1, MPI_INT, size-2, 1, MPI_COMM_WORLD);
    printf("Rank %d sent %d\n",rank, message);
  } 
  else{
    MPI_Recv(&in, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
    printf("Rank %d recieved %d\n",rank, in);
    int message=rank;
    MPI_Send(&message, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
    printf("Rank %d sent %d\n",rank, message);
    MPI_Recv(&in, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &status);
    printf("Rank %d recieved %d\n",rank, in);
    message=rank;
    MPI_Send(&message, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD);
    printf("Rank %d sent %d\n",rank, message);
  }
  MPI_Finalize();
}
