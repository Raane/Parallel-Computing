#include <stdio.h>
#include <stdlib.h>

typedef struct{
  //Add needed fields
} matrix_t;


matrix_t* new_matrix(int rows, int cols){
}


void print_matrix(matrix_t* matrix){
}


void set_value(matrix_t* matrix, int row, int col, float value){
}


float get_value(matrix_t* matrix, int row, int col){
}


int is_sparse(matrix_t matrix, float sparse_threshold){
}
      

int matrix_multiply(matrix_t* a, matrix_t* b, matrix_t** c){
}


void change_size(matrix_t* matrix, int new_rows, int new_cols){
}


void free_matrix(matrix_t* matrix){
}
        

int main(int argc, char** argv){
  
  // Create and fill matrix m
  matrix_t* m = new_matrix(3,4);
  for(int row = 0; row < 3; row++){
    for(int col = 0; col < 4; col++){
      set_value(m, row, col, row*10+col);
    }
  }
  
  // Create and fill matrix n
  matrix_t* n = new_matrix(4,4);
  for(int row = 0; row < 4; row++){
    for(int col = 0; col < 4; col++){
      set_value(n, row, col, col*10+row);
    }
  }
  
  // Create and fill matrix o
  matrix_t* o = new_matrix(5,5);
  for(int row = 0; row < 5; row++){
    for(int col = 0; col < 5; col++){
      set_value(o, row, col, row==col? 1 : 0);
    }
  }
  // Printing matrices
  printf("Matrix m:\n");
  print_matrix(m);
  /*
  Should print:
  0.00 1.00 2.00 3.00 
  10.00 11.00 12.00 13.00 
  20.00 21.00 22.00 23.00
  */
  
  printf("Matrix n:\n");
  print_matrix(n);
  /*
  Should print:
  0.00 10.00 20.00 30.00 
  1.00 11.00 21.00 31.00 
  2.00 12.00 22.00 32.00 
  3.00 13.00 23.00 33.00 
  */
  
  
  printf("Matrix o:\n");
  print_matrix(o);
  /*
  Should print:
  1.00 0.00 0.00 0.00 0.00 
  0.00 1.00 0.00 0.00 0.00 
  0.00 0.00 1.00 0.00 0.00 
  0.00 0.00 0.00 1.00 0.00 
  0.00 0.00 0.00 0.00 1.00
  */
  
  // Checking if matrices are sparse (more than 75% 0s)
  printf("Matrix m is sparse: %d\n", is_sparse(*m, 0.75)); // Not sparse, should print 0
  printf("Matrix o is sparse: %d\n", is_sparse(*o, 0.75)); // Sparse, should print 1
  
  
  // Attempting to multiply m and o, should not work
  matrix_t* p;
  int error = matrix_multiply(m,o,&p);
  printf("Error (m*o): %d\n", error); // Should print -1 
 
  // Attempting to multiply m and n, should work
  error = matrix_multiply(m,n,&p);
  print_matrix(p);
  /*
  Should print:
  14.00 74.00 134.00 194.00 
  74.00 534.00 994.00 1454.00 
  134.00 994.00 1854.00 2714.00 
  */
  
  // Shrinking m, expanding n
  change_size(m, 2,2);
  change_size(n, 5,5);
  
  printf("Matrix m:\n");
  print_matrix(m);
  /*
  Should print:
  0.00 1.00 
  10.00 11.00 
  */
  printf("Matrix n:\n");
  print_matrix(n);
  /*
  Should print:
  0.00 10.00 20.00 30.00 0.00 
  1.00 11.00 21.00 31.00 0.00 
  2.00 12.00 22.00 32.00 0.00 
  3.00 13.00 23.00 33.00 0.00 
  0.00 0.00 0.00 0.00 0.00
  */
  
  // Freeing memory
  free_matrix(m);
  free_matrix(n);
  free_matrix(o);
  free_matrix(p);
}
