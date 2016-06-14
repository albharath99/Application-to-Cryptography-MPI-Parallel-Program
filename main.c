#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char s[1000000000]; // character array to hold the input message
double t1, t2; // time before and after

int main(int argc , char * argv[])
{
	int rank;
	int p;
	int r1, c1, r2, c2, *mat1, *mat2, *row, *ans, *mat3, *ex_row, *ex_ans, port;

	MPI_Status status;
	MPI_Init( &argc , &argv ); // initialize
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank of current process
	MPI_Comm_size(MPI_COMM_WORLD, &p); // number of processes
	if(rank == 0){
		r1 = 3; // dimensions of the encoding matrix...it can be of any other dimensions or be dynamic(entered by user)
		c1 = 3;

		puts("Enter message to encode:");
		fgets(s, 100, stdin) ; // takes 100 characters can be modified to receive more
		t1 = MPI_Wtime(); // time before processing
		int sz = strlen(s);
		while(sz % 3 != 0) {
			s[sz++] = ' ';
			/* Since we are using a 3 by 3 encoding matrix, we make the message size suitable to be in a (3*X) matrix
			to be multiplied by the encoding matrix... so we complete the remaining elements by spaces */
		}
		s[sz] = '\0';

		//portion of rows to send to other processes and setting the encryption matrix
		port = r1 / p;
		// encoding matrix
		mat1 = (int*)malloc(r1*c1*sizeof(int));
		memset(mat1, 0, r1*c1*sizeof(int));
		mat1[0] = 1;
		mat1[1] = 2;
		mat1[2] = -3;
		mat1[3] = 5;
		mat1[4] = 28;
		mat1[5] = 0;
		mat1[6] = -1;
		mat1[7] = -1;
		mat1[8] = 0;

        // set elements manually. It can be modified to be set by a random number generator

		//setting message matrix
		r2 = r1;
		c2 = sz/r1;
		mat2 = (int*)malloc(sz*sizeof(int));
		memset(mat2, 0, r2*c2*sizeof(int));
		int i;
		for(i=0;i < r2;i++) {
			int j;
			for(j=0;j < c2;j++) {
				mat2[i*c2+j] = s[j*r2+i];
			}
		}

/*
    Parallelization scenario:

        Since the encoding matrix can be of any size, it is scattered to the processes so we can parallelize its multiplication
    by the message matrix. The message matrix is broadcasted to all processes and the result of the multiplication which is
    divided on the processes is gathered in the master matrix to be printed. For the remainder rows that have not been scattered
    send and receive are used.
*/

		mat3 = (int*)malloc(r1*c2*sizeof(int));
		row  = (int*)malloc(c1*port*sizeof(int));
		ans = (int*)malloc(c2*sizeof(int));
		memset(mat3, 0, r1*c2*sizeof(int));
		memset(row, 0, c1*port*sizeof(int));
		memset(ans, 0, c2*sizeof(int));
		i = 1;
		for(;i < p;i++) {
			MPI_Send(&r1, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&c1, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&r2, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&c2, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}

	//Initilizing arrays in each process
	if(rank != 0) {
		MPI_Recv(&r1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&c1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&r2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&c2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		port = r1 / p;
		mat2 = (int*)malloc(r2*c2*sizeof(int));
		row  = (int*)malloc(c1*port*sizeof(int));
		ans = (int*)malloc(c2*port*sizeof(int));
		mat3 = (int*)malloc(r1*c2*sizeof(int));
	}

	if(rank <= r1%p && r1%p!=0) {
		ex_row = (int*)malloc(c1*sizeof(int));
		ex_ans = (int*)malloc(c2*sizeof(int));
	}

	MPI_Bcast(mat2, r2*c2, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(mat1, c1*port, MPI_INT, row, c1*port, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0 && r1%p != 0) {
		int rem = r1 % p;
		int i = 1;
		for(;i <= rem;i++) {
			MPI_Send(mat1 + (port*p+i-1)*c1, c1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}

	memset(ans, 0, c2*port*sizeof(int));
	int k = 0;
	for(;k < port;k++) {
		int j = 0;
		for(;j < c2;j++) {
			int i = 0;
			for(;i < r2;i++) {
				ans[k*c2+j] += row[k*r2+i]*mat2[j + i*c2];
			}
		}
	}

	if(rank <= r1%p && r1%p!=0 && rank != 0) {
		MPI_Recv(ex_row, c1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		memset(ex_ans, 0, c2*sizeof(int));
		int k = 0;
		int j = 0;
		for(;j < c2;j++) {
			int i = 0;
			for(;i < r2;i++) {
				ex_ans[j] += ex_row[i]*mat2[j + i*c2];
			}
		}
		MPI_Send(ex_ans, c2, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Gather(ans, c2*port, MPI_INT, mat3, c2*port, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0 && r1%p!=0) {
		int rem = r1 % p;
		int i = 1;
		for(;i <= rem;i++) {
			MPI_Recv(mat3 + (port*p+i-1)*c2, c2, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
		}
	}

	t2 = MPI_Wtime(); // time after processing

	if(rank == 0) {
		puts("Encoded message sequence:");
		int i = 0;
		for(;i < r1;i++) {
			int j = 0;
			for(;j < c2;j++) {
				printf("%d ", mat3[i*c2+j]);
			}
			puts("");
		}
		printf("Total time = %.10f\n", t2 - t1);
	}

	MPI_Finalize(); // finalize
	return 0;
}
