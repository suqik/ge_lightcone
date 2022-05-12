#include<stdio.h>
#include<stdlib.h>

typedef struct
{
  int npart;
  double mass;
  double BoxSize;
  double distance;
  double redshift;
}header_struct;

void write_to_gadget_slice(int slice_idx, header_struct* input, double (* particles)[2])
{
	int i, dummy;
	float temp=0.0;
	char fname[200];
	FILE *fd;
	
	sprintf(fname, "./slices/slice.%d", slice_idx);
    if ( !(fd = fopen(fname,"w+") ) )
	{
        printf("Cannot open the file `%s'\n", fname);
        exit(0);
    }
	
	dummy = fwrite(&input->npart, sizeof(int), 1, fd);
	dummy = fwrite(&input->mass, sizeof(double), 1, fd);
	dummy = fwrite(&input->BoxSize, sizeof(double), 1, fd);
	dummy = fwrite(&input->distance, sizeof(double), 1, fd);
	dummy = fwrite(&input->redshift, sizeof(double), 1, fd);
	
	for(i=0; i<input->npart; i++)
	{
		temp = (float)particles[i][0];
		dummy = fwrite(&temp, sizeof(float), 1, fd);
		temp = (float)particles[i][1];
		dummy = fwrite(&temp, sizeof(float), 1, fd);
	}

	fclose(fd);
}

