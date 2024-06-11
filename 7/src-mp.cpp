#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// files to read and write:

#define BIGSIGNALFILEBIN	"bigsignal.bin"
#define BIGSIGNALFILEASCII	"bigsignal.txt"
#define CSVPLOTFILE		"plot.csv"

// how many elements are in the big signal:

#define NUMELEMENTS	(8*1024*1024)

// only do this many shifts, not all NUMELEMENTS of them (this is enough to uncover the secret sine wave):

#define MAXSHIFTS	1024

// how many autocorrelation sums to plot:

#define MAXPLOT		 400

// pick which file type to read, BINARY or ASCII (BINARY is much faster to read):
// (pick one, not both)

#define BINARY
//#define ASCII

// print debugging messages?

#define DEBUG		1

// globals:

float *BigSums;		// the overall MAXSHIFTS autocorrelation array
float *BigSignal;	// the overall NUMELEMENTS-big signal data

// function prototype:

void DoOneLocalAutocorrelation(int, int, float*, float*);

int main(int argc, char *argv[])
{
    int NumThreads;

    // Initialize OpenMP and get the number of threads
    #pragma omp parallel
    {
        NumThreads = omp_get_num_threads();
    }

    int PPSize = NUMELEMENTS / NumThreads;

    // Allocate space for the big signal and sums
    BigSignal = (float*)malloc((NUMELEMENTS + MAXSHIFTS) * sizeof(float));
    BigSums = (float*)malloc(MAXSHIFTS * sizeof(float));

    // Allocate 2D arrays for per-thread signal and sums
    float **PPSignal = (float**)malloc(NumThreads * sizeof(float*));
    float **PPSums = (float**)malloc(NumThreads * sizeof(float*));
    for (int i = 0; i < NumThreads; i++) {
        PPSignal[i] = (float*)malloc((PPSize + MAXSHIFTS) * sizeof(float));
        PPSums[i] = (float*)malloc(MAXSHIFTS * sizeof(float));
    }

    // Read the BigSignal array
    #ifdef ASCII
    FILE *fp = fopen(BIGSIGNALFILEASCII, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Cannot open data file '%s'\n", BIGSIGNALFILEASCII);
        return -1;
    }

    for (int i = 0; i < NUMELEMENTS; i++)
    {
        float f;
        fscanf(fp, "%f", &f);
        BigSignal[i] = f;
    }
    fclose(fp);
    #endif

    #ifdef BINARY
    FILE *fp = fopen(BIGSIGNALFILEBIN, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Cannot open data file '%s'\n", BIGSIGNALFILEBIN);
        return -1;
    }

    fread(BigSignal, sizeof(float), NUMELEMENTS, fp);
    fclose(fp);
    #endif

    // Duplicate part of the array
    for (int i = 0; i < MAXSHIFTS; i++)
    {
        BigSignal[NUMELEMENTS + i] = BigSignal[i];
    }

    // Start the timer
    double time0 = omp_get_wtime();

    #pragma omp parallel
    {
        int me = omp_get_thread_num();
        
        // Distribute data to each thread
        for (int i = 0; i < PPSize + MAXSHIFTS; i++)
        {
            PPSignal[me][i] = BigSignal[me * PPSize + i];
        }

        // Each thread does its own autocorrelation
        DoOneLocalAutocorrelation(me, PPSize, PPSignal[me], PPSums[me]);

        // Each thread adds its sums to the overall sums
        #pragma omp critical
        {
            for (int s = 0; s < MAXSHIFTS; s++)
            {
                BigSums[s] += PPSums[me][s];
            }
        }
    }

    // Stop the timer
    double time1 = omp_get_wtime();

    // Print the performance
    double seconds = time1 - time0;
    double performance = (double)MAXSHIFTS * (double)NUMELEMENTS / seconds / 1000000.;  // mega-elements computed per second
    fprintf(stderr, "%3d threads, %10d elements, %9.2lf mega-autocorrelations computed per second\n",
            NumThreads, NUMELEMENTS, performance);

    // Write the file to be plotted to look for the secret sine wave
    FILE *fp_plot = fopen(CSVPLOTFILE, "w");
    if (fp_plot == NULL)
    {
        fprintf(stderr, "Cannot write to plot file '%s'\n", CSVPLOTFILE);
    }
    else
    {
        for (int s = 1; s < MAXPLOT; s++)  // BigSums[0] is huge -- don't use it
        {
            fprintf(fp_plot, "%6d , %10.2f\n", s, BigSums[s]);
        }
        fclose(fp_plot);
    }

    // All done
    free(BigSums);
    free(BigSignal);
    for (int i = 0; i < NumThreads; i++) {
        free(PPSignal[i]);
        free(PPSums[i]);
    }
    free(PPSignal);
    free(PPSums);
    
    return 0;
}

// Read from the per-thread signal array, write into the local sums array
void DoOneLocalAutocorrelation(int me, int PPSize, float *localSignal, float *localSums)
{
    if (DEBUG) fprintf(stderr, "Thread %3d entered DoOneLocalAutocorrelation()\n", me);

    for (int s = 0; s < MAXSHIFTS; s++)
    {
        float sum = 0.;
        for (int i = 0; i < PPSize; i++)
        {
            sum += localSignal[i] * localSignal[i + s];
        }
        localSums[s] = sum;
    }
}
