#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// setting the number of threads:
#ifndef NUMT
#define NUMT		    4
#endif

#define CSV

// Constants
const float GRAIN_GROWS_PER_MONTH = 12.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;
const float ONE_WOLF_EATS_PER_MONTH = 1.0;

const float AVG_PRECIP_PER_MONTH = 7.0; // average
const float AMP_PRECIP_PER_MONTH = 6.0; // plus or minus
const float RANDOM_PRECIP = 2.0;       // plus or minus noise

const float AVG_TEMP = 60.0; // average
const float AMP_TEMP = 20.0; // plus or minus
const float RANDOM_TEMP = 10.0; // plus or minus noise

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

const int   ENDYEAR = 2030;

// Global variables
int	NowYear = 2024;		// 2024- 2029
int	NowMonth = 0;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int     NowNumDeer;		// number of deer in the current population
int     NowNumWolves;   // current wolf population

// Global lock and barrier variables
omp_lock_t	    Lock;
volatile int	NumInThreadTeam;
volatile int	NumAtBarrier;
volatile int	NumGone;

// specify how many threads will be in the barrier:
//	(also init's the Lock)
void InitBarrier(int n) {
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}

// have the calling thread wait here until all the other threads catch up:
void WaitBarrier() {
    omp_set_lock(&Lock);
    {
        NumAtBarrier++;
        if(NumAtBarrier == NumInThreadTeam)
        {
            NumGone = 0;
            NumAtBarrier = 0;
            // let all other threads get back to what they were doing
            // before this one unlocks, knowing that they might immediately
            // call WaitBarrier() again:
            while(NumGone != NumInThreadTeam-1);
            omp_unset_lock(&Lock);
            return;
        }
    }
    omp_unset_lock(&Lock);

    while(NumAtBarrier != 0);	// this waits for the nth thread to arrive

#pragma omp atomic
    NumGone++;			// this flags how many threads have returned
}

float Ranf(float low, float high) {
    float r = (float)rand(); // 0 - RAND_MAX
    float t = r / (float)RAND_MAX; // 0. - 1.

    return low + t * (high - low);
}

// call this if you want to force your program to use
// a different random number sequence every time you run it:
void TimeOfDaySeed() {
    struct tm y2k = {0};
    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;

    time_t timer;
    time(&timer);
    double seconds = difftime(timer, mktime(&y2k));
    unsigned int seed = (unsigned int)(1000. * seconds); // milliseconds
    srand(seed);
}

float SQR(float x) {
        return x*x;
}

// Function to convert inches to centimeters
float inchesToCm(float inches) {
    return inches * 2.54;
}

// Function to convert Fahrenheit to Celsius
float fToC(float fahrenheit) {
    return (5. / 9.) * (fahrenheit - 32);
}

void computeGlobalState() {
    float ang = (30.*(float)NowMonth + 15.) * (M_PI / 180.);	// angle of earth around the sun
    float temp = AVG_TEMP - AMP_TEMP * cos(ang);
    NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
    NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
    if (NowPrecip < 0.) {
        NowPrecip = 0.;
    }
}

void Deer() {
    // Computing phase
    // compute a temporary next-value for deer based on the current state of the simulation
    int nextNumDeer = NowNumDeer;
    int carryingCapacity = (int)(NowHeight);
    if (nextNumDeer < carryingCapacity)
        nextNumDeer+= 3;
    else if( nextNumDeer > carryingCapacity )
        nextNumDeer--;
    nextNumDeer -= (float)NowNumWolves * ONE_WOLF_EATS_PER_MONTH;
    if (nextNumDeer < 0) {
        nextNumDeer = 0;
    }
    WaitBarrier();

    // Assigning phase
    NowNumDeer = nextNumDeer;
    WaitBarrier();

    // Printing phase
    WaitBarrier();
}


void Wolves()
{
    // Computing phase
    // compute a temporary next-value for wolves based on the current state of the simulation
    // Wolves only show up from November to March
    int nextNumWolves = NowNumWolves;
    if (NowMonth <= 2 || NowMonth >= 10) {
        int carryingCapacity = (int)(NowHeight);
        if (nextNumWolves == 0 || nextNumWolves < carryingCapacity)
            nextNumWolves++;
        else if( nextNumWolves > carryingCapacity )
            nextNumWolves--;
    }
    else {
        nextNumWolves= 0;
    }
    if (nextNumWolves < 0) {
        nextNumWolves = 0;
    }
    WaitBarrier();

    // Assigning phase
    NowNumWolves = nextNumWolves;
    WaitBarrier();

    // Printing phase
    WaitBarrier();
}

void Grain() {
    // Computing phase
    // compute a temporary next-value for grain based on the current state of the simulation
    float tempFactor = exp(-SQR((NowTemp - MIDTEMP)/10.));
    float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP)/10.));
    float nextHeight = NowHeight;
    nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
    nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
    if (nextHeight < 0.) {
        nextHeight = 0.;
    } 
    WaitBarrier();

    // Assigning phase
    NowHeight = nextHeight;
    WaitBarrier();

    // Printing phase
    WaitBarrier();
}

void Watcher() {
    // Computing phase
    WaitBarrier();

    // Assigning phase
    WaitBarrier();
    
    // Print the current set of global state variables
#ifdef CSV
    fprintf(stderr, "%d , %d , %6.2lf , %6.2lf , %6.2lf , %2d , %2d\n", NowYear, NowMonth + 1, fToC(NowTemp), inchesToCm(NowPrecip), inchesToCm(NowHeight), NowNumDeer, NowNumWolves);
    // fprintf(stderr, "%6.2lf , %6.2lf , %6.2lf , %2d\n", fToC(NowTemp), inchesToCm(NowPrecip), inchesToCm(NowHeight), NowNumDeer);
#else
    fprintf(stderr, "%6.2lf oF, %6.2lf inches, %6.2lf inches, %2d deers\n", NowTemp, NowPrecip, NowHeight, NowNumDeer);
#endif

    // Printing phase
    // Increment the month count
    NowMonth++;
    if (NowMonth > 11) {
        NowYear++;
        NowMonth = 0;
    }

    // compute the new Temperature and Precipitation
    computeGlobalState();
    
    WaitBarrier();
}

int main(int argc, char *argv[]) {
#ifdef _OPENMP
    #ifndef CSV
        fprintf(stderr, "OpenMP is supported -- version = %d\n", _OPENMP);
    #endif
#else
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

#ifdef CSV
    fprintf(stderr, "Year , Month , Temperature , Precipitation , Grain Height , Deer , Wolf\n");
#endif

    // starting date and time:
    NowMonth = 0;
    NowYear = 2024;

    // starting state (feel free to change this if you want):
    NowNumDeer = 2;
    NowHeight = 5.;
    NowNumWolves = 0;

    omp_set_num_threads(NUMT); // same as # of sections
    InitBarrier(NUMT);

    TimeOfDaySeed();
    computeGlobalState();

    while (NowYear < ENDYEAR)
#pragma omp parallel sections
        {
#pragma omp section
            {
                Deer();
            }

#pragma omp section
            {
                Grain();
            }

#pragma omp section
            {
                Watcher();
            }

#pragma omp section
            {
                Wolves();
            }
        } // implied barrier -- all functions must return in order
    // to allow any of them to get past here

    return 0;
}
