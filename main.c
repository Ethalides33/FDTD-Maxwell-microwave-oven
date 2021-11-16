/** 
 * Title: Electromagnetic wave propagation in a microwave
 * Authors: Amaury Baret, Ionut Finta
 * Date: December 2021
 * Organization: University of Liège, Belgium
 * Course: INFO0939 - High performance scientific computing
 * Professors: Geuzaine Christophe; Hiard Samuel, Leduc Guy
 * Description:
 *    This program simulates the propagation of an electromagnetic
      wave in a microwave oven using the FDTD scheme.
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif

#define MU 1.25663706143591729538505735331180115367886775975E-6
#define EPSILON 8.854E-12

/**
------------------------------------------------------
------------------- Data Structures & Type definitions
------------------------------------------------------
**/

// Shortcuts:
typedef unsigned int uint;
typedef unsigned char uchar;

/** Definition of the scene
 * Properties:
 *    width:            a in figure (y, in paper cs)
 *    height:           b in figure (z, in paper cs)
 *    length:           d in figure (x, in paper cs)
 *    spatial_step:     delta x = delta y = delta z
 *    time_step:        delta t
 *    simulation_time:  interval of time simulated (e.g. 1s)
 *    sampling_rate:    rate at which data is printed to file 
**/
typedef struct parameters
{
    float width;
    float height;
    float length;
    uint maxi;
    uint maxj;
    uint maxk;
    double spatial_step;
    double time_step;
    float simulation_time;
    uint sampling_rate;

} Parameters;

typedef struct fields
{

    double *E_x;
    double *E_y;
    double *E_z;
    double *H_x;
    double *H_y;
    double *H_z;

} Fields;

/** Exceptionnally clever way to garbage collect **/
typedef struct chainedAllocated
{
    struct chainedAllocated *previous;
    void *ptr;
} ChainedAllocated;

static ChainedAllocated *allocatedLs;

/**
------------------------------------------------------
---------------------------- Function's Specifications
------------------------- Signatures & Implementations
------------------------------------------------------
**/

/** Free any allocated array **/
void *free_malloced()
{
    while (allocatedLs)
    {
        free(allocatedLs->ptr);
        ChainedAllocated *previous = allocatedLs->previous;
        free(allocatedLs);
        allocatedLs = previous;
    }
}

/** Critical allocation shortcut (malloc or Critical error with exit) **/
void *malloc_and_check(size_t __size)
{
    if (allocatedLs == NULL)
    {
        allocatedLs = malloc(sizeof(ChainedAllocated));
    }
    void *ptr = malloc(__size);

    allocatedLs->ptr = ptr;
    ChainedAllocated *successor = malloc(sizeof(ChainedAllocated));
    allocatedLs->previous = allocatedLs;
    allocatedLs = successor;

    if (!ptr)
    {
        perror("CRITICAL ERROR: Could not allocate enough memory!");
        free_malloced();
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/** Loads the parameters into the system memory
 * Arguments:
 *    filename: The file containing the parameters properties (.txt)
 * Return value:
 *    A pointer to the parameters structure loaded in system memory
**/
Parameters *load_parameters(char *filename)
{
    FILE *pParameters_file = fopen(filename, "r");
    Parameters *pParameters = malloc_and_check(sizeof(Parameters));

    if (!pParameters_file)
    {
        perror("Unable to open parameters file!");
        exit(EXIT_FAILURE);
    }

    fscanf(pParameters_file, "%f", &pParameters->width);
    fscanf(pParameters_file, "%f", &pParameters->height);
    fscanf(pParameters_file, "%f", &pParameters->length);
    fscanf(pParameters_file, "%lf", &pParameters->spatial_step);
    fscanf(pParameters_file, "%lf", &pParameters->time_step);
    fscanf(pParameters_file, "%f", &pParameters->simulation_time);
    fscanf(pParameters_file, "%u", &pParameters->sampling_rate);

    fclose(pParameters_file);

    pParameters->maxi = (uint)(pParameters->length / pParameters->spatial_step + 1);
    pParameters->maxj = (uint)(pParameters->width / pParameters->spatial_step + 1);
    pParameters->maxk = (uint)(pParameters->height / pParameters->spatial_step + 1);

    return pParameters;
}

//It might be possible to only store data for one time step.
Fields *initialize_fields(Parameters *params)
{
    Fields *pFields = malloc_and_check(sizeof(Fields));
    uint space_size = params->maxi * params->maxj * params->maxk;

    pFields->E_x = malloc_and_check(sizeof(double) * space_size);
    pFields->E_y = malloc_and_check(sizeof(double) * space_size);
    pFields->E_z = malloc_and_check(sizeof(double) * space_size);
    pFields->H_x = malloc_and_check(sizeof(double) * space_size);
    pFields->H_y = malloc_and_check(sizeof(double) * space_size);
    pFields->H_z = malloc_and_check(sizeof(double) * space_size);

    for (uint i = 0; i < space_size; ++i)
    {
        pFields->E_x[i] = 0.0;
        pFields->E_y[i] = 0.0;
        pFields->E_z[i] = 0.0;
        pFields->H_x[i] = 0.0;
        pFields->H_y[i] = 0.0;
        pFields->H_z[i] = 0.0;
    }

    return pFields;
}

/** Sets the initial fields **/
void set_initial_conditions(double *E_y, Parameters *params)
{
    for (uint i = 0; i < params->maxi; ++i)
    {
        for (uint j = 0; j < params->maxj; ++j)
        {
            for (uint k = 0; k < params->maxk; ++i)
            {
                E_y[i + j * params->maxj + k * params->maxk * params->maxj] = sin(1 * M_PI * i * params->spatial_step / params->width) * sin(1 * M_PI * k * params->spatial_step / params->length);
            }
        }
    }
}

void update_H_x_field(Parameters *params, double *H_x, double *E_y, double *E_z)
{
    int i, j, k;
    double factor = params->time_step / (MU * params->spatial_step);
    for (i = 0; i < params->maxi; i++)
        for (j = 0; j < params->maxj; j++)
            for (k = 0; k < params->maxk; k++)
            {
                if (i == 0 || i == params->maxi - 1)
                {
                    H_x[i + j * params->maxj + k * params->maxj * params->maxk] = 0;
                }
                else
                {
                    H_x[i + j * params->maxj + k * params->maxj * params->maxk] = H_x[i + j * params->maxj + k * params->maxj * params->maxk] + factor * (E_y[i + j * params->maxj + (k + 1) * params->maxj * params->maxk] - E_y[i + j * params->maxj + k * params->maxj * params->maxk]) - factor * (E_z[i + (j + 1) * params->maxj + k * params->maxj * params->maxk] - E_z[i + j * params->maxj + k * params->maxj * params->maxk]);
                }
            }
}

void update_H_y_field(Parameters *params, double *H_y, double *E_z, double *E_x)
{

    int i, j, k;
    double factor = params->time_step / (MU * params->spatial_step);
    for (i = 0; i < params->maxi; i++)
        for (j = 0; j < params->maxj; j++)
            for (k = 0; k < params->maxk; k++)
            {
                if (j == 0 || j == params->maxj - 1)
                {
                    H_y[i + j * params->maxj + k * params->maxj * params->maxk] = 0;
                }
                else
                {
                    H_y[i + j * params->maxj + k * params->maxj * params->maxk] = H_y[i + j * params->maxj + k * params->maxj * params->maxk] + factor * (E_z[i + 1 + j * params->maxj + k * params->maxj * params->maxk] - E_z[i + j * params->maxj + k * params->maxj * params->maxk]) - factor * (E_x[i + j * params->maxj + (k + 1) * params->maxj * params->maxk] - E_x[i + j * params->maxj + k * params->maxj * params->maxk]);
                }
            }
}

void update_H_z_field(Parameters *params, double *H_z, double *E_x, double *E_y)
{

    int i, j, k;
    double factor = params->time_step / (MU * params->spatial_step);
    for (i = 0; i < params->maxi; i++)
        for (j = 0; j < params->maxj; j++)
            for (k = 0; k < params->maxk; k++)
            {
                if (k == 0 || k == params->maxk - 1)
                {
                    H_z[i + j * params->maxj + k * params->maxj * params->maxk] = 0;
                }
                else
                {
                    H_z[i + j * params->maxj + k * params->maxj * params->maxk] = H_z[i + j * params->maxj + k * params->maxj * params->maxk] + factor * (E_x[i + (j + 1) * params->maxj + k * params->maxj * params->maxk] - E_x[i + j * params->maxj + k * params->maxj * params->maxk]) - factor * (E_y[i + 1 + j * params->maxj + k * params->maxj * params->maxk] - E_y[i + j * params->maxj + k * params->maxj * params->maxk]);
                }
            }
}

void update_E_x_field(Parameters *params, double *E_x, double *H_z, double *H_y)
{
    int i, j, k;
    double factor = params->time_step / (EPSILON * params->spatial_step);
    for (i = 0; i < params->maxi; i++)
        for (j = 0; j < params->maxj; j++)
            for (k = 0; k < params->maxk; k++)
            {
                if (j == 0 || j == params->maxj - 1 || k == 0 || k == params->maxk - 1)
                {
                    H_z[i + j * params->maxj + k * params->maxj * params->maxk] = 0;
                }
                else
                {
                    E_x[i + j * params->maxj + k * params->maxj * params->maxk] = E_x[i + j * params->maxj + k * params->maxj * params->maxk] + factor * (H_z[i + j * params->maxj + k * params->maxj * params->maxk] - H_z[i + (j - 1) * params->maxj + k * params->maxj * params->maxk]) - factor * (H_y[i + j * params->maxj + k * params->maxj * params->maxk] - H_y[i + j * params->maxj + (k - 1) * params->maxj * params->maxk]);
                }
            }
}

void update_E_y_field(Parameters *params, double *E_y, double *H_x, double *H_z)
{

    int i, j, k;
    double factor = params->time_step / (EPSILON * params->spatial_step);
    for (i = 0; i < params->maxi; i++)
        for (j = 0; j < params->maxj; j++)
            for (k = 0; k < params->maxk; k++)
            {
                if (i == 0 || i == params->maxi - 1 || k == 0 || k == params->maxk - 1)
                {
                    H_z[i + j * params->maxj + k * params->maxj * params->maxk] = 0;
                }
                else
                {
                    E_y[i + j * params->maxj + k * params->maxj * params->maxk] = E_y[i + j * params->maxj + k * params->maxj * params->maxk] + factor * (H_x[i + j * params->maxj + k * params->maxj * params->maxk] - H_x[i + j * params->maxj + (k - 1) * params->maxj * params->maxk]) - factor * (H_z[i + j * params->maxj + k * params->maxj * params->maxk] - H_z[i - 1 + j * params->maxj + k * params->maxj * params->maxk]);
                }
            }
}

void update_E_z_field(Parameters *params, double *E_z, double *H_y, double *H_x)
{

    int i, j, k;
    double factor = params->time_step / (EPSILON * params->spatial_step);
    for (i = 0; i < params->maxi; i++)
        for (j = 0; j < params->maxj; j++)
            for (k = 0; k < params->maxk; k++)
            {
                if (i == 0 || i == params->maxi - 1 || j == 0 || j == params->maxj - 1)
                {
                    E_z[i + j * params->maxj + k * params->maxj * params->maxk] = 0;
                }
                else
                {
                    E_z[i + j * params->maxj + k * params->maxj * params->maxk] = E_z[i + j * params->maxj + k * params->maxj * params->maxk] + factor * (H_y[i + j * params->maxj + k * params->maxj * params->maxk] - H_y[i - 1 + j * params->maxj + k * params->maxj * params->maxk]) - factor * (H_x[i + j * params->maxj + k * params->maxj * params->maxk] - H_x[i + (j - 1) * params->maxj + k * params->maxj * params->maxk]);
                }
            }
}

void propagate_fields(Fields *pFields, Parameters *pParams)
{

    float time_counter;
    for (time_counter = 0; time_counter <= pParams->simulation_time; time_counter += pParams->time_step)
    {
        //below should be parallelized.
        update_H_x_field(pParams, pFields->H_x, pFields->E_y, pFields->E_z); //H_x
        update_H_y_field(pParams, pFields->H_y, pFields->E_z, pFields->E_x); //H_y
        update_H_z_field(pParams, pFields->H_z, pFields->E_x, pFields->E_y); //H_z

        update_E_x_field(pParams, pFields->E_x, pFields->H_z, pFields->H_y); //E_x
        update_E_y_field(pParams, pFields->E_y, pFields->H_x, pFields->H_z); //E_y
        update_E_z_field(pParams, pFields->E_z, pFields->H_y, pFields->H_x); //should check math // E_z
    }
}

/**
------------------------------------------------------
---------------------------------------- Main function
------------------------------------------------------
**/

int main(int argc, char **argv)
{
    printf("Welcome into our microwave oven simulation engine! \n");

    if (argc != 2)
    {
        perror("This program needs 1 argument: the parameters file (.txt). Eg.: ./microwave param.txt");
        return EXIT_FAILURE;
    }

    printf("Loading the parameters...\n");
    Parameters *pParameters = load_parameters(argv[1]);
    printf("width: %f \n", pParameters->width);
    printf("height: %f \n", pParameters->height);
    printf("length: %f \n", pParameters->length);
    printf("spatial: %lf \n", pParameters->spatial_step);
    printf("time: %lf \n", pParameters->time_step);
    printf("total: %f \n", pParameters->simulation_time);
    printf("rate: %u \n", pParameters->sampling_rate);
    if (pParameters->time_step > pParameters->simulation_time)
    {
        perror("The time step must be lower than the simulation time!");
        return EXIT_FAILURE;
    }

    Fields *pFields = initialize_fields(pParameters);
    set_initial_conditions(pFields, pParameters);
    propagate_fields(pFields, pParameters);

    printf("Freeing memory...\n");
    free_malloced();

    printf("Simulation complete!\n");
    return 0;
}
