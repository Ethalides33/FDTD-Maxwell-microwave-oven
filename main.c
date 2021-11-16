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
 *    width:            a in figure
 *    height:           b in figure
 *    length:           c in figure
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
    ChainedAllocated *previous;
    void *ptr;
} ChainedAllocated;

static ChainedAllocated *allocatedLs;

/**
------------------------------------------------------
---------------------------- Function's Specifications
------------------------- Signatures & Implementations
------------------------------------------------------
**/

/** Critical allocation shortcut (malloc or Critical error with exit) **/
void *malloc_and_check_critical(size_t __size)
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

/** Loads the parameters into the system memory
 * Arguments:
 *    filename: The file containing the parameters properties (.txt)
 * Return value:
 *    A pointer to the parameters structure loaded in system memory
**/
Parameters *load_parameters(char *filename)
{
    FILE *pParameters_file = fopen(filename, "r");
    Parameters *pParameters = malloc_and_check_critical(sizeof(Parameters));

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

    return pParameters;
}

//It might be possible to only store data for one time step.
Fields *initialize_fields(Parameters *params)
{

    Fields *pFields = malloc_and_check_critical(sizeof(Fields));
    uint space_size = space_steps_x * space_steps_y * space_steps_z;

    pFields->E_x = malloc_and_check_critical(sizeof(double) * space_size);
    memset(&pFields->E_x, 0, sizeof(double) * space_size); // NOT SURE ABOUT THE &.
    pFields->E_y = malloc_and_check_critical(sizeof(double) * space_size);
    memset(&pFields->E_y, 0, sizeof(double) * space_size);
    pFields->E_z = malloc_and_check_critical(sizeof(double) * space_size);
    memset(&pFields->E_z, 0, sizeof(double) * space_size);

    pFields->H_x = malloc_and_check_critical(sizeof(double) * space_size);
    memset(&pFields->H_x, 0, sizeof(double) * space_size);
    pFields->H_y = malloc_and_check_critical(sizeof(double) * space_size);
    memset(&pFields->H_y, 0, sizeof(double) * space_size);
    pFields->H_z = malloc_and_check_critical(sizeof(double) * space_size);
    memset(&pFields->H_z, 0, sizeof(double) * space_size);

    return pFields;
}

void set_initial_conditions(Fields *pFields)
{

    print("hi");
}

void update_H_x_field()
{

    int i, j, k;
    double factor = time_step / (mu * space_step);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            for (k = 0; k < kmax; k++)
            {
                if (i == 0 || i == imax - 1)
                {
                    H_x[i + j * jmax + k * jmax * kmax] = 0;
                }
                else
                {
                    H_x[i + j * jmax + k * jmax * kmax] = H_x[i + j * jmax + k * jmax * kmax] + factor * (E_y[i + j * jmax + (k + 1) * jmax * kmax] - E_y[i + j * jmax + k * jmax * kmax]) - factor * (E_z[i + (j + 1) * jmax + k * jmax * kmax] - E_z[i + j * jmax + k * jmax * kmax]);
                }
            }
}

void update_H_y_field()
{

    int i, j, k;
    double factor = time_step / (mu * space_step);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            for (k = 0; k < kmax; k++)
            {
                if (j == 0 || j == jmax - 1)
                {
                    H_y[i + j * jmax + k * jmax * kmax] = 0;
                }
                else
                {
                    H_y[i + j * jmax + k * jmax * kmax] = H_y[i + j * jmax + k * jmax * kmax] + factor * (E_z[i + 1 + j * jmax + k * jmax * kmax] - E_z[i + j * jmax + k * jmax * kmax]) - factor * (E_x[i + j * jmax + (k + 1) * jmax * kmax] - E_x[i + j * jmax + k * jmax * kmax]);
                }
            }
}

void update_H_z_field()
{

    int i, j, k;
    double factor = time_step / (mu * space_step);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            for (k = 0; k < kmax; k++)
            {
                if (k == 0 || k == kmax - 1)
                {
                    H_z[i + j * jmax + k * jmax * kmax] = 0;
                }
                else
                {
                    H_z[i + j * jmax + k * jmax * kmax] = H_z[i + j * jmax + k * jmax * kmax] + factor * (E_x[i + (j + 1) * jmax + k * jmax * kmax] - E_x[i + j * jmax + k * jmax * kmax]) - factor * (E_y[i + 1 + j * jmax + k * jmax * kmax] - E_y[i + j * jmax + k * jmax * kmax]);
                }
            }
}

void update_E_x_field()
{

    int i, j, k;
    double factor = time_step / (epsilon * space_step);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            for (k = 0; k < kmax; k++)
            {
                if (j == 0 || j == jmax - 1 || k == 0 || k == kmax - 1)
                {
                    H_z[i + j * jmax + k * jmax * kmax] = 0;
                }
                else
                {
                    E_x[i + j * jmax + k * jmax * kmax] = E_x[i + j * jmax + k * jmax * kmax] + factor * (H_z[i + j * jmax + k * jmax * kmax] - H_z[i + (j - 1) * jmax + k * jmax * kmax]) - factor * (H_y[i + j * jmax + k * jmax * kmax] - H_y[i + j * jmax + (k - 1) * jmax * kmax]);
                }
            }
}

void update_E_y_field()
{

    int i, j, k;
    double factor = time_step / (epsilon * space_step);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            for (k = 0; k < kmax; k++)
            {
                if (i == 0 || i == imax - 1 || k == 0 || k == kmax - 1)
                {
                    H_z[i + j * jmax + k * jmax * kmax] = 0;
                }
                else
                {
                    E_y[i + j * jmax + k * jmax * kmax] = E_y[i + j * jmax + k * jmax * kmax] + factor * (H_x[i + j * jmax + k * jmax * kmax] - H_x[i + j * jmax + (k - 1) * jmax * kmax]) - factor * (H_z[i + j * jmax + k * jmax * kmax] - H_z[i - 1 + j * jmax + k * jmax * kmax]);
                }
            }
}

void update_E_z_field()
{

    int i, j, k;
    double factor = time_step / (epsilon * space_step);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            for (k = 0; k < kmax; k++)
            {
                if (i == 0 || i == imax - 1 || j == 0 || j == jmax - 1)
                {
                    H_z[i + j * jmax + k * jmax * kmax] = 0;
                }
                else
                {
                    E_z[i + j * jmax + k * jmax * kmax] = E_z[i + j * jmax + k * jmax * kmax] + factor * (H_y[i + j * jmax + k * jmax * kmax] - H_y[i - 1 + j * jmax + k * jmax * kmax]) - factor * (H_x[i + j * jmax + k * jmax * kmax] - H_x[i + (j - 1) * jmax + k * jmax * kmax]);
                }
            }
}

void propagate_fields(Fields *pFields, Parameters *pParams)
{

    float time_counter;
    for (time_counter = 0; time_counter <= pParams->simulation_time; time_counter += pParams->time_step)
    {
        //below should be parallelized.
        update_H_field(pFields->H_x, pFields->E_y, pFields->E_z, pParams); //H_x
        update_H_field(pFields->H_y, pFields->E_z, pFields->E_x, pParams); //H_y
        update_H_field(pFields->H_z, pFields->E_x, pFields->E_y, pParams); //H_z

        update_E_field(pFields->E_x, pFields->H_z, pFields->H_y, pParams); //E_x
        update_E_field(pFields->E_y, pFields->H_x, pFields->H_z, pParams); //E_y
        update_E_field(pFields->E_z, pFields->H_y, pFields->H_x, pParams); //should check math // E_z
    }
}

/** Free parameters from memory
 * Arguments:
 *  parameters: The pointer to the Parameters object
**/
void free_parameters(Parameters *parameters)
{
    if (parameters)
    {
        free(parameters);
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
    if (time_step > simulation_time)
    {
        perror("The time step must be lower than the simulation time!");
        return EXIT_FAILURE;
    }

    Fields *pFields = initialize_fields(pParameters);
    set_initial_conditions(pFields);
    propagate_fields(pFields, pParameters);

    printf("Freeing memory...\n");
    free_malloced();

    printf("Simulation complete!\n");
    return 0;
}
