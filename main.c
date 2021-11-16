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
#include <silo.h>
#include <string.h>

#define DB_FILENAME "result.silo"
#define DB_MESHNAME "mesh"

#define MU 1.25663706143591729538505735331180115367886775975E-6
#define EPSILON 8.854E-12
#define PI 3.14159265358979323846264338327950288419716939937510582097494

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
 *    maxi:             Number of grid subdivisions (x dimension)
 *    maxj:             Number of grid subdivisions (y dimension)
 *    maxk:             Number of grid subdivisions (z dimension)
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

/** A structure that rassembles all the fields components
 * Properties:
 *      E_x/y/z     The arrays of the x/y/z components of the electric field
 *      H_x/y/z     The arrays of the x/y/z componnents of the magnetic field
 **/
typedef struct fields
{

    double *E_x;
    double *E_y;
    double *E_z;
    double *H_x;
    double *H_y;
    double *H_z;

} Fields;

/** Exceptionnally clever way to garbage collect
 * Parameters:
 *      previous:   The previous allocated object
 *      ptr:        The current allocated object
 * Description:
 *      This structure allow us to store each allocated object
 *      into a chained list with a LiFo strategy so when we
 *      free up memory we take care of the inner most objects
**/
typedef struct chainedAllocated
{
    struct chainedAllocated *previous;
    void *ptr;
} ChainedAllocated;

/**
------------------------------------------------------
---------------------------- Function's Specifications
------------------------- Signatures & Implementations
------------------------------------------------------
**/

// Superglobal variable
static ChainedAllocated *allocatedLs;

/** Free any allocated object with the next Malloc function
 * Description:
 *  This function free's the memory allocated for all
 *  the objects allocated with the next Malloc function
 *  in the reverse way of their allocation (LiFo) in order
 *  to deal with nested structures.
**/
void *freeAll()
{
    while (allocatedLs)
    {
        free(allocatedLs->ptr);
        ChainedAllocated *previous = allocatedLs->previous;
        free(allocatedLs);
        allocatedLs = previous;
    }
}

/** Free the memory and throws an error, then exits.**/
void fail(const char *msg)
{
    perror(msg);
    freeAll();
    exit(EXIT_FAILURE);
}

/** Critical allocation shortcut (malloc or Critical error with exit)
 * Description:
 *  In addition to critically check if the malloc properly worked,
 *  this function stores the reference to the new allocated object
 *  in the superglobal chained list which is then used to free the
 *  memory before exit.
**/
void *Malloc(size_t size)
{
    if (allocatedLs == NULL)
    {
        allocatedLs = malloc(sizeof(ChainedAllocated));
    }
    void *ptr = malloc(size);

    allocatedLs->ptr = ptr;
    ChainedAllocated *successor = malloc(sizeof(ChainedAllocated));
    allocatedLs->previous = allocatedLs;
    allocatedLs = successor;

    if (!ptr)
    {
        fail("CRITICAL ERROR: Could not allocate enough memory!");
    }
    return ptr;
}

/** Loads the parameters into the system memory
 * Arguments:
 *    filename: The file containing the parameters properties (.txt)
 * Return value:
 *    A pointer to the parameters structure loaded in system memory
**/
Parameters *load_parameters(const char *filename)
{
    FILE *fParams = fopen(filename, "r");
    Parameters *pParameters = Malloc(sizeof(Parameters));

    if (!fParams)
    {
        fail("Unable to open parameters file!");
    }

    fscanf(fParams, "%f", &pParameters->width);
    fscanf(fParams, "%f", &pParameters->height);
    fscanf(fParams, "%f", &pParameters->length);
    fscanf(fParams, "%lf", &pParameters->spatial_step);
    fscanf(fParams, "%lf", &pParameters->time_step);
    fscanf(fParams, "%f", &pParameters->simulation_time);
    fscanf(fParams, "%u", &pParameters->sampling_rate);

    fclose(fParams);

    pParameters->maxi = (uint)(pParameters->length / pParameters->spatial_step + 1);
    pParameters->maxj = (uint)(pParameters->width / pParameters->spatial_step + 1);
    pParameters->maxk = (uint)(pParameters->height / pParameters->spatial_step + 1);

    return pParameters;
}

/** Allocates and initialize to 0.0 all the components of each field at a given time t.
 * Comment:
 *  (Amaury) It might be possible to store in system memory only the data for a single time step.
 *  (Ionut) Pay attention that I/O operations can be bottleneck... ?
**/
Fields *initialize_fields(Parameters *params)
{
    Fields *pFields = Malloc(sizeof(Fields));
    uint space_size = params->maxi * params->maxj * params->maxk;

    pFields->E_x = Malloc(sizeof(double) * space_size);
    pFields->E_y = Malloc(sizeof(double) * space_size);
    pFields->E_z = Malloc(sizeof(double) * space_size);
    pFields->H_x = Malloc(sizeof(double) * space_size);
    pFields->H_y = Malloc(sizeof(double) * space_size);
    pFields->H_z = Malloc(sizeof(double) * space_size);

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

/** Sets the initial field as asked in Question 3.a. **/
void set_initial_conditions(double *E_y, Parameters *params)
{
    for (uint i = 0; i < params->maxi; ++i)
    {
        for (uint j = 0; j < params->maxj ^ 2; j += params->maxj)
        {
            for (uint k = 0; k < params->maxk; ++i)
            {
                E_y[i + j + k * params->maxk * params->maxj] = sin(1 * PI * i * params->spatial_step / params->width) * sin(1 * PI * k * params->spatial_step / params->length);
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

void propagate_fields(Fields *pFields, Parameters *pParams, DBfile *db)
{
    int dims[] = {pParams->length, pParams->width, pParams->height};
    int ndims = 3;

    DBPutQuadvar1(db, "E_x", DB_MESHNAME, pFields->E_x, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(db, "E_y", DB_MESHNAME, pFields->E_y, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(db, "E_z", DB_MESHNAME, pFields->E_z, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(db, "H_x", DB_MESHNAME, pFields->H_x, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(db, "H_y", DB_MESHNAME, pFields->H_y, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(db, "H_z", DB_MESHNAME, pFields->H_z, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);

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

/** Draw the oven mesh with all the girds **/
void draw_oven(Parameters *params, DBfile *db)
{
    //TODO: Maybe create a Free to be able to free these once written?
    float *x = Malloc(params->maxi * sizeof(float));
    float *y = Malloc(params->maxj * sizeof(float));
    float *z = Malloc(params->maxk * sizeof(float));

    //TODO: Optimization: iterate once to the bigger and affect if in bounds of array...
    for (int i = 0; i < params->maxi; ++i)
    {
        x[i] = i;
    }

    for (int i = 0; i < params->maxj; ++i)
    {
        y[i] = i;
    }

    for (int i = 0; i < params->maxk; ++i)
    {
        z[i] = i;
    }

    int dims[] = {params->length, params->width, params->height};
    int ndims = 3;
    float *cords[] = {x, y, z};
    DBPutQuadmesh(db, "oven", NULL, cords, dims, ndims, DB_DOUBLE, DB_COLLINEAR, NULL);
}

/**
------------------------------------------------------
---------------------------------------- Main function
------------------------------------------------------
**/

int main(int argc, const char *argv[])
{
    printf("Welcome into our microwave oven eletrico-magnetic field simulator! \n");

    if (argc != 2)
    {
        perror("This program needs 1 argument: the parameters file (.txt). Eg.: ./microwave param.txt");
        return EXIT_FAILURE;
    }

    printf("Loading the parameters...\n");
    Parameters *pParameters = load_parameters(argv[1]);

    //BEGIN DEBUG
    printf("width: %f \n", pParameters->width);
    printf("height: %f \n", pParameters->height);
    printf("length: %f \n", pParameters->length);
    printf("spatial: %lf \n", pParameters->spatial_step);
    printf("time: %lf \n", pParameters->time_step);
    printf("total: %f \n", pParameters->simulation_time);
    printf("rate: %u \n", pParameters->sampling_rate);
    //END DEBUG

    if (pParameters->time_step > pParameters->simulation_time)
    {
        perror("The time step must be lower than the simulation time!");
        return EXIT_FAILURE;
    }

    Fields *pFields = initialize_fields(pParameters);

    // Open a SILO db
    DBfile *db = DBCreate(DB_FILENAME, DB_CLOBBER, DB_LOCAL, "My first SILO test", DB_PDB);
    if (!db)
    {
        fail("Could not create DB\n");
    }

    draw_oven(pParameters, db);
    set_initial_conditions(pFields->E_y, pParameters);
    propagate_fields(pFields, pParameters, db);

    DBClose(db);
    printf("Freeing memory...\n");
    freeAll();

    printf("Simulation complete!\n");
    return 0;
}
