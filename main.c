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
#include <assert.h>

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
 * Author:
 *      Anonym genius
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
    else
    {
        ChainedAllocated *successor = malloc(sizeof(ChainedAllocated));
        successor->previous = allocatedLs;
        allocatedLs = successor;
    }
    void *ptr = malloc(size);

    allocatedLs->ptr = ptr;

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

/** Returns the index of the i/j/k th element **/
size_t idx(Parameters *params, size_t i, size_t j, size_t k)
{
    return i + j * params->maxi + k * params->maxi * params->maxj;
}

/** Sets the initial field as asked in Question 3.a. **/
void set_initial_conditions(double *E_y, Parameters *p)
{
    for (size_t i = 0; i < p->maxi; ++i)
    {
        for (size_t j = 0; j < p->maxj; ++j)
        {
            for (size_t k = 0; k < p->maxk; ++k)
            {
                assert(i + j * p->maxi + k * p->maxi * p->maxj <= p->maxi * p->maxj * p->maxk);
                E_y[idx(p, i, j, k)] = sin(PI * j * p->spatial_step / p->width) * sin(PI * i * p->spatial_step / p->length);
            }
        }
    }
}

void update_H_x_field(Parameters *p, double *H_x, double *E_y, double *E_z)
{
    size_t i, j, k;
    double factor = p->time_step / (MU * p->spatial_step);
    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                double E_y_nextk = 0.0;
                double E_z_nextj = 0.0;

                if (k + 1 < p->maxk)
                    E_y_nextk = E_y[idx(p, i, j, k + 1)] - E_y[idx(p, i, j, k)];
                if (j + 1 < p->maxj)
                    E_z_nextj = E_z[idx(p, i, j + 1, k)]  - E_z[idx(p, i, j, k)];

                H_x[idx(p, i, j, k)] = H_x[idx(p, i, j, k)] + factor * E_y_nextk - factor * E_z_nextj;
            }
}

void update_H_y_field(Parameters *p, double *H_y, double *E_z, double *E_x)
{
    size_t i, j, k;
    double factor = p->time_step / (MU * p->spatial_step);
    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                double E_x_nextk = 0.0;
                double E_z_nexti = 0.0;

                if (k + 1 < p->maxk)
                    E_x_nextk = E_x[idx(p, i, j, k + 1)] - E_x[idx(p, i, j, k)];
                if (i + 1 < p->maxi)
                    E_z_nexti = E_z[idx(p, i + 1, j, k)] - E_z[idx(p, i, j, k)];

                H_y[idx(p, i, j, k)] = H_y[idx(p, i, j, k)] + factor * E_z_nexti - factor * E_x_nextk;
            }
}

void update_H_z_field(Parameters *p, double *H_z, double *E_x, double *E_y)
{

    size_t i, j, k;
    double factor = p->time_step / (MU * p->spatial_step);
    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                double E_x_nextj = 0.0;
                double E_y_nexti = 0.0;

                if (i + 1 < p->maxi)
                    E_y_nexti = E_y[idx(p, i + 1, j, k)] - E_y[idx(p, i, j, k)];
                if (j + 1 < p->maxj)
                    E_x_nextj = E_x[idx(p, i, j + 1, k)] - E_x[idx(p, i, j, k)];

                H_z[idx(p, i, j, k)] = H_z[idx(p, i, j, k)] + factor * E_x_nextj - factor * E_y_nexti;
            }
}

void update_E_x_field(Parameters *p, double *E_x, double *H_z, double *H_y)
{
    size_t i, j, k;
    double factor = p->time_step / (EPSILON * p->spatial_step);
    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                double H_z_prevj = 0.0;
                double H_y_prevk = 0.0;

                if (0 < j)
                    H_z_prevj = H_z[idx(p, i, j, k)] - H_z[idx(p, i, j - 1, k)];
                if (0 < k)
                    H_y_prevk = H_y[idx(p, i, j, k)] - H_y[idx(p, i, j, k - 1)];

                E_x[idx(p, i, j, k)] = E_x[idx(p, i, j, k)] + factor *  H_z_prevj - factor * H_y_prevk;
            }
}

void update_E_y_field(Parameters *p, double *E_y, double *H_x, double *H_z)
{
    size_t i, j, k;
    double factor = p->time_step / (EPSILON * p->spatial_step);
    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                double H_x_prevk = 0.0;
                double H_z_previ = 0.0;

                if (0 < i)
                    H_z_previ = H_z[idx(p, i, j, k)] - H_z[idx(p, i - 1, j, k)];
                if (0 < k)
                    H_x_prevk = H_x[idx(p, i, j, k)] -  H_x[idx(p, i, j, k - 1)];

                E_y[idx(p, i, j, k)] = E_y[idx(p, i, j, k)] + factor * H_x_prevk - factor *  H_z_previ;
            }
}

void update_E_z_field(Parameters *p, double *E_z, double *H_y, double *H_x)
{

    size_t i, j, k;
    double factor = p->time_step / (EPSILON * p->spatial_step);
    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                double H_y_previ = 0.0;
                double H_x_prevj = 0.0;

                if (0 < i)
                    H_y_previ = H_y[idx(p, i, j, k)] - H_y[idx(p, i - 1, j, k)];
                if (0 < j)
                    H_x_prevj = H_x[idx(p, i, j, k)] - H_x[idx(p, i, j - 1, k)];

                E_z[idx(p, i, j, k)] = E_z[idx(p, i, j, k)] + factor * H_y_previ - factor * H_x_prevj;
            }
}

/** Draw the oven mesh with all the girds **/
void draw_oven(Parameters *p, DBfile *db)
{
    //TODO: Maybe create a Free to be able to free these once written?
    double *x = Malloc(p->maxi * sizeof(double));
    double *y = Malloc(p->maxj * sizeof(double));
    double *z = Malloc(p->maxk * sizeof(double));

    //TODO: Optimization: iterate once to the bigger and affect if in bounds of array...
    double dx = p->spatial_step;
    for (int i = 0; i < p->maxi; ++i)
    {
        x[i] = i * dx;
    }

    for (int i = 0; i < p->maxj; ++i)
    {
        y[i] = i * dx;
    }

    for (int i = 0; i < p->maxk; ++i)
    {
        z[i] = i * dx;
    }

    int dims[] = {p->maxi, p->maxj, p->maxk};
    int ndims = 3;
    double *cords[] = {x, y, z};
    DBPutQuadmesh(db, DB_MESHNAME, NULL, cords, dims, ndims, DB_DOUBLE, DB_COLLINEAR, NULL);
}

void write_silo(Fields *pFields, Parameters *pParams, int iteration, int *dims, int ndims)
{

    char filename[100];
    sprintf(filename, "r/output%04d.silo", iteration);

    DBfile *dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL, "My first SILO test", DB_PDB);
    if (!dbfile)
    {
        fail("Could not create DB\n");
    }

    draw_oven(pParams, dbfile);

    DBPutQuadvar1(dbfile, "ex", DB_MESHNAME, pFields->E_x, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "ey", DB_MESHNAME, pFields->E_y, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "ez", DB_MESHNAME, pFields->E_z, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "hx", DB_MESHNAME, pFields->H_x, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "hy", DB_MESHNAME, pFields->H_y, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "hz", DB_MESHNAME, pFields->H_z, dims, ndims, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);

    DBClose(dbfile);
}

double calculate_electrical_energy(Fields *pFields, Parameters *params)
{

    int i, j, k;
    double elec_energy = 0;
    for (i = 1; i < params->maxi - 1; i++)
        for (j = 1; j < params->maxj - 1; j++)
            for (k = 1; k < params->maxk - 1; k++)
            {
                elec_energy += pow(pFields->E_x[i + j * params->maxi + k * params->maxi * params->maxj], 2) +
                               pow(pFields->E_y[i + j * params->maxi + k * params->maxi * params->maxj], 2) +
                               pow(pFields->E_z[i + j * params->maxi + k * params->maxi * params->maxj], 2);
            }

    elec_energy *= EPSILON / 2.;

    return elec_energy;
}

double calculate_magnetic_energy(Fields *pFields, Parameters *params)
{

    int i, j, k;
    double mag_energy = 0;
    for (i = 1; i < params->maxi - 1; i++)
        for (j = 1; j < params->maxj - 1; j++)
            for (k = 1; k < params->maxk - 1; k++)
            {
                mag_energy += pow(pFields->H_x[i + j * params->maxi + k * params->maxi * params->maxj], 2) +
                              pow(pFields->H_y[i + j * params->maxi + k * params->maxi * params->maxj], 2) +
                              pow(pFields->H_z[i + j * params->maxi + k * params->maxi * params->maxj], 2);
            }

    mag_energy *= MU / 2.;

    return mag_energy;
}
void propagate_fields(Fields *pFields, Parameters *pParams)
{
    int dims[] = {pParams->maxi, pParams->maxj, pParams->maxk};
    int ndims = 3;

    double time_counter;
    int iteration = 0;
    double total_energy = calculate_electrical_energy(pFields, pParams) + calculate_magnetic_energy(pFields, pParams);
    write_silo(pFields, pParams, iteration, dims, ndims);
    iteration++;
    for (time_counter = 0; time_counter <= pParams->simulation_time; time_counter += pParams->time_step, iteration++)
    {
        printf("time: %0.10f s\n", time_counter);
        //below should be parallelized.
        update_H_x_field(pParams, pFields->H_x, pFields->E_y, pFields->E_z); //H_x
        update_H_y_field(pParams, pFields->H_y, pFields->E_z, pFields->E_x); //H_y
        update_H_z_field(pParams, pFields->H_z, pFields->E_x, pFields->E_y); //H_z

        update_E_x_field(pParams, pFields->E_x, pFields->H_z, pFields->H_y); //E_x
        update_E_y_field(pParams, pFields->E_y, pFields->H_x, pFields->H_z); //E_y
        update_E_z_field(pParams, pFields->E_z, pFields->H_y, pFields->H_x); //should check math // E_z

        //printf("Electrical energy: %0.10f \n", calculate_electrical_energy(pFields, pParams));
        //printf("Magnetic energy: %0.10f \n", calculate_magnetic_energy(pFields, pParams));
        //printf("Tot energy: %0.10f \n", calculate_electrical_energy(pFields, pParams) + calculate_magnetic_energy(pFields, pParams));
        assert((calculate_electrical_energy(pFields, pParams) + calculate_magnetic_energy(pFields, pParams) - total_energy) <= 0.000001);
        write_silo(pFields, pParams, iteration, dims, ndims);
    }
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

    printf("Initializing fields\n");
    Fields *pFields = initialize_fields(pParameters);

    printf("Creating mesh\n");

    printf("Setting initial conditions\n");
    set_initial_conditions(pFields->E_y, pParameters);
    printf("Launching simulation\n");
    propagate_fields(pFields, pParameters);

    printf("Freeing memory...\n");
    freeAll();

    printf("Simulation complete!\n");
    return 0;
}
