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

#define DB_FILENAME "r/result%04d.silo"
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

typedef enum MODE
{
    VALIDATION_MODE = 0,
    COMPUTATION_MODE = 1
} MODE;

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
 *    simulation_time:  interval of time simulated (in seconds)
 *    sampling_rate:    rate at which data is printed to file (in #steps)
 *    mode:             0 for validation mode, 1 for computation       
**/
typedef struct parameters
{
    float width;
    float height;
    float length;
    size_t maxi;
    size_t maxj;
    size_t maxk;
    double spatial_step;
    double time_step;
    float simulation_time;
    uint sampling_rate;
    MODE mode;

} Parameters;

/** Definition of the oven mesh (can be derived directly from Parameters)
 * Properties:
 *  dims:    Array of the sizes of each dimension (maxi, maxj, maxk)
 *  coords:  Cordinates of each mesh point (grid)
**/
typedef struct oven
{
    int *dims;
    double **coords;
} Oven;

/** A structure that rassembles all the fields components
 * Properties:
 *      Ex/y/z     The arrays of the x/y/z components of the electric field
 *      Hx/y/z     The arrays of the x/y/z componnents of the magnetic field
 **/
typedef struct fields
{

    double *Ex;
    double *Ey;
    double *Ez;
    double *Hx;
    double *Hy;
    double *Hz;

} Fields;

/** Exceptionnally clever way to garbage collect in C
 * Properties:
 *      previous:   The previous allocated object
 *      ptr:        The current allocated object
 * Description:
 *      This structure allow us to store each allocated object
 *      into a chained list with a LiFo strategy so when we
 *      free up all memory we take care of the inner most objects
 * Author:
 *      Anonymous genius
**/
typedef struct chainedAllocated
{
    struct chainedAllocated *previous;
    void *ptr;
} ChainedAllocated;

// Superglobal variable
static ChainedAllocated *allocatedLs;

/**
------------------------------------------------------
---------------------------- Function's Specifications
------------------------- Signatures & Implementations
------------------------------------------------------
**/

/** Free any allocated object with the Malloc function (see below)
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

/** Free the memory and throws an error, then exits with EXIT_FAILURE.
 * Parameters:
 *  msg:    The message to throw before exit
**/
void fail(const char *msg)
{
    perror(msg);
    freeAll();
    exit(EXIT_FAILURE);
}

/** Critical allocation (malloc or Critical error with exit)
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

/** Nice free which takes care of our superglobal chained list
 * Parameters
 *  ptr:    The pointer to free
**/
void Free(void *ptr)
{
    ChainedAllocated *successor = allocatedLs;
    ChainedAllocated *current = allocatedLs;
    while (current->ptr != ptr)
    {
        successor = current;
        current = current->previous;
    }

    free(current->ptr);
    successor->previous = current->previous;
    free(current);
}

/** Loads the parameters into the system memory
 * Parameters:
 *    filename: The file containing the parameters properties (.txt)
 * Return:
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
    fscanf(fParams, "%x", &pParameters->mode);

    fclose(fParams);

    pParameters->maxi = (size_t)(pParameters->length / pParameters->spatial_step + 1);
    pParameters->maxj = (size_t)(pParameters->width / pParameters->spatial_step + 1);
    pParameters->maxk = (size_t)(pParameters->height / pParameters->spatial_step + 1);

    return pParameters;
}

/** Compute the oven properties and returns it
 * Parameters:
 *  params: The parameters of the simulation
 * Returns:
 *  A pointer to the oven structure
**/
Oven *compute_oven(Parameters *params)
{
    Oven *r = Malloc(sizeof(Oven));
    r->dims = Malloc(sizeof(size_t) * 3);
    r->coords = Malloc(sizeof(double **) * 3);
    r->dims[0] = params->maxi;
    r->dims[1] = params->maxj;
    r->dims[2] = params->maxk;

    double *x = Malloc(params->maxi * sizeof(double));
    double *y = Malloc(params->maxj * sizeof(double));
    double *z = Malloc(params->maxk * sizeof(double));

    //TODO: Optimization: iterate once to the bigger and affect if in bounds of array...
    double dx = params->spatial_step;
    for (int i = 0; i < params->maxi; ++i)
    {
        x[i] = i * dx;
    }

    for (int i = 0; i < params->maxj; ++i)
    {
        y[i] = i * dx;
    }

    for (int i = 0; i < params->maxk; ++i)
    {
        z[i] = i * dx;
    }

    int dims[] = {params->maxi, params->maxj, params->maxk};
    int ndims = 3;
    double *cords[] = {x, y, z};

    r->coords[0] = x;
    r->coords[1] = y;
    r->coords[2] = z;

    return r;
}

/** Allocates and initialize to 0.0 all the components of each field at a given time t.
 * Parameters:
 *  params: The parameters of the simulation
**/
Fields *initialize_fields(Parameters *params)
{
    Fields *pFields = Malloc(sizeof(Fields));
    size_t space_size = params->maxi * params->maxj * params->maxk;

    pFields->Ex = Malloc(sizeof(double) * space_size);
    pFields->Ey = Malloc(sizeof(double) * space_size);
    pFields->Ez = Malloc(sizeof(double) * space_size);
    pFields->Hx = Malloc(sizeof(double) * space_size);
    pFields->Hy = Malloc(sizeof(double) * space_size);
    pFields->Hz = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Ex[space_size] = 0.0;
        pFields->Ey[space_size] = 0.0;
        pFields->Ez[space_size] = 0.0;
        pFields->Hx[space_size] = 0.0;
        pFields->Hy[space_size] = 0.0;
        pFields->Hz[space_size] = 0.0;
    }

    return pFields;
}

/** Fast shortcut to get the index of a field at i, j and k
 * Parameters:
 *  params:  The params of the simulation
 *  i, j, k: The coordinates of the wanted field
 * Returns:
 *  The index in a 1D array
**/
size_t idx(Parameters *params, size_t i, size_t j, size_t k)
{
    return i + j * params->maxi + k * params->maxi * params->maxj;
}

/** Sets the initial field as asked in Question 3.a.
 * Parameters:
 *  Ey: The y component of the Energy fields
 *  p:   The parameters of the simulation
 * Remark: 
 * (Ionut) - Should only be done for VALIDATION_MODE ?
**/
void set_initial_conditions(double *Ey, Parameters *p)
{
    size_t i, j, k;
    for (i = 0; i < p->maxi; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 0; k < p->maxk; ++k)
            {
                assert(i + j * p->maxi + k * p->maxi * p->maxj <= p->maxi * p->maxj * p->maxk);
                Ey[idx(p, i, j, k)] = sin(PI * j * p->spatial_step / p->width) * sin(PI * i * p->spatial_step / p->length);
            }
}

/** Updates the H field
 * Parameters:
 *  p:      The parameters of the simulation
 *  fields: The fields
**/
void update_H_field(Parameters *p, Fields *fields)
{
    // Shortcuts to avoid pointers exploration in the loop.
    double *Ex = fields->Ex;
    double *Ey = fields->Ey;
    double *Ez = fields->Ez;
    double *Hx = fields->Hx;
    double *Hy = fields->Hy;
    double *Hz = fields->Hz;

    double factor = p->time_step / (MU * p->spatial_step),
           Ey_nexti = 0.0, Ez_nexti = 0.0,
           Ex_nextj = 0.0, Ez_nextj = 0.0,
           Ex_nextk = 0.0, Ey_nextk = 0.0;

    size_t i, j, k;

    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                Ey_nexti = 0.0;
                Ez_nexti = 0.0;
                Ex_nextj = 0.0;
                Ez_nextj = 0.0;
                Ey_nextk = 0.0;
                Ex_nextk = 0.0;

                if (i + 1 < p->maxi)
                {
                    Ez_nexti = Ez[idx(p, i + 1, j, k)] - Ez[idx(p, i, j, k)];
                    Ey_nexti = Ey[idx(p, i + 1, j, k)] - Ey[idx(p, i, j, k)];
                }
                if (k + 1 < p->maxk)
                {
                    Ey_nextk = Ey[idx(p, i, j, k + 1)] - Ey[idx(p, i, j, k)];
                    Ex_nextk = Ex[idx(p, i, j, k + 1)] - Ex[idx(p, i, j, k)];
                }
                if (j + 1 < p->maxj)
                {
                    Ez_nextj = Ez[idx(p, i, j + 1, k)] - Ez[idx(p, i, j, k)];
                    Ex_nextj = Ex[idx(p, i, j + 1, k)] - Ex[idx(p, i, j, k)];
                }

                Hx[idx(p, i, j, k)] = Hx[idx(p, i, j, k)] + factor * Ey_nextk - factor * Ez_nextj;
                Hy[idx(p, i, j, k)] = Hy[idx(p, i, j, k)] + factor * Ez_nexti - factor * Ex_nextk;
                Hz[idx(p, i, j, k)] = Hz[idx(p, i, j, k)] + factor * Ex_nextj - factor * Ey_nexti;
            }
}

/** Updates the E field
 * Parameters:
 *  p:      The parameters of the simulation
 *  fields: The fields
**/
void update_E_field(Parameters *p, Fields *fields)
{
    // Shortcuts to avoid pointers exploration in the loop.
    double *Ex = fields->Ex;
    double *Ey = fields->Ey;
    double *Ez = fields->Ez;
    double *Hx = fields->Hx;
    double *Hy = fields->Hy;
    double *Hz = fields->Hz;

    double factor = p->time_step / (EPSILON * p->spatial_step),
           Hy_previ = 0.0, Hz_previ = 0.0,
           Hx_prevj = 0.0, Hz_prevj = 0.0,
           Hx_prevk = 0.0, Hy_prevk = 0.0;

    size_t i, j, k;

    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
            {
                Hy_previ = 0.0;
                Hz_previ = 0.0;
                Hx_prevj = 0.0;
                Hz_prevj = 0.0;
                Hx_prevk = 0.0;
                Hy_prevk = 0.0;

                if (0 < i)
                {
                    Hz_previ = Hz[idx(p, i, j, k)] - Hz[idx(p, i - 1, j, k)];
                    Hy_previ = Hy[idx(p, i, j, k)] - Hy[idx(p, i - 1, j, k)];
                }
                if (0 < j)
                {
                    Hz_prevj = Hz[idx(p, i, j, k)] - Hz[idx(p, i, j - 1, k)];
                    Hx_prevj = Hx[idx(p, i, j, k)] - Hx[idx(p, i, j - 1, k)];
                }
                if (0 < k)
                {
                    Hy_prevk = Hy[idx(p, i, j, k)] - Hy[idx(p, i, j, k - 1)];
                    Hx_prevk = Hx[idx(p, i, j, k)] - Hx[idx(p, i, j, k - 1)];
                }

                Ex[idx(p, i, j, k)] = Ex[idx(p, i, j, k)] + factor * Hz_prevj - factor * Hy_prevk;
                Ey[idx(p, i, j, k)] = Ey[idx(p, i, j, k)] + factor * Hx_prevk - factor * Hz_previ;
                Ez[idx(p, i, j, k)] = Ez[idx(p, i, j, k)] + factor * Hy_previ - factor * Hx_prevj;
            }
}

void write_silo(Fields *pFields, Parameters *pParams, Oven *pOven, int iteration)
{
    char filename[100];
    sprintf(filename, DB_FILENAME, iteration);

    DBfile *dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL, "My first SILO test", DB_PDB);
    if (!dbfile)
    {
        fail("Could not create DB\n");
    }

    DBPutQuadmesh(dbfile, DB_MESHNAME, NULL, pOven->coords, pOven->dims, 3, DB_DOUBLE, DB_COLLINEAR, NULL);
    DBPutQuadvar1(dbfile, "ex", DB_MESHNAME, pFields->Ex, pOven->dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "ey", DB_MESHNAME, pFields->Ey, pOven->dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "ez", DB_MESHNAME, pFields->Ez, pOven->dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "hx", DB_MESHNAME, pFields->Hx, pOven->dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "hy", DB_MESHNAME, pFields->Hy, pOven->dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);
    DBPutQuadvar1(dbfile, "hz", DB_MESHNAME, pFields->Hz, pOven->dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT, NULL);

    DBClose(dbfile);
}

double calculate_electrical_energy(Fields *pFields, Parameters *p)
{
    double elec_energy = 0;
    size_t i, j, k;
    for (i = 1; i < p->maxi - 1; i++)
        for (j = 1; j < p->maxj - 1; j++)
            for (k = 1; k < p->maxk - 1; k++)
            {
                elec_energy += pow(pFields->Ex[idx(p, i, j, k)], 2) +
                               pow(pFields->Ey[idx(p, i, j, k)], 2) +
                               pow(pFields->Ez[idx(p, i, j, k)], 2);
            }

    elec_energy *= EPSILON / 2.;

    return elec_energy;
}

double calculate_magnetic_energy(Fields *pFields, Parameters *p)
{
    double mag_energy = 0;
    size_t i, j, k;
    for (i = 1; i < p->maxi - 1; i++)
        for (j = 1; j < p->maxj - 1; j++)
            for (k = 1; k < p->maxk - 1; k++)
            {
                mag_energy += pow(pFields->Hx[idx(p, i, j, k)], 2) +
                              pow(pFields->Hy[idx(p, i, j, k)], 2) +
                              pow(pFields->Hz[idx(p, i, j, k)], 2);
            }

    mag_energy *= MU / 2.;

    return mag_energy;
}
void propagate_fields(Fields *pFields, Parameters *pParams, Oven *pOven)
{
    double time_counter;
    int iteration = 1;
    double total_energy = calculate_electrical_energy(pFields, pParams) + calculate_magnetic_energy(pFields, pParams);
    write_silo(pFields, pParams, pOven, iteration);
    for (time_counter = 0; time_counter <= pParams->simulation_time; time_counter += pParams->time_step, iteration++)
    {
        //printf("time: %0.10f s\n", time_counter);
        //below should be parallelized.
        update_H_field(pParams, pFields);
        update_E_field(pParams, pFields);

        //printf("Electrical energy: %0.10f \n", calculate_electrical_energy(pFields, pParams));
        //printf("Magnetic energy: %0.10f \n", calculate_magnetic_energy(pFields, pParams));
        //printf("Tot energy: %0.10f \n", calculate_electrical_energy(pFields, pParams) + calculate_magnetic_energy(pFields, pParams));
        assert((calculate_electrical_energy(pFields, pParams) + calculate_magnetic_energy(pFields, pParams) - total_energy) <= 0.000001);
        if (iteration % pParams->sampling_rate == 0)
        {
            write_silo(pFields, pParams, pOven, iteration);
        }
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
        fail("This program needs 1 argument: the parameters file (.txt). Eg.: ./microwave param.txt");
    }

    printf("Loading the parameters...\n");
    Parameters *pParameters = load_parameters(argv[1]);
    if (pParameters->time_step > pParameters->simulation_time)
    {
        fail("The time step must be lower than the simulation time!");
    }

    Oven *pOven = compute_oven(pParameters);

    printf("Initializing fields\n");
    Fields *pFields = initialize_fields(pParameters);

    printf("Creating mesh\n");

    printf("Setting initial conditions\n");
    set_initial_conditions(pFields->Ey, pParameters);
    printf("Launching simulation\n");
    propagate_fields(pFields, pParameters, pOven);

    printf("Freeing memory...\n");
    freeAll();

    printf("Simulation complete!\n");
    return 0;
}
