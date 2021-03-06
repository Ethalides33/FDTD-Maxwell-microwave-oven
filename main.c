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
#define CELERITY 299792458.0

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
 *    width:            a=>x in figure (y, in paper cs) 
 *    height:           b =>y in figure (z, in paper cs)
 *    length:           d=>d in figure (x, in paper cs)
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
 *  dims:    Array of the sizes of each dimension (maxi+1, maxj+1, maxk+1)
 *  vdims:   Array of the sizez of each dimension for variables (maxi, maxj, mak)
 *  coords:  Cordinates of each mesh point (grid)
 *  tmpV:    A temporary vector of size maxi*maxj*maxk for averaging the fields before exporting to silo.
**/
typedef struct oven
{
    int *dims;
    int *vdims;
    double **coords;
    double *tmpV;
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

    fscanf(fParams, "%f", &pParameters->length);
    fscanf(fParams, "%f", &pParameters->width);
    fscanf(fParams, "%f", &pParameters->height);
    fscanf(fParams, "%lf", &pParameters->spatial_step);
    fscanf(fParams, "%lf", &pParameters->time_step);
    fscanf(fParams, "%f", &pParameters->simulation_time);
    fscanf(fParams, "%u", &pParameters->sampling_rate);
    fscanf(fParams, "%x", &pParameters->mode);

    fclose(fParams);

    pParameters->maxi = (size_t)(pParameters->length / pParameters->spatial_step);
    pParameters->maxj = (size_t)(pParameters->width / pParameters->spatial_step);
    pParameters->maxk = (size_t)(pParameters->height / pParameters->spatial_step);

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
    r->vdims = Malloc(sizeof(size_t) * 3);
    r->coords = Malloc(sizeof(double **) * 3);
    r->dims[0] = params->maxi + 1;
    r->dims[1] = params->maxj + 1;
    r->dims[2] = params->maxk + 1;
    r->vdims[0] = params->maxi;
    r->vdims[1] = params->maxj;
    r->vdims[2] = params->maxk;
    r->tmpV = Malloc(sizeof(double) * params->maxi * params->maxj * params->maxk);

    double *x = Malloc((params->maxi + 1) * sizeof(double));
    double *y = Malloc((params->maxj + 1) * sizeof(double));
    double *z = Malloc((params->maxk + 1) * sizeof(double));

    //TODO: Optimization: iterate once to the bigger and affect if in bounds of array...
    double dx = params->spatial_step;
    for (int i = 0; i < params->maxi + 1; ++i)
        x[i] = i * dx;

    for (int i = 0; i < params->maxj + 1; ++i)
        y[i] = i * dx;

    for (int i = 0; i < params->maxk + 1; ++i)
        z[i] = i * dx;

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

    // Ex
    size_t space_size = params->maxi * (params->maxj + 1) * (params->maxk + 1);

    pFields->Ex = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Ex[space_size] = 0.0;
    }

    // Ey
    space_size = (params->maxi + 1) * params->maxj * (params->maxk + 1);

    pFields->Ey = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Ey[space_size] = 0.0;
    }

    // Ez
    space_size = (params->maxi + 1) * (params->maxj + 1) * params->maxk;

    pFields->Ez = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Ez[space_size] = 0.0;
    }

    // Hx
    space_size = (params->maxi + 1) * params->maxj * params->maxk;

    pFields->Hx = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Hx[space_size] = 0.0;
    }

    // Hy
    space_size = params->maxi * (params->maxj + 1) * params->maxk;

    pFields->Hy = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Hy[space_size] = 0.0;
    }

    // Hz
    space_size = params->maxi * params->maxj * (params->maxk + 1);
    pFields->Hz = Malloc(sizeof(double) * space_size);

    while (0 < space_size)
    {
        --space_size;
        pFields->Hz[space_size] = 0.0;
    }

    return pFields;
}

/** Fast shortcut to get the index of a field at i, j and k
 * Parameters:
 *  params: The parameters of the simulation
 *  i, j, k: The coordinates of the wanted field
 *  mi, mj: The additionnal sizes of dimensions X and Y.
 * Returns:
 *  The index in a 1D array
**/
static inline size_t idx(Parameters *params, size_t i, size_t j, size_t k, size_t mi, size_t mj)
{
    return i + j * (params->maxi + mi) + k * (params->maxi + mi) * (params->maxj + mj);
}

static inline size_t kEx(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * p->maxi + k * p->maxi * (p->maxj + 1);
}

static inline size_t kEy(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * (p->maxi + 1) + k * (p->maxi + 1) * p->maxj;
}

static inline size_t kEz(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * (p->maxi + 1) + k * (p->maxi + 1) * (p->maxj + 1);
}

static inline size_t kHx(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * (p->maxi + 1) + k * (p->maxi + 1) * p->maxj;
}

static inline size_t kHy(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * p->maxi + k * p->maxi * (p->maxj + 1);
}

static inline size_t kHz(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * p->maxi + k * p->maxi * p->maxj;
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
    for (k = 0; k < p->maxk + 1; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 0; i < p->maxi + 1; ++i)
                Ey[kEy(p, i, j, k)] = sin(PI * k * p->spatial_step / p->height) *
                                      sin(PI * i * p->spatial_step / p->length);
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

    double factor = p->time_step / (MU * p->spatial_step);

    size_t i, j, k;

    for (k = 0; k < p->maxk; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 0; i < p->maxi + 1; ++i)
                Hx[kHx(p, i, j, k)] += factor * ((Ey[kEy(p, i, j, k + 1)] - Ey[kEy(p, i, j, k)]) -
                                                 (Ez[kEz(p, i, j + 1, k)] - Ez[kEz(p, i, j, k)]));

    for (k = 0; k < p->maxk; ++k)
        for (j = 0; j < p->maxj + 1; ++j)
            for (i = 0; i < p->maxi; ++i)
                Hy[kHy(p, i, j, k)] += factor * ((Ez[kEz(p, i + 1, j, k)] - Ez[kEz(p, i, j, k)]) -
                                                 (Ex[kEx(p, i, j, k + 1)] - Ex[kEx(p, i, j, k)]));

    for (k = 0; k < p->maxk + 1; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 0; i < p->maxi; ++i)
                Hz[kHz(p, i, j, k)] += factor * ((Ex[kEx(p, i, j + 1, k)] - Ex[kEx(p, i, j, k)]) -
                                                 (Ey[kEy(p, i + 1, j, k)] - Ey[kEy(p, i, j, k)]));
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

    double factor = p->time_step / (EPSILON * p->spatial_step);

    size_t i, j, k;

    for (k = 1; k < p->maxk; ++k)
        for (j = 1; j < p->maxj; ++j)
            for (i = 0; i < p->maxi; ++i)
                Ex[kEx(p, i, j, k)] += factor * ((Hz[kHz(p, i, j, k)] - Hz[kHz(p, i, j - 1, k)]) -
                                                 (Hy[kHy(p, i, j, k)] - Hy[kHy(p, i, j, k - 1)]));

    for (k = 1; k < p->maxk; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 1; i < p->maxi; ++i)
                Ey[kEy(p, i, j, k)] += factor * ((Hx[kHx(p, i, j, k)] - Hx[kHx(p, i, j, k - 1)]) -
                                                 (Hz[kHz(p, i, j, k)] - Hz[kHz(p, i - 1, j, k)]));

    for (k = 0; k < p->maxk; ++k)
        for (j = 1; j < p->maxj; ++j)
            for (i = 1; i < p->maxi; ++i)
                Ez[kEz(p, i, j, k)] += factor * ((Hy[kHy(p, i, j, k)] - Hy[kHy(p, i - 1, j, k)]) -
                                                 (Hx[kHx(p, i, j, k)] - Hx[kHx(p, i, j - 1, k)]));
}

/** Computes the mean of an electrical field 
 * Parameters:
 *  p: The simulation parameters
 *  Ef: The E field component in one direction
 *  r:  The result aggregated vector of size (maxi, maxj, maxk)
 *  ofi: The offset in X (related to the space size)
 *  ofj: The offset in Y (related to the space size)
 *  ofk: The offset in Z (related to the space size)
**/
void aggregate_E_field(Parameters *p, double *Ef, double *r, size_t ofi, size_t ofj, size_t ofk)
{
    size_t t = 0;
    for (size_t k = 0; k < p->maxk; ++k)
        for (size_t j = 0; j < p->maxj; ++j)
            for (size_t i = 0; i < p->maxi; ++i)
                r[t++] = .25 * (Ef[idx(p, i, j, k, ofi, ofj)] +
                                Ef[idx(p, i + ofi, j + ofj, k + ofk, ofi, ofj)] +
                                Ef[idx(p, i, j + ofj, k + ofk, ofi, ofj)] +
                                Ef[idx(p, i + ofi, j, k + ofk, ofi, ofj)]);
}

/** Computes the mean of an magnetic field 
 * Parameters:
 *  p: The simulation parameters
 *  Hf: The H field component in one direction
 *  r:  The result aggregated vector of size (maxi, maxj, maxk)
 *  ofi: The offset in X (related to the space size)
 *  ofj: The offset in Y (related to the space size)
 *  ofk: The offset in Z (related to the space size)
**/
void aggregate_H_field(Parameters *p, double *Hf, double *r, size_t ofi, size_t ofj, size_t ofk)
{
    size_t t = 0;
    for (size_t k = 0; k < p->maxk; ++k)
        for (size_t j = 0; j < p->maxj; ++j)
            for (size_t i = 0; i < p->maxi; ++i)
                r[t++] = .5 * (Hf[idx(p, i, j, k, ofi, ofj)] +
                               Hf[idx(p, i + ofi, j + ofj, k + ofk, ofi, ofj)]);
}

/** Writes a silo file of the simulation in the given timestamp
 * Parameters:
 *  pFields: The fields
 *  pValidationFields: The validation fields
 *  pParams: The parameters of the simulation
 *  pOven: The oven computed
 *  iteration: The iteration count
**/
void write_silo(Fields *pFields, Fields *pValidationFields, Parameters *pParams, Oven *pOven, int iteration)
{
    char filename[100];
    sprintf(filename, DB_FILENAME, iteration);

    DBfile *dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
    if (!dbfile)
    {
        fail("Could not create DB\n");
    }

    DBPutQuadmesh(dbfile, DB_MESHNAME, NULL, pOven->coords, pOven->dims, 3, DB_DOUBLE, DB_COLLINEAR, NULL);

    aggregate_E_field(pParams, pFields->Ex, pOven->tmpV, 0, 1, 1);
    DBPutQuadvar1(dbfile, "ex", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    aggregate_E_field(pParams, pFields->Ey, pOven->tmpV, 1, 0, 1);
    DBPutQuadvar1(dbfile, "ey", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    aggregate_E_field(pParams, pFields->Ez, pOven->tmpV, 1, 1, 0);
    DBPutQuadvar1(dbfile, "ez", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    aggregate_H_field(pParams, pFields->Hx, pOven->tmpV, 1, 0, 0);
    DBPutQuadvar1(dbfile, "hx", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    aggregate_H_field(pParams, pFields->Hy, pOven->tmpV, 0, 1, 0);
    DBPutQuadvar1(dbfile, "hy", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    aggregate_H_field(pParams, pFields->Hz, pOven->tmpV, 0, 0, 1);
    DBPutQuadvar1(dbfile, "hz", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    if (pParams->mode == VALIDATION_MODE)
    {
        aggregate_E_field(pParams, pValidationFields->Ey, pOven->tmpV, 1, 0, 1);
        DBPutQuadvar1(dbfile, "aEy", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
        aggregate_H_field(pParams, pFields->Hx, pOven->tmpV, 1, 0, 0);
        DBPutQuadvar1(dbfile, "aHx", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
        aggregate_H_field(pParams, pFields->Hz, pOven->tmpV, 0, 0, 1);
        DBPutQuadvar1(dbfile, "aHz", DB_MESHNAME, pOven->tmpV, pOven->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    }

    const char *names[] = {"E", "H"};
    const char *defs[] = {"{ex, ey, ez}", "{hx, hy, hz}"};
    const int types[] = {DB_VARTYPE_VECTOR, DB_VARTYPE_VECTOR};

    DBPutDefvars(dbfile, "vecs", 2, names, types, defs, NULL);

    DBClose(dbfile);
}

/** Computes the total electrical energy in the system
 * **/
double calculate_E_energy(Fields *pFields, Parameters *p)
{
    double *Ex = pFields->Ex;
    double *Ey = pFields->Ey;
    double *Ez = pFields->Ez;

    double ex_energy = 0.0;
    double ey_energy = 0.0;
    double ez_energy = 0.0;

    double mean_ex, mean_ey, mean_ez;
    double dv = pow(p->spatial_step, 3); // volume element

    size_t i, j, k;

    for (k = 0; k < p->maxk; k++)
        for (j = 0; j < p->maxj; j++)
            for (i = 0; i < p->maxi; i++)
            {
                mean_ex = (Ex[kEx(p, i, j, k)] + Ex[kEx(p, i, j, k + 1)] + Ex[kEx(p, i, j + 1, k)] + Ex[kEx(p, i, j + 1, k + 1)]) / 4.;
                ex_energy += pow(mean_ex, 2) * dv;

                mean_ey = (Ey[kEy(p, i, j, k)] + Ey[kEy(p, i + 1, j, k)] + Ey[kEy(p, i, j, k + 1)] + Ey[kEy(p, i + 1, j, k + 1)]) / 4.;
                ey_energy += pow(mean_ey, 2) * dv;

                mean_ez = (Ez[kHz(p, i, j, k)] + Ez[kHz(p, i, j + 1, k)] + Ez[kHz(p, i + 1, j, k)] + Ez[kHz(p, i + 1, j + 1, k)]) / 4.;
                ez_energy += pow(mean_ez, 2) * dv;
            }

    double E_energy = (ex_energy + ey_energy + ez_energy) * EPSILON / 2.;

    return E_energy;
}

double calculate_H_energy(Fields *pFields, Parameters *p)
{
    double *Hx = pFields->Hx;
    double *Hy = pFields->Hy;
    double *Hz = pFields->Hz;

    double hx_energy = 0.0;
    double hy_energy = 0.0;
    double hz_energy = 0.0;

    double mean_hx, mean_hy, mean_hz;
    double dv = pow(p->spatial_step, 3); // volume element

    size_t i, j, k;

    for (k = 0; k < p->maxk; k++)
        for (j = 0; j < p->maxj; j++)
            for (i = 0; i < p->maxi; i++)
            {
                mean_hx = (Hx[kHx(p, i, j, k)] + Hx[kHx(p, i + 1, j, k)]) / 2.;
                hx_energy += pow(mean_hx, 2) * dv;

                mean_hy = (Hy[kHy(p, i, j, k)] + Hy[kHy(p, i, j + 1, k)]) / 2.;
                hy_energy += pow(mean_hy, 2) * dv;

                mean_hz = (Hz[kHz(p, i, j, k)] + Hz[kHz(p, i, j, k + 1)]) / 2.;
                hz_energy += pow(mean_hz, 2) * dv;
            }

    double H_energy = (hx_energy + hy_energy + hz_energy) * MU / 2.;

    return H_energy;
}

void update_validation_fields_then_subfdtd(Parameters *p, Fields *pFields, Fields *pValidationFields, double time_counter)
{
    double f_mnl = 0.5 * CELERITY * sqrt(pow(PI / p->height, 2) + pow(PI / p->length, 2)) / PI;
    //printf("fmnl: %0.20f \n", f_mnl);
    double omega = 2.0 * PI * f_mnl;
    double Z_te = (omega * MU) / sqrt(pow(omega, 2) * MU * EPSILON - pow(PI / p->length, 2));
    //printf("Zte: %0.20f \n", Z_te);
    //printf("frequency: %0.10f \n", f_mnl);
    //printf("z_te: %0.10f \n", Z_te);

    double *vEy = pValidationFields->Ey;
    double *vHx = pValidationFields->Hx;
    double *vHz = pValidationFields->Hz;

    size_t i, j, k;
    for (k = 0; k < p->maxk + 1; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 0; i < p->maxi + 1; ++i)
                vEy[kEy(p, i, j, k)] = (cos(2 * PI * f_mnl * time_counter) *
                                        sin(PI * k * p->spatial_step / p->height) *
                                        sin(PI * i * p->spatial_step / p->length)) -
                                       pFields->Ey[kEy(p, i, j, k)];

    for (k = 0; k < p->maxk; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 0; i < p->maxi + 1; ++i)
                vHx[kHx(p, i, j, k)] = ((1.0 / Z_te) *
                                        sin(2 * PI * f_mnl * time_counter) *
                                        sin(PI * k * p->spatial_step / p->height) *
                                        cos(PI * i * p->spatial_step / p->length)) -
                                       pFields->Hx[kHx(p, i, j, k)];

    for (k = 0; k < p->maxk + 1; ++k)
        for (j = 0; j < p->maxj; ++j)
            for (i = 0; i < p->maxi; ++i)
                vHz[kHz(p, i, j, k)] = (-PI / (omega * MU * p->length) *
                                        sin(2 * PI * f_mnl * time_counter) *
                                        cos(PI * k * p->spatial_step / p->height) *
                                        sin(PI * i * p->spatial_step / p->length)) -
                                       pFields->Hz[kHz(p, i, j, k)];
}

void set_source(Parameters *p, Fields *pFields, double time_counter)
{

    double *Ex = pFields->Ex;
    double *Ez = pFields->Ez;
    double *Hx = pFields->Hx;
    double *Hz = pFields->Hz;

    double aprime = 0.005;
    double bprime = 0.005;

    double min_y = p->width / 2. - aprime / 2.;
    double max_y = min_y + aprime;

    double min_x = p->length / 2. - bprime / 2.;
    double max_x = min_x + bprime;

    double min_j = (int)(min_y / p->spatial_step) - 1;
    double max_j = (int)(max_y / p->spatial_step) + 1;

    double min_i = (int)(min_x / p->spatial_step) - 1;
    double max_i = (int)(max_x / p->spatial_step) + 1;

    double f = 2.45e10;

    double f_mnl = 0.5 * CELERITY * sqrt(pow(PI / p->width, 2) + pow(PI / p->length, 2)) / PI;
    double omega = 2.0 * PI * f_mnl;
    double Z_te = (omega * MU) / sqrt(pow(omega, 2) * MU * EPSILON - pow(PI / p->width, 2));

    size_t i, j, k;
    size_t shift_i, shift_j;

    //printf("%f, %f, %f, %f, \n", min_i, max_i, min_j, max_j);
    for (i = min_i, shift_i = 0; i < max_i; ++i, ++shift_i)
        for (j = min_j, shift_j = 0; j < max_j; ++j, ++shift_j)
        {
            Ez[kEz(p, i, j, 0)] = sin(2 * PI * f * time_counter) * sin(PI * (shift_i * p->spatial_step) / aprime); // i = 0 pour face x =0 // -aprime??
            Ex[kEx(p, i, j, 0)] = 0;
            Hz[kHz(p, i, j, 0)] = 0;
            Hx[kHx(p, i, j, 0)] = -(1.0 / Z_te) * sin(2 * PI * f * time_counter) * sin(PI * (shift_i * p->spatial_step) / aprime);
        }
}

void propagate_fields(Fields *pFields, Fields *pValidationFields, Parameters *pParams, Oven *pOven)
{
    double time_counter;
    int iteration = 1;
    double total_energy = calculate_E_energy(pFields, pParams) + calculate_H_energy(pFields, pParams);
    if (pParams->mode == VALIDATION_MODE)
    {
        update_validation_fields_then_subfdtd(pParams, pFields, pValidationFields, 0.0);
    }
    write_silo(pFields, pValidationFields, pParams, pOven, iteration);
    for (time_counter = 0; time_counter <= pParams->simulation_time; time_counter += pParams->time_step, iteration++)
    {
        //printf("time: %0.10f s\n", time_counter);
        //below should be parallelized.

        if (pParams->mode == COMPUTATION_MODE)
        {
            set_source(pParams, pFields, time_counter);
        }
        update_H_field(pParams, pFields);
        if (pParams->mode == COMPUTATION_MODE)
        {
            set_source(pParams, pFields, time_counter);
        }
        update_E_field(pParams, pFields);

        if (pParams->mode == VALIDATION_MODE)
        {
            update_validation_fields_then_subfdtd(pParams, pFields, pValidationFields, time_counter);
        }

        if (pParams->mode == VALIDATION_MODE)
        {
            //printf("Electrical energy: %0.20f \n", calculate_E_energy(pFields, pParams));
            //printf("Magnetic energy: %0.20f \n", calculate_H_energy(pFields, pParams));
            //printf("Tot energy: %0.20f \n", calculate_E_energy(pFields, pParams) + calculate_H_energy(pFields, pParams));
            //printf("Theoretical energy: %0.20f \n", (EPSILON * pParams->length * pParams->width * pParams->height) / 8.);
            //assert((calculate_E_energy(pFields, pParams) + calculate_H_energy(pFields, pParams) - total_energy) <= 0.000001);
        }
        if (iteration % pParams->sampling_rate == 0)
        {
            write_silo(pFields, pValidationFields, pParams, pOven, iteration);
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
    Fields *pValidationFields;

    if (pParameters->mode == VALIDATION_MODE)
    {
        pValidationFields = initialize_fields(pParameters);
        printf("Validation mode activated. \n");
        // Free what's not needed for validation.
        Free(pValidationFields->Ex);
        Free(pValidationFields->Ez);
        Free(pValidationFields->Hy);
    }

    printf("Creating mesh\n");

    printf("Setting initial conditions\n");

    if (pParameters->mode == VALIDATION_MODE)
        set_initial_conditions(pFields->Ey, pParameters);

    printf("Launching simulation\n");
    propagate_fields(pFields, pValidationFields, pParameters, pOven);
    printf("Freeing memory...\n");
    freeAll();

    printf("Simulation complete!\n");
    return 0;
}
