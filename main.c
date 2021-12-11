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
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <silo.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>

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
*/
/** Execution modes for this program.
 * VALIDATION_MODE: There is no source but an initial condition given by the equantion in the statement
 * COMOPUTATION_MODE: The source is set and continuously radiates the simulation box
*/
typedef enum MODE
{
    VALIDATION_MODE = 0,
    COMPUTATION_MODE = 1
} MODE;

/** TAGS used for diverse OpenMPI communications */
typedef enum TAGS
{
    EX_TAG_TO_UP = 0,
    EY_TAG_TO_UP,
    EZ_TAG_TO_UP,
    HX_TAG_TO_UP,
    HY_TAG_TO_UP,
    HZ_TAG_TO_UP,
    EX_TAG_TO_DOWN,
    EY_TAG_TO_DOWN,
    EZ_TAG_TO_DOWN,
    HX_TAG_TO_DOWN,
    HY_TAG_TO_DOWN,
    HZ_TAG_TO_DOWN,

    EX_TAG_TO_MAIN,
    EY_TAG_TO_MAIN,
    EZ_TAG_TO_MAIN,
    HX_TAG_TO_MAIN,
    HY_TAG_TO_MAIN,
    HZ_TAG_TO_MAIN,
} TAGS;

/** An approach to manage memory allocations in a chained list
 * Properties:
 *      previous:   The previous allocated object
 *      ptr:        The current allocated object
 * Description:
 *      This structure allow us to store each allocated object
 *      into a chained list with a LiFo strategy so when we
 *      free up all memory we take care of the inner most objects
*/
typedef struct chainedAllocated
{
    struct chainedAllocated *previous;
    void *ptr;
} ChainedAllocated;

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
 * 
 *    rank:             The rank of the current CPU (MPI)
 *    ranks:            The number of ranks, aka. world_size (MPI)
 *    k_layers:         The number of layers on the (X,Y) plane treaten by current CPU
 *    lower_cpu:        The rank of the CPU working on the plane below (Z-1)
 *    upper_cpu:        The rank of the CPU working on the plane above (Z+1)
 *    ls:               The chained list of allocated objects
*/
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
    unsigned int sampling_rate;
    MODE mode;

    // Parallelization
    int rank;
    int ranks;
    size_t startk;
    size_t k_layers;
    int lower_cpu;
    int upper_cpu;

    ChainedAllocated *ls;
} Parameters;

/** Definition of the oven mesh (can be derived directly from Parameters), it's useless to recompute it at each timestep
 * Properties:
 *  dims:    Array of the sizes of each dimension (maxi+1, maxj+1, maxk+1)
 *  vdims:   Array of the sizez of each dimension for variables (maxi, maxj, mak)
 *  coords:  Cordinates of each mesh point (grid)
 *  tmpV:    A temporary vector of size maxi*maxj*maxk for averaging the fields before writting silo file.
*/
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
 */
typedef struct fields
{

    double *Ex;
    double *Ey;
    double *Ez;
    double *Hx;
    double *Hy;
    double *Hz;

} Fields;


/**
------------------------------------------------------
---------------------------- Function's Specifications
------------------------- Signatures & Implementations
------------------------------------------------------
*/
/** Free any allocated object with the Malloc function (see below)
 * Parameters:
 *  ls: The list of allocated objects
 * Description:
 *  This function free's the memory allocated for all
 *  the objects allocated with the next Malloc function
 *  in the reverse way of their allocation (LiFo) in order
 *  to deal with nested structures.
*/
void *freeAll(ChainedAllocated *ls)
{
    while (ls && ls->ptr != ls)
    {
        printf("Freing %p...\n", ls->ptr);
        free(ls->ptr);
        ChainedAllocated *previous = ls->previous;
        free(ls);
        ls = previous;
    }
    free(ls);
}

/** Free the memory and throws an error, then exits with EXIT_FAILURE.
 * Parameters:
 *  ls:     The chained list of allocated objects
 *  msg:    The message to throw before exit
*/
void fail(ChainedAllocated *ls, const char *msg)
{
    perror(msg);
    freeAll(ls);
    MPI_Finalize();
    exit(EXIT_FAILURE);
}

/** Critical allocation (malloc or fail)
 * Parameters:
 *  pLs:    A pointer to the pointer of the list of allocated objects.
 * Description:
 *  In addition to critically check if the malloc properly worked,
 *  this function stores the reference to the new allocated object
 *  in the superglobal chained list which is then used to free the
 *  memory before exit.
*/
void *Malloc(ChainedAllocated **pLs, size_t size)
{
    ChainedAllocated *ls = *pLs;
    if (ls == NULL)
    {
        ls = malloc(sizeof(ChainedAllocated));
        ls->ptr = ls;
        ls->previous = NULL;
        *pLs=ls;
    }
    
    ChainedAllocated *successor = malloc(sizeof(ChainedAllocated));
    successor->previous = ls;
    ls = successor;
    *pLs=ls;
    
    void *ptr = malloc(size);

    ls->ptr = ptr;

    if (!ptr)
        fail(ls, "CRITICAL ERROR: Could not allocate enough memory!");

    return ptr;
}

/**
 * @brief Utility to allocate array of doubles and initialize it to 0.0
 * @param pLs The pointer to a pointer of the list of allocated pointers
 * @param len The length of the array 
 */
double *Malloc_Double(ChainedAllocated **pLs, size_t len)
{
    double *ptr = Malloc(pLs, sizeof(double) * len);
    while(0 < len){
        --len;
        ptr[len] = 0.0;
    }
    return ptr;
}

/**
 * @brief Utility to print the chained list of pointers
 * 
 * @param ls The pointer to the chained list of poinetrs
 */
static void printHeap(ChainedAllocated *ls){
    printf("==== Heap addresses dump ==== \n");
    while (ls)
    {
        printf("Current: %p || ptr: %p || Previous: %p\n", ls, ls->ptr, ls->previous);
        ls = ls->previous;
    }
    printf("==== Heap addresses dump END ==== \n");
}

/** For an object allocated with the Malloc function above, frees the memory and removes the entry from the chained list.
 * Parameters:
 *  ls:    A pointer to the list containing the allocated objects
 *  ptr:    The pointer to free
*/
void Free(ChainedAllocated **pLs, void *ptr)
{
    ChainedAllocated *current = *pLs;
    if (current->ptr == ptr){
        *pLs = current->previous;
        free(current->ptr);
        free(current);
        return;
    }
    Free(&current->previous, ptr);
}

/** Loads the parameters into the system memory
 * Parameters:
 *    filename: The file containing the parameters properties (.txt)
 * Return:
 *    A pointer to the parameters structure loaded in system memory
*/
Parameters *load_parameters(const char *filename)
{
    FILE *fParams = fopen(filename, "r");
    ChainedAllocated *ls = NULL;
    Parameters *pParameters = Malloc(&ls, sizeof(Parameters));

    if (!fParams)
        fail(ls, "Unable to open parameters file!");

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

    // Parallelization:
    MPI_Comm_size(MPI_COMM_WORLD, &pParameters->ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &pParameters->rank);

    pParameters->startk = pParameters->maxk * pParameters->rank / pParameters->ranks;
    size_t endk = pParameters->maxk * (pParameters->rank + 1) / pParameters->ranks;
    pParameters->k_layers = endk - pParameters->startk;
    pParameters->lower_cpu = pParameters->rank > 0 ? pParameters->rank - 1 : MPI_PROC_NULL;
    pParameters->upper_cpu = pParameters->rank < pParameters->ranks - 1 ? pParameters->rank + 1 : MPI_PROC_NULL;
    pParameters->ls=ls;

    return pParameters;
}

/** Compute the oven properties and returns it
 * Parameters:
 *  params: The parameters of the simulation
 * Returns:
 *  A pointer to the oven structure
*/
Oven *compute_oven(Parameters *params)
{
    Oven *r = Malloc(&params->ls, sizeof(Oven));

    r->dims = Malloc(&params->ls, sizeof(size_t) * 3);
    r->vdims = Malloc(&params->ls, sizeof(size_t) * 3);
    r->coords = Malloc(&params->ls, sizeof(double **) * 3);
    r->dims[0] = params->maxi + 1;
    r->dims[1] = params->maxj + 1;
    r->dims[2] = params->maxk + 1;
    r->vdims[0] = params->maxi;
    r->vdims[1] = params->maxj;
    r->vdims[2] = params->maxk;
    r->tmpV = Malloc(&params->ls, sizeof(double) * params->maxi * params->maxj * params->maxk);

    double *x = Malloc(&params->ls, (params->maxi + 1) * sizeof(double));
    double *y = Malloc(&params->ls, (params->maxj + 1) * sizeof(double));
    double *z = Malloc(&params->ls, (params->maxk + 1) * sizeof(double));

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

/** Gives the size of the simulation XY plane depending on the field/field component
 * Parameters:
 *  p:      The parameters of the simulation
 *  fields: The fields
 *  field:  The field for which you want the size of XY plane
**/
size_t sizeof_xy_plane(Parameters *p, Fields *fields, double *field)
{
    if(field == fields->Ex)
        return p->maxi * (p->maxj+1);
    if(field == fields->Ey)
        return (p->maxi+1)*p->maxj;
    if(field == fields->Ez)
        return (p->maxi+1) * (p->maxj+1);
    if(field == fields->Hx)
        return (p->maxi+1)*p->maxj;
    if(field == fields->Hy)
        return p->maxi*(p->maxj+1);
    if(field == fields->Hz)
        return p->maxi * p->maxj;
    return 0;
}

/** Allocates and initialize to 0.0 all the components of each field at a given time t.
 * Parameters:
 *  params: The parameters of the simulation
*/
static Fields *initialize_fields(Parameters *params)
{
    Fields *pFields = Malloc(&params->ls, sizeof(Fields));

    // Ex
    size_t space_size = params->maxi * (params->maxj + 1) * (params->maxk + 1);
    pFields->Ex = Malloc_Double(&params->ls, space_size);

    // Ey
    space_size = (params->maxi + 1) * params->maxj * (params->maxk + 1);
    pFields->Ey = Malloc_Double(&params->ls, space_size);

    // Ez
    space_size = (params->maxi + 1) * (params->maxj + 1) * params->maxk;
    pFields->Ez = Malloc_Double(&params->ls, space_size);

    // Hx
    space_size = (params->maxi + 1) * params->maxj * params->maxk;
    pFields->Hx = Malloc_Double(&params->ls, space_size);


    // Hy
    space_size = params->maxi * (params->maxj + 1) * params->maxk;
    pFields->Hy = Malloc_Double(&params->ls, space_size);

    // Hz
    space_size = params->maxi * params->maxj * (params->maxk + 1);
    pFields->Hz = Malloc_Double(&params->ls, space_size);

    return pFields;
}

/** Allocates and initialize to 0.0 all the components of each field at a given time t for this cpu
 * Parameters:
 *  params: The parameters of the simulation
*/
Fields *initialize_cpu_fields(Parameters *params)
{
    Fields *pFields = Malloc(&params->ls, sizeof(Fields));

    // Ex
    size_t space_size = params->maxi * (params->maxj + 1) * (params->k_layers + 2);
    pFields->Ex = Malloc_Double(&params->ls, space_size);

    // Ey
    space_size = (params->maxi + 1) * params->maxj * (params->k_layers + 2);
    pFields->Ey = Malloc_Double(&params->ls, space_size);

    // Ez
    space_size = (params->maxi + 1) * (params->maxj + 1) * (params->k_layers + 2);
    pFields->Ez = Malloc_Double(&params->ls, space_size);

    // Hx
    space_size = (params->maxi + 1) * params->maxj * (params->k_layers + 2);
    pFields->Hx = Malloc_Double(&params->ls, space_size);

    // Hy
    space_size = params->maxi * (params->maxj + 1) * (params->k_layers + 2);
    pFields->Hy = Malloc_Double(&params->ls, space_size);

    // Hz
    space_size = params->maxi * params->maxj * (params->k_layers + 2);
    pFields->Hz = Malloc_Double(&params->ls, space_size);

    return pFields;
}

/** Fast shortcut to get the index of a field at i, j and k
 * Parameters:
 *  params: The parameters of the simulation
 *  i, j, k: The coordinates of the wanted field
 *  mi, mj: The additionnal sizes of dimensions X and Y.
 * Returns:
 *  The index in a 1D array
*/
size_t idx(Parameters *params, size_t i, size_t j, size_t k, size_t mi, size_t mj)
{
    return i + j * (params->maxi + mi) + k * (params->maxi + mi) * (params->maxj + mj);
}

size_t kEx(Parameters *p, size_t i, size_t j, size_t k)
{
    return idx(p, i, j, k, 0, 1);
}

size_t kEy(Parameters *p, size_t i, size_t j, size_t k)
{
    return idx(p, i, j, k, 1, 0);
}

size_t kEz(Parameters *p, size_t i, size_t j, size_t k)
{
    return idx(p, i, j, k, 1, 1);
}

size_t kHx(Parameters *p, size_t i, size_t j, size_t k)
{
    return idx(p, i, j, k, 1, 0);
}

size_t kHy(Parameters *p, size_t i, size_t j, size_t k)
{
    return idx(p, i, j, k, 0, 1);
}

size_t kHz(Parameters *p, size_t i, size_t j, size_t k)
{
    return idx(p, i, j, k, 0, 0);
}

/** Sets the initial field as asked in Question 3.a.
 * Parameters:
 *  Ey: The y component of the Energy fields
 *  p:   The parameters of the simulation
*/
void set_initial_conditions(double *Ey, Parameters *p)
{
    size_t i, j, k;
    for (i = 0; i < p->maxi + 1; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 1; k < p->k_layers+2; ++k)
                Ey[kEy(p, i, j, k)] = sin(PI * (p->startk + k) * p->spatial_step / p->width) *
                                      sin(PI * i * p->spatial_step / p->length);
}

/** Updates the H field
 * Parameters:
 *  p:      The parameters of the simulation
 *  fields: The fields
*/
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

    for (i = 0; i < p->maxi + 1; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 1; k < p->k_layers + 1; ++k)
                Hx[kHx(p, i, j, k)] += factor * ((Ey[kEy(p, i, j, k + 1)] - Ey[kEy(p, i, j, k)]) -
                                                 (Ez[kEz(p, i, j + 1, k)] - Ez[kEz(p, i, j, k)]));

    for (i = 0; i < p->maxi; ++i)
        for (j = 0; j < p->maxj + 1; ++j)
            for (k = 1; k < p->k_layers + 1; ++k)
                Hy[kHy(p, i, j, k)] += factor * ((Ez[kEz(p, i + 1, j, k)] - Ez[kEz(p, i, j, k)]) -
                                                 (Ex[kEx(p, i, j, k + 1)] - Ex[kEx(p, i, j, k)]));
    size_t ofst = p->rank == p->ranks-1?2:1;
    ofst += p->k_layers;
    for (i = 0; i < p->maxi; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 1; k < ofst; ++k)
                Hz[kHz(p, i, j, k)] += factor * ((Ex[kEx(p, i, j + 1, k)] - Ex[kEx(p, i, j, k)]) -
                                                 (Ey[kEy(p, i + 1, j, k)] - Ey[kEy(p, i, j, k)]));
}

/** Updates the E field
 * Parameters:
 *  p:      The parameters of the simulation
 *  fields: The fields
*/
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

    size_t startk = p->rank==0?2:1;
    for (i = 0; i < p->maxi; ++i)
        for (j = 1; j < p->maxj; ++j)
            for (k = startk; k < p->k_layers + 1; ++k)
                Ex[kEx(p, i, j, k)] += factor * ((Hz[kHz(p, i, j, k)] - Hz[kHz(p, i, j - 1, k)]) -
                                                 (Hy[kHy(p, i, j, k)] - Hy[kHy(p, i, j, k - 1)]));
    for (i = 1; i < p->maxi; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = startk; k < p->k_layers + 1; ++k)
                Ey[kEy(p, i, j, k)] += factor * ((Hx[kHx(p, i, j, k)] - Hx[kHx(p, i, j, k - 1)]) -
                                                 (Hz[kHz(p, i, j, k)] - Hz[kHz(p, i - 1, j, k)]));

    for (i = 1; i < p->maxi; ++i)
        for (j = 1; j < p->maxj; ++j)
            for (k = 1; k < p->k_layers + 1; ++k){
                Ez[kEz(p, i, j, k)] += factor * ((Hy[kHy(p, i, j, k)] - Hy[kHy(p, i - 1, j, k)]) -
                                                 (Hx[kHx(p, i, j, k)] - Hx[kHx(p, i, j - 1, k)]));
            }/*             if( k==p->k_layers) //k==p->k_layers ||
                    Ez[kEz(p, i, j, k)] = 2;
                if (k==1)
                    Ez[kEz(p, i, j, k)] = -2;*/
}

/** Computes the mean of an electrical field 
 * Parameters:
 *  p: The simulation parameters
 *  Ef: The E field component in one direction
 *  r:  The result aggregated vector of size (maxi, maxj, maxk)
 *  ofi: The offset in X (related to the space size)
 *  ofj: The offset in Y (related to the space size)
 *  ofk: The offset in Z (related to the space size)
*/
void aggregate_E_field(Parameters *p, double *Ef, double *r, size_t ofi, size_t ofj, size_t ofk)
{
    size_t t = 0;
    for (size_t i = 0; i < p->maxi; ++i)
        for (size_t j = 0; j < p->maxj; ++j)
            for (size_t k = 0; k < p->maxk; ++k)
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
*/
void aggregate_H_field(Parameters *p, double *Hf, double *r, size_t ofi, size_t ofj, size_t ofk)
{
    size_t t = 0;
    for (size_t i = 0; i < p->maxi; ++i)
        for (size_t j = 0; j < p->maxj; ++j)
            for (size_t k = 0; k < p->maxk; ++k)
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
*/
void write_silo(Fields *pFields, Fields *pValidationFields, Parameters *pParams, Oven *pOven, int iteration)
{
    char filename[100];
    sprintf(filename, DB_FILENAME, iteration);

    DBfile *dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
    if (!dbfile)
        fail(pParams->ls, "Could not create DB\n");

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
 * Parameters:
 *  pFields: The simulated fields
 *  p:  The parameters of the simulation
*/
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

    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
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

/** Computes the H total energy
 * Parameters:
 *  pFields: The simulated fields
 *  p: The parameters of the simulation
*/
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

    for (i = 0; i < p->maxi; i++)
        for (j = 0; j < p->maxj; j++)
            for (k = 0; k < p->maxk; k++)
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

/** Updates the validation fields and substract the simulated fields to see the difference easier
 * Parameters:
 *  p:  The parameters of the simulation
 *  pFields: The simulated fields
 *  pValidationFields: The containers for the validation fields
 *  timer: The time of the simulation (in seconds)
*/
void update_validation_fields_then_subfdtd(Parameters *p, Fields *pFields, Fields *pValidationFields, double timer)
{
    double f_mnl = 0.5 * CELERITY * sqrt(pow(PI / p->width, 2) + pow(PI / p->length, 2)) / PI;
    double omega = 2.0 * PI * f_mnl;
    double Z_te = (omega * MU) / sqrt(pow(omega, 2) * MU * EPSILON - pow(PI / p->width, 2));
    //printf("frequency: %0.10f \n", f_mnl);
    //printf("z_te: %0.10f \n", Z_te);

    double *vEy = pValidationFields->Ey;
    double *vHx = pValidationFields->Hx;
    double *vHz = pValidationFields->Hz;

    size_t i, j, k;
    for (i = 1; i < p->maxi + 1; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 1; k < p->maxk + 1; ++k)
                vEy[kEy(p, i, j, k)] = (cos(2 * PI * f_mnl * timer) *
                                        sin(PI * j * p->spatial_step / p->width) *
                                        sin(PI * i * p->spatial_step / p->length)) -
                                       pFields->Ey[kEy(p, i, j, k)];

    for (i = 1; i < p->maxi + 1; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 0; k < p->maxk; ++k)
                vHx[kHx(p, i, j, k)] = ((1.0 / Z_te) *
                                        sin(2 * PI * f_mnl * timer) *
                                        sin(PI * j * p->spatial_step / p->width) *
                                        cos(PI * i * p->spatial_step / p->length)) -
                                       pFields->Hx[kHx(p, i, j, k)];

    for (i = 0; i < p->maxi; ++i)
        for (j = 0; j < p->maxj; ++j)
            for (k = 1; k < p->maxk + 1; ++k)
                vHz[kHz(p, i, j, k)] = (-PI / (omega * MU * p->width) *
                                        sin(2 * PI * f_mnl * timer) *
                                        cos(PI * j * p->spatial_step / p->width) *
                                        sin(PI * i * p->spatial_step / p->length)) -
                                       pFields->Hz[kHz(p, i, j, k)];
}

/** Sets the source in computation mode
 * Parameters:
 *  p: The parameters of the simulation
 *  pFields: The fields
 *  timer: The time of the simulation
*/
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

    size_t i, j;
    size_t shift_i, shift_j;

    //printf("%f, %f, %f, %f, \n", min_i, max_i, min_j, max_j);
    for (i = min_i, shift_i = 0; i < max_i; ++i, ++shift_i)
        for (j = min_j, shift_j = 0; j < max_j; ++j, ++shift_j)
        {
            Ez[kEz(p, i, j, 1)] = sin(2 * PI * f * time_counter) * sin(PI * (shift_i * p->spatial_step) / aprime); // i = 0 pour face x =0 // -aprime??
            Ex[kEx(p, i, j, 1)] = 0;
            Hz[kHz(p, i, j, 1)] = 0;
            Hx[kHx(p, i, j, 1)] = -(1.0 / Z_te) * sin(2 * PI * f * time_counter) * sin(PI * (shift_i * p->spatial_step) / aprime);
        }
}

/** Joins the fields of each MPI process into the joined_field parameter
 * Parameters:
 *  joined_fields: The container for the joined fields
 *  p:             The parameters of the simulation
 * Warning:
 *  This function needs that each process sends its results. Risk of deadlock!
*/
void join_fields(Fields *join_fields, Parameters *p, Fields *pFields)
{
    //printf("rank in fct: %d \n", p->rank);
    for (int i = 1; i < p->ranks; ++i)
    {
        size_t k_offset = i * p->k_layers;
        MPI_Recv(&join_fields->Ex[kEx(p, 0, 0, k_offset)], sizeof_xy_plane(p, join_fields,join_fields->Ex)* (p->k_layers + p->ranks-1), MPI_DOUBLE, i, EX_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&join_fields->Ey[kEy(p, 0, 0, k_offset)], sizeof_xy_plane(p, join_fields,join_fields->Ey)* (p->k_layers + p->ranks-1), MPI_DOUBLE, i, EY_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&join_fields->Ez[kEz(p, 0, 0, k_offset)], sizeof_xy_plane(p, join_fields,join_fields->Ez)* (p->k_layers + p->ranks-1), MPI_DOUBLE, i, EZ_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&join_fields->Hx[kHx(p, 0, 0, k_offset)], sizeof_xy_plane(p, join_fields,join_fields->Hx)* (p->k_layers + p->ranks-1), MPI_DOUBLE, i, HX_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&join_fields->Hy[kHy(p, 0, 0, k_offset)], sizeof_xy_plane(p, join_fields,join_fields->Hy)* (p->k_layers + p->ranks-1), MPI_DOUBLE, i, HY_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&join_fields->Hz[kHz(p, 0, 0, k_offset)], sizeof_xy_plane(p, join_fields,join_fields->Hz)* (p->k_layers + p->ranks-1), MPI_DOUBLE, i, HZ_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i=0; i<p->maxi; i++)
        for(int j=0; j<p->maxj; j++)
            for (int k=1; k<p->k_layers+1; k++)
            {
                //printf("rank: %d \n", p->rank);
                join_fields->Ey[kEy(p, i, j, k-1)] = pFields->Ey[kEy(p, i, j, k)];
                join_fields->Ex[kEx(p, i, j, k-1)] = pFields->Ex[kEx(p, i, j, k)];
                join_fields->Ez[kEz(p, i, j, k-1)] = pFields->Ez[kEz(p, i, j, k)];
                join_fields->Hx[kHx(p, i, j, k-1)] = pFields->Hx[kHx(p, i, j, k)];
                join_fields->Hy[kHy(p, i, j, k-1)] = pFields->Hy[kHy(p, i, j, k)];
                join_fields->Hz[kHz(p, i, j, k-1)] = pFields->Hz[kHz(p, i, j, k)];
            }
}

/** Sends each fields to main thread to be joined
 * Parameters:
 *  pFields: The simulated fields
 *  p: The parameters of the simulation
 * Warning:
 *  This function needs that the main process receives the result. Risk of deadlock!
*/
void send_fields_to_main(Fields *pFields, Parameters *p)
{
    MPI_Send(&pFields->Ex[kEx(p, 0, 0, 1)], sizeof_xy_plane(p, pFields, pFields->Ex) * (p->k_layers+1), MPI_DOUBLE, 0, EX_TAG_TO_MAIN, MPI_COMM_WORLD);
    MPI_Send(&pFields->Ey[kEy(p, 0, 0, 1)], sizeof_xy_plane(p, pFields, pFields->Ey) * (p->k_layers+1), MPI_DOUBLE, 0, EY_TAG_TO_MAIN, MPI_COMM_WORLD);
    MPI_Send(&pFields->Ez[kEz(p, 0, 0, 1)], sizeof_xy_plane(p, pFields, pFields->Ez) * (p->k_layers+1), MPI_DOUBLE, 0, EZ_TAG_TO_MAIN, MPI_COMM_WORLD);
    MPI_Send(&pFields->Hx[kHx(p, 0, 0, 1)], sizeof_xy_plane(p, pFields, pFields->Hx) * (p->k_layers+1), MPI_DOUBLE, 0, HX_TAG_TO_MAIN, MPI_COMM_WORLD);
    MPI_Send(&pFields->Hy[kHy(p, 0, 0, 1)], sizeof_xy_plane(p, pFields, pFields->Hy) * (p->k_layers+1), MPI_DOUBLE, 0, HY_TAG_TO_MAIN, MPI_COMM_WORLD);
    MPI_Send(&pFields->Hz[kHz(p, 0, 0, 1)], sizeof_xy_plane(p, pFields, pFields->Hz) * (p->k_layers+1), MPI_DOUBLE, 0, HZ_TAG_TO_MAIN, MPI_COMM_WORLD);
}

/** MPI: Exhanges E field between ranks
 * Parameters:
 *  p: The parameters of the simulation
 *  f: The simulated fields
*/
void exchange_E_field(Parameters *p, Fields *f)
{
    if (p->rank % 2 == 0)
    {
        MPI_Send(&f->Ex[kEx(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->lower_cpu, EX_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(&f->Ex[kEx(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->upper_cpu, EX_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ex[kEx(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->upper_cpu, EX_TAG_TO_UP, MPI_COMM_WORLD);
        MPI_Recv(f->Ex, sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->lower_cpu, EX_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Send(&f->Ey[kEy(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->lower_cpu, EY_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(&f->Ey[kEy(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->upper_cpu, EY_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ey[kEy(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->upper_cpu, EY_TAG_TO_UP, MPI_COMM_WORLD);
        MPI_Recv(f->Ey, sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->lower_cpu, EY_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Send(&f->Ez[kEz(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->lower_cpu, EZ_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(&f->Ez[kEz(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->upper_cpu, EZ_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ez[kEz(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->upper_cpu, EZ_TAG_TO_UP, MPI_COMM_WORLD);
        MPI_Recv(f->Ez, sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->lower_cpu, EZ_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Recv(&f->Ex[kEx(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->upper_cpu, EX_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ex[kEx(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->lower_cpu, EX_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(f->Ex, sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->lower_cpu, EX_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ex[kEx(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Ex), MPI_DOUBLE, p->upper_cpu, EX_TAG_TO_UP, MPI_COMM_WORLD);

        MPI_Recv(&f->Ey[kEy(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->upper_cpu, EY_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ey[kEy(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->lower_cpu, EY_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(f->Ey, sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->lower_cpu, EY_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ey[kEy(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Ey), MPI_DOUBLE, p->upper_cpu, EY_TAG_TO_UP, MPI_COMM_WORLD);

        MPI_Recv(&f->Ez[kEz(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->upper_cpu, EZ_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ez[kEz(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->lower_cpu, EZ_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(f->Ez, sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->lower_cpu, EZ_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Ez[kEz(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Ez), MPI_DOUBLE, p->upper_cpu, EZ_TAG_TO_UP, MPI_COMM_WORLD);
    }
}

/** Echanges the H field between ranks (MPI)
 * Parameters:
 *  p: The parameters of the simulation
 *  f: The simulated fields
*/
void exchange_H_field(Parameters *p, Fields *f)
{
    if (p->rank % 2 == 0)
    {
        MPI_Send(&f->Hx[kHx(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->lower_cpu, HX_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(&f->Hx[kHx(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->upper_cpu, HX_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hx[kHx(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->upper_cpu, HX_TAG_TO_UP, MPI_COMM_WORLD);
        MPI_Recv(f->Hx, sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->lower_cpu, HX_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Send(&f->Hy[kHy(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->lower_cpu, HY_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(&f->Hy[kHy(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->upper_cpu, HY_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hy[kHy(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->upper_cpu, HY_TAG_TO_UP, MPI_COMM_WORLD);
        MPI_Recv(f->Hy, sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->lower_cpu, HY_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Send(&f->Hz[kHz(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->lower_cpu, HZ_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(&f->Hz[kHz(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->upper_cpu, HZ_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hz[kHz(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->upper_cpu, HZ_TAG_TO_UP, MPI_COMM_WORLD);
        MPI_Recv(f->Hz, sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->lower_cpu, HZ_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Recv(&f->Hx[kHx(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->upper_cpu, HX_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hx[kHx(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->lower_cpu, HX_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(f->Hx, sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->lower_cpu, HX_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hx[kHx(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Hx), MPI_DOUBLE, p->upper_cpu, HX_TAG_TO_UP, MPI_COMM_WORLD);

        MPI_Recv(&f->Hy[kHy(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->upper_cpu, HY_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hy[kHy(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->lower_cpu, HY_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(f->Hy, sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->lower_cpu, HY_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hy[kHy(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Hy), MPI_DOUBLE, p->upper_cpu, HY_TAG_TO_UP, MPI_COMM_WORLD);

        MPI_Recv(&f->Hz[kHz(p, 0, 0, p->k_layers + 1)], sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->upper_cpu, HZ_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hz[kHz(p, 0, 0, 1)], sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->lower_cpu, HZ_TAG_TO_DOWN, MPI_COMM_WORLD);
        MPI_Recv(f->Hz, sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->lower_cpu, HZ_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&f->Hz[kHz(p, 0, 0, p->k_layers)], sizeof_xy_plane(p, f, f->Hz), MPI_DOUBLE, p->upper_cpu, HZ_TAG_TO_UP, MPI_COMM_WORLD);
    }
}

/** Propagate the Electrical and Magnetic field using FDTD algorithm
 * Parameters:
 *  pFields: The fields
 *  pValidationFields: The ValidationFields
 *  pParams:    The parameters of the simulation
 *  pOven: The oven properties
*/
static
void propagate_fields(Fields *pFields, Fields *pValidationFields, Parameters *pParams, Oven *pOven)
{
    double timer;
    double total_energy;
    int iteration = 1;

    static Fields *joined_fields;
    //printf("rank before: %d \n", pParams->rank);
    if (pParams->rank == 0)
    {
        //printf("rank: %d \n", pParams->rank);
        joined_fields = initialize_fields(pParams);
        join_fields(joined_fields, pParams, pFields);
        total_energy = calculate_E_energy(joined_fields, pParams) + calculate_H_energy(joined_fields, pParams);

        if (pParams->mode == VALIDATION_MODE)
            update_validation_fields_then_subfdtd(pParams, joined_fields, pValidationFields, 0.0);
    }
    else{
        //printf("else: rank: %d \n", pParams->rank);
        send_fields_to_main(pFields, pParams);

    }

    for (timer = 0; timer <= pParams->simulation_time; timer += pParams->time_step, iteration++)
    {
        if (pParams->rank == 0 && joined_fields != NULL && iteration % pParams->sampling_rate == 0)
            write_silo(joined_fields, pValidationFields, pParams, pOven, iteration);

        if (pParams->mode == COMPUTATION_MODE && pParams->rank==0)
            set_source(pParams, pFields, timer);

        exchange_E_field(pParams, pFields);

        update_H_field(pParams, pFields);

        exchange_H_field(pParams, pFields);

        if (pParams->mode == COMPUTATION_MODE && pParams->rank==0)
            set_source(pParams, pFields, timer);

        update_E_field(pParams, pFields);

        if (pParams->rank == 0 && joined_fields != NULL)
            join_fields(joined_fields, pParams, pFields);
        else
            send_fields_to_main(pFields, pParams);

        if (pParams->rank == 0 && pParams->mode == VALIDATION_MODE)
        {
            update_validation_fields_then_subfdtd(pParams, joined_fields, pValidationFields, timer);
            //printf("Electrical energy: %0.20f \n", calculate_E_energy(joined_fields, pParams));
            //printf("Magnetic energy: %0.20f \n", calculate_H_energy(joined_fields, pParams));
            //printf("Tot energy: %0.20f \n", calculate_E_energy(joined_fields, pParams) + calculate_H_energy(joined_fields, pParams));
            //printf("Theoretical energy: %0.20f \n", (EPSILON * pParams->length * pParams->width * pParams->height) / 8.);
            //assert((calculate_E_energy(joined_fields, pParams) + calculate_H_energy(joined_fields, pParams) - total_energy) <= 0.000001);
        }
    }
}

/**
------------------------------------------------------
---------------------------------------- Main function
------------------------------------------------------
*/
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        printf("Welcome into our microwave oven eletrico-magnetic field simulator! \n");

    if (argc != 2)
        fail(NULL, "This program needs 1 argument: the parameters file (.txt). Eg.: ./microwave param.txt");

    printf("Process %d: Loading the parameters of the simulation...\n", rank);
    Parameters *pParameters = load_parameters(argv[1]);

    if (pParameters->time_step > pParameters->simulation_time)
        fail(pParameters->ls, "The time step must be lower than the simulation time!");

    printf("Process %d: Creating mesh\n", rank);
    Oven *pOven = compute_oven(pParameters);

    printf("Process %d: Initializing fields\n", rank);
    Fields *pFields = initialize_cpu_fields(pParameters);

    Fields *pValidationFields;
    if (pParameters->mode == VALIDATION_MODE)
    {
        if (rank == 0)
        {
            pValidationFields = initialize_fields(pParameters);
            printf("Main process: Validation mode activated. \n");
            // Free what's not needed for validation.
            Free(&(pParameters->ls),pValidationFields->Ex);
            Free(&(pParameters->ls),pValidationFields->Ez);
            Free(&(pParameters->ls),pValidationFields->Hy);
        }
        printf("Process %d: Setting initial conditions\n", rank);
        set_initial_conditions(pFields->Ey, pParameters);
    }

    printf("Process %d: Launching simulation\n", rank);
    propagate_fields(pFields, pValidationFields, pParameters, pOven);
    printf("Process %d: Freeing memory...\n", rank);

    freeAll(pParameters->ls);

    if (rank == 0)
        printf("Simulation complete!\n");

    MPI_Finalize();
    return 0;
}