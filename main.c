/** 
 * Title: Electromagnetic wave propagation in a microwave
 * Authors: Amaury Baret, Ionut Finta
 * Date: December 2021
 * Organization: University of Liège, Belgium
 * Course: INFO0939 - High performance scientific computing
 * Professors: Geuzaine Christophe; Hiard Samuel, Leduc Guy
 * Description:
 *    This program simulates the propagation of an electromagnetic
 *    wave in a microwave oven using the FDTD scheme. The parallel
 *    branch is for parallel execution of the simulation.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <silo.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <errno.h>
#include <stdint.h>

// --- Silo constants
#define DB_FILENAME "r/result%04d.silo"
#define DB_MESHNAME "mesh"

// --- Mathematical/Physical constants, precise enough for double computation
#define MU 1.25663706143591729538505735331180115367886775975E-6
#define EPSILON 8.854E-12
#define PI 3.14159265358979323846264338327950288419716939937510582097494
#define CELERITY 299792458.0

// --- Execution modes for this program
// VALIDATION:  There is no source but an initial condition given by the equation in the statement
#define VALIDATION_MODE 0

// COMPUTATION: The source is set and continuously radiates the simulation box
#define COMPUTATION_MODE 1

// Preventing Nx/Ny/Nz to be bigger than maximum of size_t
// Solution adapted from https://stackoverflow.com/a/53988522
// `double` value 1 past SIZE_MAX:
#define SIZE_MAX_P1_DOUBLE ((SIZE_MAX / 2 + 1) * 2.0f)

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------- Data Structures & Type definitions
//--------------------------------------------------------------------------------------------------
/// @brief TAGS used for diverse OpenMPI communications
static const enum TAGS { HX_TAG_TO_UP,
                         HY_TAG_TO_UP,
                         EX_TAG_TO_DOWN,
                         EY_TAG_TO_DOWN,
                         EX_TAG_TO_MAIN,
                         EY_TAG_TO_MAIN,
                         EZ_TAG_TO_MAIN,
                         HX_TAG_TO_MAIN,
                         HY_TAG_TO_MAIN,
                         HZ_TAG_TO_MAIN,
} tags;

/**
 * @brief  An approach to manage memory allocations in a chained list
 * @note    This structure allow us to store each allocated object
 *          into a chained list with a LiFo strategy so when we
 *          free up all memory we take care of the inner most objects
*/
typedef struct chainedAllocated
{
    struct chainedAllocated *previous; // The previous node of the chained list
    void *ptr;                         // The current allocated object
} ChainedAllocated;

/// @brief A structure that reassembles all the fields components
typedef struct fields
{
    double *Ex; // The arrays of the x components of the electric field
    double *Ey; // The arrays of the y components of the electric field
    double *Ez; // The arrays of the z components of the electric field
    double *Hx; // The arrays of the x components of the magnetic field
    double *Hy; // The arrays of the y components of the magnetic field
    double *Hz; // The arrays of the z components of the magnetic field

} Fields;

/// @brief Parameters of the simulation
typedef struct parameters
{
    double width;  // a, x in figure (y, in yee paper cs)
    double height; // b, y in figure (z, in yee paper cs)
    double length; // d, z in figure (x, in yee paper cs)

    size_t Nx; // Number of grid subdivisions (x dimension)
    size_t Ny; // Number of grid subdivisions (y dimension)
    size_t Nz; // Number of grid subdivisions (z dimension)

    double spatial_step; // delta x = delta y = delta z
    double time_step;    // delta t

    double simulation_time;     // interval of time simulated (in seconds)
    unsigned int sampling_rate; // rate at which data is printed to file (in #steps)
    unsigned int mode;          // 0 for validation mode, 1 for computation
    unsigned int dump_csv;      // 0 for not dumping any csv, 1 to dump the csv file or errors.

    // Mesh directly derived from parameters above and useless to recompute at each timestep
    // (only for process with rank == 0)
    int *dims;       // Array of the sizes of each dimension for the mesh {Nx+1, Ny+1, Nz+1}
    int *vdims;      // Array of the sizes of each dimension for variables {Nx, Ny, mak}
    double **coords; // Coordinates of each mesh point (grid)

    Fields *mean;              // Mean fields
    Fields *validation_fields; // Raw validation fields

    size_t *start_k_of_rank; // The starting k of each process

    // Parallelization stuff
    int rank;             // The rank of the current process
    int ranks;            // The number of ranks, aka. MPI_Comm_size
    size_t startk;        // The index of the first XY plane treated by the current process
    size_t k_layers;      // The number of XY planes treated by current process
    int lower_cpu;        // The rank of the process working on the plane below (Z-1)
    int upper_cpu;        // The rank of the CPU working on the plane above (Z+1)
    ChainedAllocated *ls; // The chained list of allocated objects by this process
} Parameters;

//--------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------ Function's Specifications
//--------------------------------------------------------------------- Signatures & Implementations
//--------------------------------------------------------------------------------------------------
/**
 * @brief Free any allocated object with the Malloc function (see below)
 * @param ls The list of allocated objects
 * @note This function free's the memory allocated for all
 *       the objects allocated with the next Malloc function
 *       in the reverse way of their allocation (LiFo) in order
 *       to deal with nested structures.
*/
void *freeAll(ChainedAllocated *ls)
{
    while (ls && ls->ptr != ls)
    {
        free(ls->ptr);
        ChainedAllocated *previous = ls->previous;
        free(ls);
        ls = previous;
    }
    free(ls);
}

/**
 * @brief Free the memory and throws an error, then exits with EXIT_FAILURE.
 * @param ls   The chained list of allocated objects
 * @param msg  The message to throw before exit
*/
void fail(ChainedAllocated *ls, const char *msg)
{
    perror(msg);
    freeAll(ls);

    if (MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE) != MPI_SUCCESS)
        perror("MPI_Abort failed. Use CTRL+C or pkill to kill the MPI processes. \n");

    if (MPI_Finalize() != MPI_SUCCESS)
        perror("Something happend with the MPI_Finalize function...\n");
    exit(EXIT_FAILURE);
}

/**
 * @brief Critical allocation (malloc or fail)
 * @param pLs A pointer to the pointer of the list of allocated objects.
 * @note In addition to critically check if the malloc properly worked,
 *       this function stores the reference to the new allocated object
 *       in the chained list which is then used to free the memory before
 *       exit. A pointer to the pointer is used as the pointer will be
 *       modified.
*/
void *Malloc(ChainedAllocated **pLs, size_t size)
{
    // Very first allocation: The first ptr references the node itself
    ChainedAllocated *ls = *pLs;
    if (ls == NULL)
    {
        ls = malloc(sizeof(ChainedAllocated));
        ls->ptr = ls;
        ls->previous = NULL;
    }

    // Creates next node and points pLs to it
    ChainedAllocated *successor = malloc(sizeof(ChainedAllocated));
    successor->previous = ls;
    ls = successor;
    *pLs = ls;

    // Allocate the demanded size or fail
    void *ptr = malloc(size);

    ls->ptr = ptr;

    if (!ptr)
        fail(ls, "CRITICAL ERROR: Could not allocate enough memory!\n");

    return ptr;
}

/**
 * @brief Utility to critically allocate array of doubles and initialize all fields to 0.0
 * @param pLs The pointer to a pointer of the list of allocated objects.
 * @param len The length of the array 
*/
double *Malloc_Double(ChainedAllocated **pLs, size_t len)
{
    double *ptr = Malloc(pLs, sizeof(double) * len);
    for (--len; 0 < len; --len)
        ptr[len] = 0.0;
    return ptr;
}

/**
 * @brief Debug utility to print the chained list of pointers in heap
 * @param ls The pointer to the chained list of pointers
 * @note It's better to never need that!
*/
static void print_heap(ChainedAllocated *ls)
{
    printf("==== Heap addresses dump ==== \n");
    for (; ls; ls = ls->previous)
        printf("Current: %p || ptr: %p || Previous: %p\n", ls, ls->ptr, ls->previous);
    printf("==== Heap addresses dump END ==== \n\n");
}

/**
 * @brief Frees the memory and removes the entry from the chained list.
 * @pre ptr was allocated with the Malloc function above.
 * @param ls   A pointer to the list containing the allocated objects
 * @param ptr  The pointer to the object to free
*/
void Free(ChainedAllocated **pLs, void *ptr)
{
    // If the pointer is here, free it
    ChainedAllocated *current = *pLs;
    if (current->ptr == ptr)
    {
        *pLs = current->previous;
        free(current->ptr);
        free(current);
        return;
    }

    // Ensure termination, if you get this error, maybe you didn't allocate the object
    // with the Malloc function provided above, or you lost the initial pointer of
    // the very first node. You can use printHeap to debug it before calling Free.
    if (current->previous == NULL)
    {
        fprintf(stderr, "Trying to Free ptr %p which is not found in heap list\n", ptr);
        return;
    }

    // Otherwise, recurse beginning from the previous node
    Free(&current->previous, ptr);
}

/**
 * @brief Utility to round double to size_t with bounds checking. (for Nx/Ny/Nz)
 * 
 * @param dbl The double value
 * @return size_t The rounded double to size_t
 */
size_t double_rounded_to_size_t(double dbl)
{
    // In range?
    if (dbl >= -0.5 && dbl - SIZE_MAX_P1_DOUBLE < -0.5)
    {
        size_t sz = (size_t)dbl;
        double frac = dbl - (double)sz;
        if (frac > 0.5 || (frac >= 0.5 && sz % 2))
        {
            sz++;
        }
        return sz;
    }
    if (dbl >= 0.0)
    {
        errno = ERANGE;
        return SIZE_MAX; // dbl is too great
    }
    if (dbl < 0.0)
    {
        errno = ERANGE;
        return 0; // dbl is too negative
    }
    errno = EDOM;
    return 0; // dbl is not-a-number
}

/**
 * @brief Loads the parameters of the simulation into the system memory.
 * @param filename The file containing the parameters properties (.txt)
 * @param rank  The rank of current process
 * @return A pointer to the Parameters struct loaded in system memory
 * @note In parallel mode, this also sets the state of the process.
*/
Parameters *load_parameters(const char *filename, int rank)
{
    FILE *fParams = fopen(filename, "r");
    if (!fParams)
        fail(NULL, "Unable to open parameters file!\n");

    ChainedAllocated *ls = NULL;
    Parameters *pParameters = Malloc(&ls, sizeof(Parameters));

    if (fscanf(fParams, "%lf\n", &pParameters->length) != 1 || pParameters->length <= 0.0)
        fail(ls, "Bad parameters file! Cannot parse length (must be strictly positive). Abort.\n");
    if (fscanf(fParams, "%lf\n", &pParameters->width) != 1 || pParameters->width <= 0.0)
        fail(ls, "Bad parameters file! Cannot parse width (must be strictly positive). Abort.\n");
    if (fscanf(fParams, "%lf\n", &pParameters->height) != 1 || pParameters->height <= 0.0)
        fail(ls, "Bad parameters file! Cannot parse height (must be strictly positive). Abort.\n");
    if (fscanf(fParams, "%lf\n", &pParameters->spatial_step) != 1 || pParameters->spatial_step <= 0.0)
        fail(ls, "Bad parameters file! Cannot parse spatial delta (must be strictly positive). Abort.\n");
    if (fscanf(fParams, "%lf\n", &pParameters->time_step) != 1 || pParameters->time_step <= 0.0)
        fail(ls, "Bad parameters file! Cannot parse time delta (must be strictly positive). Abort.\n");
    if (fscanf(fParams, "%lf\n", &pParameters->simulation_time) != 1 || pParameters->simulation_time <= 0.0)
        fail(ls, "Bad parameters file! Cannot parse simulation time (must be strictly positive). Abort.\n");
    if (fscanf(fParams, "%u\n", &pParameters->sampling_rate) != 1)
        fail(ls, "Bad parameters file! Cannot parse sampling rate (unsigned integer needed). Abort.\n");
    if (fscanf(fParams, "%u\n", &pParameters->mode) != 1 || (pParameters->mode != 0 && pParameters->mode != 1))
        fail(ls, "Bad parameters file! Cannot parse execution mode can be either 0 or 1. Abort.\n");

    fclose(fParams);

    pParameters->Nx = double_rounded_to_size_t(pParameters->length / pParameters->spatial_step);
    pParameters->Ny = double_rounded_to_size_t(pParameters->width / pParameters->spatial_step);
    pParameters->Nz = double_rounded_to_size_t(pParameters->height / pParameters->spatial_step);
    if (pParameters->Nx == 0 || pParameters->Ny == 0 || pParameters->Nz == 0 || pParameters->Nx >= SIZE_MAX - 2 || pParameters->Ny >= SIZE_MAX - 2 || pParameters->Nz >= SIZE_MAX - 2)
    {
        perror("Something went wrong while computing Nx/Ny/Nz \n ERRNO: ");
        perror(strerror(errno));
        fail(ls, "\nPlease check your parameters file and try again.\n");
    }
    pParameters->dump_csv = 0;

    // Parallelization:
    pParameters->rank = rank;
    if (MPI_Comm_size(MPI_COMM_WORLD, &pParameters->ranks) != MPI_SUCCESS)
        fail(ls, "Could not get MPI communicator size. Abort.\n");

    // Solution to equally divide the workload between processes
    // Found on Stackoverflow: https://stackoverflow.com/a/26554699
    size_t count = pParameters->Nz / pParameters->ranks;
    size_t remainder = pParameters->Nz % pParameters->ranks;
    size_t stop;

    if (pParameters->rank < remainder)
    {
        // The first 'remainder' ranks get 'count + 1' tasks each
        pParameters->startk = pParameters->rank * (count + 1);
        stop = pParameters->startk + count;
        stop++;
    }
    else
    {
        // The remaining 'size - remainder' ranks get 'count' task each
        pParameters->startk = pParameters->rank * count + remainder;
        stop = pParameters->startk + (count - 1);
        stop++;
    }
    pParameters->k_layers = stop - pParameters->startk;

    // Gather all the start_k in an array of process 0
    if (pParameters->rank == 0)
        pParameters->start_k_of_rank = Malloc(&ls, pParameters->ranks * sizeof(size_t));

    if (MPI_Gather(&pParameters->startk, 1, MPI_UNSIGNED_LONG, pParameters->start_k_of_rank, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
        fail(ls, "Could not gather the starting index of each process. Abort.\n");

    pParameters->lower_cpu = pParameters->rank > 0 ? pParameters->rank - 1 : MPI_PROC_NULL;
    pParameters->upper_cpu = pParameters->rank < pParameters->ranks - 1 ? pParameters->rank + 1 : MPI_PROC_NULL;
    pParameters->ls = ls;

    return pParameters;
}

/** 
 * @brief Compute the oven properties and persist it into params
 * @param params The parameters of the simulation
*/
void *compute_oven(Parameters *params)
{
    params->dims = Malloc(&params->ls, sizeof(size_t) * 3);
    params->vdims = Malloc(&params->ls, sizeof(size_t) * 3);
    params->coords = Malloc(&params->ls, sizeof(double *) * 3);
    params->dims[0] = params->Nx + 1;
    params->dims[1] = params->Ny + 1;
    params->dims[2] = params->Nz + 1;
    params->vdims[0] = params->Nx;
    params->vdims[1] = params->Ny;
    params->vdims[2] = params->Nz;

    double *x = Malloc(&params->ls, (params->Nx + 1) * sizeof(double));
    double *y = Malloc(&params->ls, (params->Ny + 1) * sizeof(double));
    double *z = Malloc(&params->ls, (params->Nz + 1) * sizeof(double));

    double dx = params->spatial_step;
    size_t i;
    for (i = 0; i < params->Nx + 1; ++i)
        x[i] = i * dx;

    for (i = 0; i < params->Ny + 1; ++i)
        y[i] = i * dx;

    for (i = 0; i < params->Nz + 1; ++i)
        z[i] = i * dx;

    double *cords[] = {x, y, z};

    params->coords[0] = x;
    params->coords[1] = y;
    params->coords[2] = z;
}

/**
 * @brief Gives the size of the simulation XY plane depending on the field component
 * @param p      The parameters of the simulation
 * @param fields The fields
 * @param field  The field for which you want the size of XY plane
 * @return The size of the XY plane
**/
size_t sizeof_XY(Parameters *p, Fields *fields, double *field)
{
    if (field == fields->Ex)
        return p->Nx * (p->Ny + 1);
    if (field == fields->Ey)
        return (p->Nx + 1) * p->Ny;
    if (field == fields->Ez)
        return (p->Nx + 1) * (p->Ny + 1);
    if (field == fields->Hx)
        return (p->Nx + 1) * p->Ny;
    if (field == fields->Hy)
        return p->Nx * (p->Ny + 1);

    assert(field == fields->Hz);
    return p->Nx * p->Ny;
}

/**
 * @brief Allocates and initialize to 0.0 all the components of each simulation field
 * @param params The parameters of the simulation
 * @return A pointer to the allocated Field struct and all its fields
*/
static Fields *initialize_fields(Parameters *params)
{
    Fields *pFields = Malloc(&params->ls, sizeof(Fields));

    // Ex
    size_t len = params->Nx * (params->Ny + 1) * (params->Nz + 1);
    pFields->Ex = Malloc_Double(&params->ls, len);

    // Ey
    len = (params->Nx + 1) * params->Ny * (params->Nz + 1);
    pFields->Ey = Malloc_Double(&params->ls, len);

    // Ez
    len = (params->Nx + 1) * (params->Ny + 1) * params->Nz;
    pFields->Ez = Malloc_Double(&params->ls, len);

    // Hx
    len = (params->Nx + 1) * params->Ny * params->Nz;
    pFields->Hx = Malloc_Double(&params->ls, len);

    // Hy
    len = params->Nx * (params->Ny + 1) * params->Nz;
    pFields->Hy = Malloc_Double(&params->ls, len);

    // Hz
    len = params->Nx * params->Ny * (params->Nz + 1);
    pFields->Hz = Malloc_Double(&params->ls, len);

    return pFields;
}

/**
 * @brief Allocates and initialize to 0.0 all the components of each field for mean/validation.
 * @param params The parameters of the simulation
 * @return A pointer to the allocated Field struct and all its fields
*/
static Fields *initialize_mean_fields(Parameters *params)
{
    Fields *f = Malloc(&params->ls, sizeof(Fields));

    size_t len = params->Nx * params->Ny * params->Nz + 1;
    f->Ex = Malloc_Double(&params->ls, len);
    f->Ey = Malloc_Double(&params->ls, len);
    f->Ez = Malloc_Double(&params->ls, len);
    f->Hx = Malloc_Double(&params->ls, len);
    f->Hy = Malloc_Double(&params->ls, len);
    f->Hz = Malloc_Double(&params->ls, len);

    return f;
}

/**
 * @brief Allocates and initialize to 0.0 all the components of each field for this process [MPI]
 * @param params The parameters of the simulation
*/
Fields *initialize_cpu_fields(Parameters *params)
{
    Fields *pFields = Malloc(&params->ls, sizeof(Fields));

    // Ex
    size_t len = params->Nx * (params->Ny + 1) * (params->k_layers + 2);
    pFields->Ex = Malloc_Double(&params->ls, len);

    // Ey
    len = (params->Nx + 1) * params->Ny * (params->k_layers + 2);
    pFields->Ey = Malloc_Double(&params->ls, len);

    // Ez
    len = (params->Nx + 1) * (params->Ny + 1) * (params->k_layers + 2);
    pFields->Ez = Malloc_Double(&params->ls, len);

    // Hx
    len = (params->Nx + 1) * params->Ny * (params->k_layers + 2);
    pFields->Hx = Malloc_Double(&params->ls, len);

    // Hy
    len = params->Nx * (params->Ny + 1) * (params->k_layers + 2);
    pFields->Hy = Malloc_Double(&params->ls, len);

    // Hz
    len = params->Nx * params->Ny * (params->k_layers + 2);
    pFields->Hz = Malloc_Double(&params->ls, len);

    return pFields;
}

/**
 * @brief Fast shortcut to get the index of a field at i, j and k
 * @param params The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @param mi_mj  The additional sizes of dimensions X and Y.
 * @return The index in a 1D array
*/
static inline size_t idx(Parameters *params, size_t i, size_t j, size_t k, size_t mi, size_t mj)
{
    return i + j * (params->Nx + mi) + k * (params->Nx + mi) * (params->Ny + mj);
}

/**
 * @brief Fast shortcut to get the index of Ex field at i, j and k
 * @param p The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @return The index in a 1D array
*/
static inline size_t kEx(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * p->Nx + k * p->Nx * (p->Ny + 1);
}

/**
 * @brief Fast shortcut to get the index of Ey field at i, j and k
 * @param p The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @return The index in a 1D array
*/
static inline size_t kEy(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * (p->Nx + 1) + k * (p->Nx + 1) * p->Ny;
}

/**
 * @brief Fast shortcut to get the index of Ez field at i, j and k
 * @param p The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @return The index in a 1D array
*/
static inline size_t kEz(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * (p->Nx + 1) + k * (p->Nx + 1) * (p->Ny + 1);
}

/**
 * @brief Fast shortcut to get the index of Hx field at i, j and k
 * @param p The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @return The index in a 1D array
*/
static inline size_t kHx(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * (p->Nx + 1) + k * (p->Nx + 1) * p->Ny;
}

/**
 * @brief Fast shortcut to get the index of Hy field at i, j and k
 * @param p The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @return The index in a 1D array
*/
static inline size_t kHy(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + j * p->Nx + k * p->Nx * (p->Ny + 1);
}

/**
 * @brief Fast shortcut to get the index of Hz field at i, j and k
 * @param p The parameters of the simulation
 * @param i_j_k  The coordinates of the wanted field
 * @return The index in a 1D array
*/
static inline size_t kHz(Parameters *p, size_t i, size_t j, size_t k)
{
    return i + (j + k * p->Ny) * p->Nx;
}

/**
 * @brief Sets the initial field as stated in Question 3.a.
 * @param Ey The y component of the Energy fields
 * @param p  The parameters of the simulation
*/
void set_initial_conditions(double *Ey, Parameters *p)
{
    size_t i, j, k;
    for (k = 1; k < p->k_layers + 2; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx + 1; ++i)
                Ey[kEy(p, i, j, k)] = sin(PI * (p->startk + k - 1) * p->spatial_step / p->height) *
                                      sin(PI * i * p->spatial_step / p->length);
}

/** 
 * @brief Updates the H field
 * @param p      The parameters of the simulation
 * @param fields All the fields
*/
void update_H_field(Parameters *p, Fields *fields)
{
    // Shortcuts to avoid pointers exploration in the loops.
    double *Ex = fields->Ex;
    double *Ey = fields->Ey;
    double *Ez = fields->Ez;
    double *Hx = fields->Hx;
    double *Hy = fields->Hy;
    double *Hz = fields->Hz;

    double factor = p->time_step / (MU * p->spatial_step);

    size_t i, j, k;

    for (k = 1; k < p->k_layers + 1; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx + 1; ++i)
                Hx[kHx(p, i, j, k)] += factor * ((Ey[kEy(p, i, j, k + 1)] - Ey[kEy(p, i, j, k)]) -
                                                 (Ez[kEz(p, i, j + 1, k)] - Ez[kEz(p, i, j, k)]));

    for (k = 1; k < p->k_layers + 1; ++k)
        for (j = 0; j < p->Ny + 1; ++j)
            for (i = 0; i < p->Nx; ++i)
                Hy[kHy(p, i, j, k)] += factor * ((Ez[kEz(p, i + 1, j, k)] - Ez[kEz(p, i, j, k)]) -
                                                 (Ex[kEx(p, i, j, k + 1)] - Ex[kEx(p, i, j, k)]));

    size_t ofst = p->rank == p->ranks - 1 ? 2 : 1;
    ofst += p->k_layers;
    for (k = 1; k < ofst; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx; ++i)
                Hz[kHz(p, i, j, k)] += factor * ((Ex[kEx(p, i, j + 1, k)] - Ex[kEx(p, i, j, k)]) -
                                                 (Ey[kEy(p, i + 1, j, k)] - Ey[kEy(p, i, j, k)]));
}

/** 
 * @brief Updates the E field
 * @param p      The parameters of the simulation
 * @param fields All the fields
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

    size_t startk = p->rank == 0 ? 2 : 1;
    for (k = startk; k < p->k_layers + 1; ++k)
        for (j = 1; j < p->Ny; ++j)
            for (i = 0; i < p->Nx; ++i)
                Ex[kEx(p, i, j, k)] += factor * ((Hz[kHz(p, i, j, k)] - Hz[kHz(p, i, j - 1, k)]) -
                                                 (Hy[kHy(p, i, j, k)] - Hy[kHy(p, i, j, k - 1)]));

    for (k = startk; k < p->k_layers + 1; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 1; i < p->Nx; ++i)
                Ey[kEy(p, i, j, k)] += factor * ((Hx[kHx(p, i, j, k)] - Hx[kHx(p, i, j, k - 1)]) -
                                                 (Hz[kHz(p, i, j, k)] - Hz[kHz(p, i - 1, j, k)]));

    for (k = 1; k < p->k_layers + 1; ++k)
        for (j = 1; j < p->Ny; ++j)
            for (i = 1; i < p->Nx; ++i)
                Ez[kEz(p, i, j, k)] += factor * ((Hy[kHy(p, i, j, k)] - Hy[kHy(p, i - 1, j, k)]) -
                                                 (Hx[kHx(p, i, j, k)] - Hx[kHx(p, i, j - 1, k)]));
}

/**
 * @brief Computes the mean of an electrical field
 * @param p   The simulation parameters
 * @param Ef  The E field component in one direction
 * @param r   The result aggregated vector of size Nx * Ny * Nz
 * @param ofi [0 or 1] The offset in X (related to the space size)
 * @param ofj [0 or 1] The offset in Y (related to the space size)
 * @param ofk [0 or 1] The offset in Z (related to the space size)
*/
void aggregate_E_field(Parameters *p, double *Ef, double *r, size_t ofi, size_t ofj, size_t ofk)
{
    size_t i, j, k;
    size_t t = 0;
    for (k = 0; k < p->Nz; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx; ++i)
                r[t++] = .25 * (Ef[idx(p, i, j, k, ofi, ofj)] +
                                Ef[idx(p, i + ofi, j + ofj, k + ofk, ofi, ofj)] +
                                Ef[idx(p, i, j + ofj, k + ofk, ofi, ofj)] +
                                Ef[idx(p, i + ofi, j, k + ofk, ofi, ofj)]);
}

/** 
 * @brief Computes the mean of an magnetic field 
 * @param p   The simulation parameters
 * @param Hf  The H field component in one direction
 * @param r   The result aggregated vector of size (Nx, Ny, Nz)
 * @param ofi [0 or 1] The offset in X (related to the space size)
 * @param ofj [0 or 1] The offset in Y (related to the space size)
 * @param ofk [0 or 1] The offset in Z (related to the space size)
*/
void aggregate_H_field(Parameters *p, double *Hf, double *r, size_t ofi, size_t ofj, size_t ofk)
{
    size_t i, j, k;
    size_t t = 0;
    for (k = 0; k < p->Nz; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx; ++i)
                r[t++] = .5 * (Hf[idx(p, i, j, k, ofi, ofj)] +
                               Hf[idx(p, i + ofi, j + ofj, k + ofk, ofi, ofj)]);
}

/**
 * @brief Compute the mean of FDTD simulated fields into the mesh space
 * 
 * @param p Parameters of the simulation
 * @param f Simulated fields
 */
void mean_fields(Parameters *p, Fields *f)
{
    aggregate_E_field(p, f->Ex, p->mean->Ex, 0, 1, 1);
    aggregate_E_field(p, f->Ey, p->mean->Ey, 1, 0, 1);
    aggregate_E_field(p, f->Ez, p->mean->Ez, 1, 1, 0);
    aggregate_H_field(p, f->Hx, p->mean->Hx, 1, 0, 0);
    aggregate_H_field(p, f->Hy, p->mean->Hy, 0, 1, 0);
    aggregate_H_field(p, f->Hz, p->mean->Hz, 0, 0, 1);
}

/**
 * @brief Writes a silo file of the simulation in the given timestamp
 * @param pValidationFields The validation fields
 * @param pParams           The parameters of the simulation
 * @param iteration         The iteration count
 * @pre   pParams->mean contains the aggregated fields
*/
void write_silo(Fields *pValidationFields, Parameters *pParams, int iteration)
{
    char filename[100];
    sprintf(filename, DB_FILENAME, iteration);

    DBfile *dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
    if (!dbfile)
        fail(pParams->ls, "Could not create DB\n");

    int err[11];

    //TODO: Silo doesn't throw any error when running out of disk free space. Report the bug to them.
    err[0] = DBPutQuadmesh(dbfile, DB_MESHNAME, NULL, pParams->coords, pParams->dims, 3, DB_DOUBLE, DB_COLLINEAR, NULL);

    err[1] = DBPutQuadvar1(dbfile, "ex", DB_MESHNAME, pParams->mean->Ex, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    err[2] = DBPutQuadvar1(dbfile, "ey", DB_MESHNAME, pParams->mean->Ey, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    err[3] = DBPutQuadvar1(dbfile, "ez", DB_MESHNAME, pParams->mean->Ez, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    err[4] = DBPutQuadvar1(dbfile, "hx", DB_MESHNAME, pParams->mean->Hx, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    err[5] = DBPutQuadvar1(dbfile, "hy", DB_MESHNAME, pParams->mean->Hy, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    err[6] = DBPutQuadvar1(dbfile, "hz", DB_MESHNAME, pParams->mean->Hz, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);

    if (pParams->mode == VALIDATION_MODE)
    {
        err[7] = DBPutQuadvar1(dbfile, "aEy", DB_MESHNAME, pValidationFields->Ey, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
        err[8] = DBPutQuadvar1(dbfile, "aHx", DB_MESHNAME, pValidationFields->Hx, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
        err[9] = DBPutQuadvar1(dbfile, "aHz", DB_MESHNAME, pValidationFields->Hz, pParams->vdims, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
    }

    const char *names[] = {"E", "H"};
    const char *defs[] = {"{ex, ey, ez}", "{hx, hy, hz}"};
    const int types[] = {DB_VARTYPE_VECTOR, DB_VARTYPE_VECTOR};

    err[10] = DBPutDefvars(dbfile, "vecs", 2, names, types, defs, NULL);

    for (int i = 0; i < 11; ++i)
        if (err[i] != 0)
            fail(pParams->ls, "An error occurred while writting .silo files. Please check your file system and available disk space. Abort.\n");

    if (DBClose(dbfile) != 0)
        fail(pParams->ls, "An error occurred while writting .silo files. Please check your file system and available disk space. Abort.\n");
}

/** 
 * @brief Computes the total electrical energy in the system
 * @param p The parameters of the simulation that contains the mean of sim. fields
*/
double calculate_E_energy(Parameters *p)
{
    double *Ex = p->mean->Ex;
    double *Ey = p->mean->Ey;
    double *Ez = p->mean->Ez;

    double ex_energy = 0.0;
    double ey_energy = 0.0;
    double ez_energy = 0.0;

    double dv = pow(p->spatial_step, 3); // volume element

    size_t i, j, k;

    for (k = 0; k < p->Nz; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx; ++i)
            {
                ex_energy += pow(Ex[idx(p, i, j, k, 0, 0)], 2);
                ey_energy += pow(Ey[idx(p, i, j, k, 0, 0)], 2);
                ez_energy += pow(Ez[idx(p, i, j, k, 0, 0)], 2);
            }

    ex_energy *= dv;
    ey_energy *= dv;
    ez_energy *= dv;
    double E_energy = (ex_energy + ey_energy + ez_energy) * EPSILON / 2.;

    return E_energy;
}

/**
 * @brief Computes the H total energy
 * @param p The parameters of the simulation that contains the mean of sim. fields
*/
double calculate_H_energy(Parameters *p)
{
    double *Hx = p->mean->Hx;
    double *Hy = p->mean->Hy;
    double *Hz = p->mean->Hz;

    double hx_energy = 0.0;
    double hy_energy = 0.0;
    double hz_energy = 0.0;

    double dv = pow(p->spatial_step, 3); // volume element

    size_t i, j, k;

    for (k = 0; k < p->Nz; k++)
        for (j = 0; j < p->Ny; j++)
            for (i = 0; i < p->Nx; i++)
            {
                hx_energy += pow(Hx[idx(p, i, j, k, 0, 0)], 2);
                hy_energy += pow(Hy[idx(p, i, j, k, 0, 0)], 2);
                hz_energy += pow(Hz[idx(p, i, j, k, 0, 0)], 2);
            }

    hx_energy *= dv;
    hy_energy *= dv;
    hz_energy *= dv;
    double H_energy = (hx_energy + hy_energy + hz_energy) * MU / 2.;

    return H_energy;
}

/**
 * @brief Updates the validation fields and aggregate them into the second parameter
 * @param p  The parameters of the simulation that contains an allocated array for validation fields
 * @param pValidationFields The containers for the mean of the validation fields
 * @param timer The time of the simulation (in seconds)
*/
void update_validation_fields(Parameters *p, Fields *pValidationFields, double time_counter)
{
    double f_mnl = 0.5 * CELERITY * sqrt(pow(PI / p->height, 2) + pow(PI / p->length, 2)) / PI;
    //printf("fmnl: %0.20f \n", f_mnl);
    double omega = 2.0 * PI * f_mnl;
    double Z_te = (omega * MU) / sqrt(pow(omega, 2) * MU * EPSILON - pow(PI / p->length, 2));
    //printf("Zte: %0.20f \n", Z_te);
    //printf("frequency: %0.10f \n", f_mnl);

    double *vEy = p->validation_fields->Ey;
    double *vHx = p->validation_fields->Hx;
    double *vHz = p->validation_fields->Hz;

    size_t i, j, k;
    for (k = 0; k < p->Nz + 1; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx + 1; ++i)
            {
                vEy[kEy(p, i, j, k)] = (cos(2 * PI * f_mnl * time_counter) *
                                        sin(PI * i * p->spatial_step / p->length) *
                                        sin(PI * k * p->spatial_step / p->height));

                if (k != p->Nz)
                    vHx[kHx(p, i, j, k)] = ((1.0 / Z_te) *
                                            sin(2 * PI * f_mnl * time_counter) *
                                            sin(PI * i * p->spatial_step / p->length) *
                                            cos(PI * k * p->spatial_step / p->height));

                if (i != p->Nx)
                    vHz[kHz(p, i, j, k)] = (-PI / (omega * MU * p->length) *
                                            sin(2 * PI * f_mnl * time_counter) *
                                            cos(PI * i * p->spatial_step / p->length) *
                                            sin(PI * k * p->spatial_step / p->height));
            }

    aggregate_E_field(p, vEy, pValidationFields->Ey, 1, 0, 1);
    aggregate_H_field(p, vHx, pValidationFields->Hx, 1, 0, 0);
    aggregate_H_field(p, vHz, pValidationFields->Hz, 0, 0, 1);
}

/** 
 * @brief Sets the source in computation mode
 * @param p The parameters of the simulation
 * @param pFields The fields
 * @param timer The time of the simulation
*/
void set_source(Parameters *p, Fields *pFields, double time_counter)
{
    double *Ex = pFields->Ex;
    double *Ey = pFields->Ey;
    double *Hx = pFields->Hx;
    double *Hy = pFields->Hy;

    const double a_prime = 0.1;
    const double b_prime = 0.05;
    const double f = 2.45e10;

    double min_y = p->width / 2. - a_prime / 2.;
    double max_y = min_y + a_prime;

    double min_x = p->length / 2. - b_prime / 2.;
    double max_x = min_x + b_prime;

    double min_j = (int)(min_y / p->spatial_step) - 1;
    double max_j = (int)(max_y / p->spatial_step) + 1;

    double min_i = (int)(min_x / p->spatial_step) - 1;
    double max_i = (int)(max_x / p->spatial_step) + 1;

    double f_mnl = 0.5 * CELERITY * sqrt(pow(PI / p->width, 2) + pow(PI / p->length, 2)) / PI;
    double omega = 2.0 * PI * f_mnl;
    double Z_te = (omega * MU) / sqrt(pow(omega, 2) * MU * EPSILON - pow(PI / p->width, 2));

    size_t i, j;
    size_t shift_i, shift_j;

    //printf("%f, %f, %f, %f, \n", min_i, max_i, min_j, max_j);
    for (j = min_j, shift_j = 0; j < max_j; ++j, ++shift_j)
        for (i = min_i, shift_i = 0; i < max_i; ++i, ++shift_i)
        {
            Ey[kEy(p, i, j, 1)] = sin(2 * PI * f * time_counter) * sin(PI * (shift_i * p->spatial_step) / a_prime); // i = 0 pour face x =0 // -a_prime??
            Ex[kEx(p, i, j, 1)] = 0;
            Hy[kHy(p, i, j, 1)] = 0;
            Hx[kHx(p, i, j, 1)] = -(1.0 / Z_te) * sin(2 * PI * f * time_counter) * sin(PI * (shift_i * p->spatial_step) / a_prime);
        }
}

/**
 * @brief Computes the normalized mean square error of the simulation compared to the analytical solution and dumps a csv file entry with multiple variables.
 * @param p    The parameters of the simulation containing the mean aggregation of simulated fields
 * @param v    The meaned validation fields
 */
void norm_mse_dump_csv(Parameters *p, Fields *v, double timer)
{
    if (!p->dump_csv)
        return;

    double *vEy = v->Ey;
    double *vHx = v->Hx;
    double *vHz = v->Hz;

    double Er_Ey_num = 0.0;
    double Er_Ey_div = 0.0;
    double Er_Hx_num = 0.0;
    double Er_Hx_div = 0.0;
    double Er_Hz_num = 0.0;
    double Er_Hz_div = 0.0;

    size_t k, j, i;
    for (k = 0; k < p->Nz; ++k)
        for (j = 0; j < p->Ny; ++j)
            for (i = 0; i < p->Nx; ++i)
            {
                Er_Ey_num += pow(p->mean->Ey[idx(p, i, j, k, 0, 0)] - vEy[idx(p, i, j, k, 0, 0)], 2.0);
                Er_Ey_div += pow(vEy[idx(p, i, j, k, 0, 0)], 2.0);
                Er_Hx_num += pow(p->mean->Hx[idx(p, i, j, k, 0, 0)] - vHx[idx(p, i, j, k, 0, 0)], 2.0);
                Er_Hx_div += pow(vHx[idx(p, i, j, k, 0, 0)], 2.0);
                Er_Hz_num += pow(p->mean->Hz[idx(p, i, j, k, 0, 0)] - vHz[idx(p, i, j, k, 0, 0)], 2.0);
                Er_Hz_div += pow(vHz[idx(p, i, j, k, 0, 0)], 2.0);
            }

    FILE *csv;
    csv = fopen("data.csv", "a");
    if (csv == NULL)
        fail(p->ls, "Cannot open file data.csv\n");

    const double energyE = calculate_E_energy(p);
    const double energyH = calculate_H_energy(p);

    // Floating Division by 0.0 isn't standardized: Most of the times it returns inf.
    Er_Ey_div = Er_Ey_div == 0.0 ? 1e-300 : Er_Ey_div;
    Er_Hx_div = Er_Hx_div == 0.0 ? 1e-300 : Er_Hx_div;
    Er_Hz_div = Er_Hz_div == 0.0 ? 1e-300 : Er_Hz_div;

    // fprintf(csv, "timer,NormMSEEy,NormMSEHx,NormMSEHz,EnergyElectric,EnergyMagnetic,EnergyTotal,EnergyTotalTheory\n");
    fprintf(csv, "%.20lf,%.20lf,%.20lf,%.20lf,%.20lf,%.20lf,%.20lf,%.20lf\n",
            timer,
            Er_Ey_num / Er_Ey_div,
            Er_Hx_num / Er_Hx_div,
            Er_Hz_num / Er_Hz_div,
            energyE,
            energyH,
            energyE + energyH,
            (EPSILON * p->length * p->width * p->height) / 8.);

    fclose(csv);
}

/** 
 * @brief Joins the fields of each MPI process into the joined_field parameter
 * @param joined_fields The container for the joined fields
 * @param p             The parameters of the simulation
 * @warning This function needs that each process sends its results. Risk of deadlock!
*/
void join_fields(Fields *join_fields, Parameters *p, Fields *pFields)
{
    int r[6];

    for (int i = 1; i < p->ranks; ++i)
    {
        size_t k_offset = p->start_k_of_rank[i];
        r[0] = MPI_Recv(&join_fields->Ex[kEx(p, 0, 0, k_offset)], sizeof_XY(p, join_fields, join_fields->Ex) * (p->Nz + 1 - k_offset), MPI_DOUBLE, i, EX_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        r[1] = MPI_Recv(&join_fields->Ey[kEy(p, 0, 0, k_offset)], sizeof_XY(p, join_fields, join_fields->Ey) * (p->Nz + 1 - k_offset), MPI_DOUBLE, i, EY_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        r[2] = MPI_Recv(&join_fields->Ez[kEz(p, 0, 0, k_offset)], sizeof_XY(p, join_fields, join_fields->Ez) * (p->Nz - k_offset), MPI_DOUBLE, i, EZ_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        r[3] = MPI_Recv(&join_fields->Hx[kHx(p, 0, 0, k_offset)], sizeof_XY(p, join_fields, join_fields->Hx) * (p->Nz - k_offset), MPI_DOUBLE, i, HX_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        r[4] = MPI_Recv(&join_fields->Hy[kHy(p, 0, 0, k_offset)], sizeof_XY(p, join_fields, join_fields->Hy) * (p->Nz - k_offset), MPI_DOUBLE, i, HY_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        r[5] = MPI_Recv(&join_fields->Hz[kHz(p, 0, 0, k_offset)], sizeof_XY(p, join_fields, join_fields->Hz) * (p->Nz + 1 - k_offset), MPI_DOUBLE, i, HZ_TAG_TO_MAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (r[0] != MPI_SUCCESS || r[1] != MPI_SUCCESS || r[2] != MPI_SUCCESS || r[3] != MPI_SUCCESS || r[4] != MPI_SUCCESS || r[5] != MPI_SUCCESS)
            fail(p->ls, "An error occurred while receiving the fields of all processes in process with rank 0. Aborting...\n");
    }

    size_t i, j, k;
    for (k = 1; k < p->k_layers + 1; ++k)
        for (j = 0; j < p->Ny + 1; ++j)
            for (i = 0; i < p->Nx + 1; ++i)
            {
                if (i != p->Nx)
                {
                    join_fields->Ex[kEx(p, i, j, k - 1)] = pFields->Ex[kEx(p, i, j, k)];
                    join_fields->Hy[kHy(p, i, j, k - 1)] = pFields->Hy[kHy(p, i, j, k)];
                }
                if (j != p->Ny)
                {
                    join_fields->Ey[kEy(p, i, j, k - 1)] = pFields->Ey[kEy(p, i, j, k)];
                    join_fields->Hx[kHx(p, i, j, k - 1)] = pFields->Hx[kHx(p, i, j, k)];
                }
                join_fields->Ez[kEz(p, i, j, k - 1)] = pFields->Ez[kEz(p, i, j, k)];

                if (j != p->Ny && i != p->Nx)
                    join_fields->Hz[kHz(p, i, j, k - 1)] = pFields->Hz[kHz(p, i, j, k)];
            }
}

/** 
 * @brief Sends each fields to main thread to be joined
 * @param pFields The simulated fields
 * @param p The parameters of the simulation
 * @warning This function needs that the main process receives the result. Risk of deadlock!
*/
void send_fields_to_main(Fields *pFields, Parameters *p)
{
    int r[6];
    r[0] = MPI_Send(&pFields->Ex[kEx(p, 0, 0, 1)], sizeof_XY(p, pFields, pFields->Ex) * (p->k_layers + 1), MPI_DOUBLE, 0, EX_TAG_TO_MAIN, MPI_COMM_WORLD);
    r[1] = MPI_Send(&pFields->Ey[kEy(p, 0, 0, 1)], sizeof_XY(p, pFields, pFields->Ey) * (p->k_layers + 1), MPI_DOUBLE, 0, EY_TAG_TO_MAIN, MPI_COMM_WORLD);
    r[2] = MPI_Send(&pFields->Ez[kEz(p, 0, 0, 1)], sizeof_XY(p, pFields, pFields->Ez) * (p->k_layers), MPI_DOUBLE, 0, EZ_TAG_TO_MAIN, MPI_COMM_WORLD);
    r[3] = MPI_Send(&pFields->Hx[kHx(p, 0, 0, 1)], sizeof_XY(p, pFields, pFields->Hx) * (p->k_layers), MPI_DOUBLE, 0, HX_TAG_TO_MAIN, MPI_COMM_WORLD);
    r[4] = MPI_Send(&pFields->Hy[kHy(p, 0, 0, 1)], sizeof_XY(p, pFields, pFields->Hy) * (p->k_layers), MPI_DOUBLE, 0, HY_TAG_TO_MAIN, MPI_COMM_WORLD);
    r[5] = MPI_Send(&pFields->Hz[kHz(p, 0, 0, 1)], sizeof_XY(p, pFields, pFields->Hz) * (p->k_layers + 1), MPI_DOUBLE, 0, HZ_TAG_TO_MAIN, MPI_COMM_WORLD);
    if (r[0] != MPI_SUCCESS || r[1] != MPI_SUCCESS || r[2] != MPI_SUCCESS || r[3] != MPI_SUCCESS || r[4] != MPI_SUCCESS || r[5] != MPI_SUCCESS)
        fail(p->ls, "An error occurred while sending the fields of all processes in process with rank 0. Aborting...\n");
}

/**
 * @brief (MPI) Exchanges E field between ranks
 * @param p The parameters of the simulation
 * @param f The simulated fields
*/
void exchange_E_field(Parameters *p, Fields *f)
{
    MPI_Request req[2];
    int r[5];

    r[0] = MPI_Isend(&f->Ex[kEx(p, 0, 0, 1)], sizeof_XY(p, f, f->Ex), MPI_DOUBLE, p->lower_cpu, EX_TAG_TO_DOWN, MPI_COMM_WORLD, &req[0]);
    r[1] = MPI_Isend(&f->Ey[kEy(p, 0, 0, 1)], sizeof_XY(p, f, f->Ey), MPI_DOUBLE, p->lower_cpu, EY_TAG_TO_DOWN, MPI_COMM_WORLD, &req[1]);

    r[2] = MPI_Recv(&f->Ey[kEy(p, 0, 0, p->k_layers + 1)], sizeof_XY(p, f, f->Ey), MPI_DOUBLE, p->upper_cpu, EY_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    r[3] = MPI_Recv(&f->Ex[kEx(p, 0, 0, p->k_layers + 1)], sizeof_XY(p, f, f->Ex), MPI_DOUBLE, p->upper_cpu, EX_TAG_TO_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    r[4] = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    if (r[0] != MPI_SUCCESS || r[1] != MPI_SUCCESS || r[2] != MPI_SUCCESS || r[3] != MPI_SUCCESS || r[4] != MPI_SUCCESS)
        fail(p->ls, "An error occurred while exchanging E fields between processes. Aborting...\n");
}

/** 
 * @brief Exchanges the H field between ranks (MPI)
 * @param p The parameters of the simulation
 * @param f The simulated fields
*/
void exchange_H_field(Parameters *p, Fields *f)
{
    MPI_Request req[2];
    int r[5];

    r[0] = MPI_Isend(&f->Hx[kHx(p, 0, 0, p->k_layers)], sizeof_XY(p, f, f->Hx), MPI_DOUBLE, p->upper_cpu, HX_TAG_TO_UP, MPI_COMM_WORLD, &req[0]);
    r[1] = MPI_Isend(&f->Hy[kHy(p, 0, 0, p->k_layers)], sizeof_XY(p, f, f->Hy), MPI_DOUBLE, p->upper_cpu, HY_TAG_TO_UP, MPI_COMM_WORLD, &req[1]);

    r[2] = MPI_Recv(f->Hx, sizeof_XY(p, f, f->Hx), MPI_DOUBLE, p->lower_cpu, HX_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    r[3] = MPI_Recv(f->Hy, sizeof_XY(p, f, f->Hy), MPI_DOUBLE, p->lower_cpu, HY_TAG_TO_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    r[4] = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    if (r[0] != MPI_SUCCESS || r[1] != MPI_SUCCESS || r[2] != MPI_SUCCESS || r[3] != MPI_SUCCESS || r[4] != MPI_SUCCESS)
        fail(p->ls, "An error occurred while exchanging H fields between processes. Aborting...\n");
}

/** 
 * @brief Propagate the Electrical and Magnetic field using FDTD algorithm
 * @param pFields           The fields
 * @param pValidationFields The ValidationFields
 * @param pParams           The parameters of the simulation
*/
static void propagate_fields(Fields *pFields, Fields *pValidationFields, Parameters *pParams)
{
    double timer;
    double total_energy;
    int iteration = 1;
    static Fields *joined_fields;

    // Initial states and first dump
    if (pParams->rank == 0)
    {
        joined_fields = initialize_fields(pParams);
        join_fields(joined_fields, pParams, pFields);
        mean_fields(pParams, joined_fields);

        if (pParams->mode == VALIDATION_MODE)
            update_validation_fields(pParams, pValidationFields, 0.0);

        write_silo(pValidationFields, pParams, 1);
    }
    else
        send_fields_to_main(pFields, pParams);

    // Propagation and dumps
    double start_time = MPI_Wtime();
    for (timer = 0; timer <= pParams->simulation_time; timer += pParams->time_step, iteration++)
    {
        exchange_E_field(pParams, pFields);

        if (pParams->mode == COMPUTATION_MODE && pParams->rank == 0)
            set_source(pParams, pFields, timer);

        update_H_field(pParams, pFields);

        exchange_H_field(pParams, pFields);

        if (pParams->mode == COMPUTATION_MODE && pParams->rank == 0)
            set_source(pParams, pFields, timer);

        update_E_field(pParams, pFields);

        if (iteration % pParams->sampling_rate == 0)
        {
            if (pParams->rank == 0)
            {
                assert(joined_fields != NULL);

                join_fields(joined_fields, pParams, pFields);
                mean_fields(pParams, joined_fields);
                if (pParams->mode == VALIDATION_MODE)
                {
                    update_validation_fields(pParams, pValidationFields, timer);
                    norm_mse_dump_csv(pParams, pValidationFields, timer);
                }

                write_silo(pValidationFields, pParams, iteration);
            }
            else
                send_fields_to_main(pFields, pParams);
        }
    }
    double end_time = MPI_Wtime();
}

//--------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------ Main function
//--------------------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        fail(NULL, "Could not init MPI. Abort.\n");

    int rank;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        fail(NULL, "Could not get the rank of current process. Abort.\n");

    if (rank == 0)
        printf("Welcome into our microwave oven electromagnetic field simulator! \n");

    if (argc < 2)
        fail(NULL, "This program needs at least 1 argument: the parameters file (.txt). Eg.: ./microwave param.txt\n");

    printf("Process %d: Loading the parameters of the simulation...\n", rank);

    Parameters *pParameters = load_parameters(argv[1], rank);

    if (pParameters->time_step > pParameters->simulation_time)
        fail(pParameters->ls, "The time step must be smaller than the simulation time!\n");

    if (rank == 0)
    {
        printf("Main process: Creating mesh\n");
        pParameters->mean = initialize_mean_fields(pParameters);
        compute_oven(pParameters);
    }

    printf("Process %d: Initializing fields\n", rank);
    Fields *pFields = initialize_cpu_fields(pParameters);

    Fields *pValidationFields;
    if (pParameters->mode == VALIDATION_MODE)
    {
        if (rank == 0)
        {

            if (argc == 3 && strlen(argv[2]) == 10 && strncmp(argv[2], "--dump-csv", 10) == 0)
            {
                pParameters->dump_csv = 1;
                // Prepare the csv file
                FILE *csv;
                csv = fopen("data.csv", "w");
                if (csv == NULL)
                    fail(pParameters->ls, "Cannot open file data.csv\n");
                fprintf(csv, "timer,NormMSEEy,NormMSEHx,NormMSEHz,EnergyElectric,EnergyMagnetic,EnergyTotal,EnergyTotalTheory\n");
                fclose(csv);
                printf("Main process: Energy and errors will be dumped into data.csv.\n");
            }
            else
                printf("Main process: Did you know? You can use --dump-csv option in order to dump extra information about the simulation into a csv file.\n");

            pValidationFields = initialize_mean_fields(pParameters);
            pParameters->validation_fields = initialize_fields(pParameters);
            printf("Main process: Validation mode activated. \n");
            // Free what's not needed for validation.
            Free(&(pParameters->ls), pValidationFields->Ex);
            Free(&(pParameters->ls), pValidationFields->Ez);
            Free(&(pParameters->ls), pValidationFields->Hy);
            Free(&(pParameters->ls), pParameters->validation_fields->Ex);
            Free(&(pParameters->ls), pParameters->validation_fields->Ez);
            Free(&(pParameters->ls), pParameters->validation_fields->Hy);
        }
        printf("Process %d: Setting initial conditions\n", rank);
        set_initial_conditions(pFields->Ey, pParameters);
    }

    printf("Process %d: Launching simulation\n", rank);
    propagate_fields(pFields, pValidationFields, pParameters);
    printf("Process %d: Freeing memory...\n", rank);

    freeAll(pParameters->ls);

    if (rank == 0)
        printf("Simulation complete!\n");

    if (MPI_Finalize() != MPI_SUCCESS)
        perror("Something happend with the MPI_Finalize function...\n");

    return 0;
}