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
typedef struct parameters{
  float width;
  float height;
  float length;
  double spatial_step;
  double time_step;
  float simulation_time;
  uint sampling_rate;

} Parameters;

typedef struct fields{

double* E_x;
double* E_y;
double* E_z;
double* H_x;
double* H_y;
double* H_z;


} Fields;

/**
------------------------------------------------------
---------------------------- Function's Specifications
------------------------- Signatures & Implementations
------------------------------------------------------
**/

/** Critical allocation shortcut (malloc or Critical error with exit) **/
void *malloc_and_check_critical(size_t __size){
  void *ptr = malloc(__size);
  if(!ptr){
    perror("CRITICAL ERROR: Could not allocate enough memory!");
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
  FILE* pParameters_file = fopen(filename, "r");
  Parameters *pParameters = malloc_and_check_critical(sizeof(Parameters));

  if(!pParameters_file){
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


Fields *initialize_fields(uint space_steps_x, uint space_steps_y, uint space_steps_z){

    Fields *pFields = malloc_and_check_critical(sizeof(Fields));
    uint space_size = space_steps_x*space_steps_y*space_steps_z;

    pFields->E_x = malloc_and_check_critical(sizeof(double)*space_size);
    memset(&pFields->E_x, 0, sizeof(double)*space_size);  // NOT SURE ABOUT THE &.
    pFields->E_y = malloc_and_check_critical(sizeof(double)*space_size);
    memset(&pFields->E_y, 0, sizeof(double)*space_size);
    pFields->E_z = malloc_and_check_critical(sizeof(double)*space_size);
    memset(&pFields->E_z, 0, sizeof(double)*space_size);

    pFields->H_x = malloc_and_check_critical(sizeof(double)*space_size);
    memset(&pFields->H_x, 0, sizeof(double)*space_size);
    pFields->H_y = malloc_and_check_critical(sizeof(double)*space_size);
    memset(&pFields->H_y, 0, sizeof(double)*space_size);
    pFields->H_z = malloc_and_check_critical(sizeof(double)*space_size);
    memset(&pFields->H_z, 0, sizeof(double)*space_size);

    return pFields;
}

void set_initial_conditions(Fields *pFields){

    print("hi");

}


void update_H_field(){

}

void update_E_field(){

}


void propagate_fields(Fields *pFields, Parameters *pParams){

    float time_counter;
    for(time_counter = 0; time_counter <= pParams->simulation_time; time_counter += pParams->time_step){
        //below should be parallelized.
        update_H_field(pFields->H_x, pFields->E_y, pFields-> E_z, pParams); // need & ?  //H_x
        update_H_field(pFields->H_y, pFields->E_z, pFields-> E_x, pParams); // need & ? //H_y
        update_H_field(pFields->H_z, pFields->E_x, pFields-> E_y, pParams); // need & ? //H_z

        update_E_field(pFields->E_x, pFields->H_z, pFields-> H_y, pParams); // need & ? //E_x
        update_E_field(pFields->E_y, pFields->H_x, pFields-> H_z, pParams); // need & ? //E_y
        update_E_field(pFields->E_z, pFields->H_y, pFields-> H_x, pParams); // need & ? should check math // E_z


    }
}



/** Free parameters from memory
 * Arguments:
 *  parameters: The pointer to the Parameters object
**/
void free_parameters(Parameters *parameters)
{
  if(parameters){
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
  printf("Welcome into our microwave oven simulation engine ! \n");

  if(argc != 2){
    perror("This program needs 1 argument: the parameters file (.txt). Eg.: ./microwave param.txt");
    return EXIT_FAILURE;
  }


  printf("Loading the parameters...\n");
  Parameters *pParameters = load_parameters(argv[1]);  // mParameters ?
  printf("width: %f \n", pParameters->width);
  printf("height: %f \n", pParameters->height);
  printf("length: %f \n", pParameters->length);
  printf("spatial: %lf \n", pParameters->spatial_step);
  printf("time: %lf \n", pParameters->time_step);
  printf("total: %f \n", pParameters->simulation_time);
  printf("rate: %u \n", pParameters->sampling_rate);
    if(time_step>simulation_time){
        perror("The time step must be lower than the simulation time!");
        return EXIT_FAILURE;
    }

  Fields *pFields = initialize_fields();
  set_initial_conditions(pFields);
  propagate_fields(pFields, pParameters);

  printf("Freeing memory...\n");
  // Free allocated spaces
  free_parameters(pParameters);
  
  printf("Simulation complete!\n");
  return 0;
}
