# FDTD-Maxwell-microwave-oven
Simulation of EM waves in a microwave oven using the FDTD scheme.
Based on the article from Kane S. Yee, "Numerical Solution of Initial Boundary Value Problems Involving Maxwell’s Equations in Isotropic Media", 1961.

## Perequisites

This project uses silo library to dump the computed data.
In order to compile this code, make sure you have silo library installed:

On linux: 

```bash 
$ sudo apt install -y libsilo-dev libsiloh5-0
```

On mac (using homebrew): 

```bash
$ brew tap datafl4sh/code && brew install datafl4sh/code/silo
```

On NIC5, don't forget to load the silo module: 

```bash
$ module load releases/2020b Silo/4.11-foss-2020b
```

<details>
  <summary> On Windows, you have to compile the library itself (Click to expand)</summary>


  ```c
/**
* ... extracted from: https://gitlab.onelab.info/mcicuttin/snippets/-/blob/master/silo_example/silo.c
* 1) Get the library from
     https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2-bsd.tgz
 * 2) tar -zxvf silo-4.10.2-bsd.tgz
 * 3) cd silo-4.10.2-bsd
 * 4) export SILO_PREFIX=<path where you want to install silo>
 * 5) ./configure --prefix=$SILO_PREFIX
 * 6) make && make install
 *
 * In order to compile your program and link against the copy of SILO you
 * have just configured and installed, do
 *
 * gcc -I$SILO_PREFIX/include -L$SILO_PREFIX/lib myprogram.c -lsilo
 * ...
 **/
```
  
</details>

[More information on Silo](https://wci.llnl.gov/sites/wci/files/2020-08/GettingDataIntoVisIt2.0.0.pdf).


This branch is parallel, you must install OpenMPI and OpenMP:

```bash
$ sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi1.10 libopenmpi-dev libomp-dev
```

## Compilation
Simply run 
```bash
$ make
```
This will create an executable (`microwave`) that you can run.


If you installed SILO from your operating system package manager, there  should be no need to specify `-I` and `-L` paths, however there is a chance that you will get the HDF5-enabled version. In that case, if `-lsilo` does not work, with `-lsiloh5`:

```bash
$ make h5
```


## Usage
Check the tutorial [here](https://mpitutorial.com/tutorials/mpi-hello-world/) for the parallel instructions
```bash
$ ./microwave ./params.txt
```
When runned with a good parameters file , the programm produces a `result.silo` file which contains data that can be vizualized with [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit) in a few steps:

* Launch VisIt
* Open your file using the Open function
* ? (check if appropriate) Add -> Mesh -> mesh
* ? (check if appropriate) Add -> Pseudocolor -> u

## Debug with gdb
In order to debug the code with gdb, use the following commands:


```bash
$ ./make debug
$ gdb --args ./microwave ./params.txt
```


ENJOY!!!

