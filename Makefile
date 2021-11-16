SRC=main.c
EXE=microwave
all:
	gcc ${SRC} -lm -lsilo -std=c99 -o ${EXE}

parallel:
	# Check to add MPI library ?
	gcc ${SRC} -lm -lsilo -fopenmp -std=c99 -o ${EXE}

clean:
	rm -rf ${EXE} *.o
