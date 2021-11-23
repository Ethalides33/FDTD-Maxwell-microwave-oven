SRC=main.c
EXE=microwave

all:
	make clean
	mkdir r
	gcc ${SRC} -lm -lsiloh5 -std=c99 -o ${EXE}

noth5:
	@echo "If you get error 'not found silo library', run 'make h5'"
	gcc ${SRC} -lm -lsilo -std=c99 -o ${EXE}

debug:
	gcc ${SRC} -g -lm -lsiloh5 -std=c99 -o ${EXE}

parallel:
	# Check to add MPI library ?
	gcc ${SRC} -lm -lsilo -fopenmp -std=c99 -o ${EXE}

clean:
	rm -rf r
	rm -rf ${EXE} *.o
