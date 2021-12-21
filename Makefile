SRC=main.c
EXE=microwave

all:
	@echo "If you get error 'not found silo library', run 'make noth5'"
	@echo "Use 'make optimized' for tremendous execution speed!"
	@echo "----"
	@echo "Removing old executables..."
	make clean
	mkdir r
	@echo "Compiling..."
	gcc ${SRC} -lm -lsiloh5 -std=c99 -o ${EXE}

noth5:
	gcc ${SRC} -lm -lsilo -std=c99 -o ${EXE}

optimized:
	gcc ${SRC} -lm -lsiloh5 -std=c99 -O3 -o ${EXE}

debug:
	gcc ${SRC} -g -lm -lsiloh5 -std=c99 -o ${EXE}

clean:
	rm -rf r
	rm -rf ${EXE} *.o
