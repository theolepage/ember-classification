CC=gcc
CFLAGS= -Wall -Wextra -Werror -std=c99 -pedantic -O3 -fopenmp
LDFLAGS= -fopenmp
LDLIBS= -lm

BIN=kmeans
OBJ=main.o kmeans.o init_methods.o

all: $(BIN)

$(BIN): $(OBJ)

test: $(BIN)
	rm -f out.dat
	time ./kmeans 3 20 1.0 2351 900000 ../ember/Xtrain.dat out.dat;
	python ../python/eval.py ../ember/Ytrain.dat out.dat

clean:
	$(RM) $(OBJ) $(BIN)
