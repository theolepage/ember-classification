CC=gcc
CFLAGS= -Wall -Wextra -Werror -std=c99 -pedantic -O3
LDFLAGS= -fopenmp
LDLIBS= -lm

BIN=kmeans
OBJ=kmeans.o kmeanspp.o

all: $(BIN)

$(BIN): $(OBJ)

clean:
	$(RM) $(OBJ) $(BIN)
