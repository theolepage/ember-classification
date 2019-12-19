CC=gcc
CFLAGS= -Wall -Wextra -Werror -std=c99 -pedantic -O3
LDFLAGS= -fopenmp -lm

BIN=kmeans
OBJ=kmeans.o

all: $(BIN)

$(BIN): $(OBJ)

clean:
	$(RM) $(OBJ) $(BIN)
