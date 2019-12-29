#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <omp.h>
#include <time.h>
#include <err.h>

#include "kmeans.h"

/**
** \brief Load vectors data from a file.
** \param filename The path to the input file.
** \param vect_count The number of vectors to load.
** \param vect_dim The dimension of the vectors.
** \return A pointer to an array of all vectors values.
**/
static float *load_data(char *filename, unsigned vect_count, unsigned vect_dim)
{
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        err(1, "Error while opening %s", filename);

    struct stat st;
    size_t total_size = vect_count * vect_dim * sizeof(float);
    if (fstat(fd, &st) != -1 && total_size > (size_t) st.st_size)
        errx(1, "Error in parameters");

    void *tab = mmap(NULL, vect_count * vect_dim * sizeof(float), PROT_READ,
                     MAP_SHARED, fd, 0);
    if (tab == MAP_FAILED)
        err(1, "Error while mmap");
    close(fd);

    return tab;
}

/**
** \brief Save k-means results to a file.
** \param data The array of assignments.
** \param vect_count The number of vectors in data.
** \param filename The path to the output file.
**/
static void save_output(
        unsigned char *data,
        unsigned vect_count,
        char *filename
)
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
        err(1, "Cannot create file: %s\n", filename);

    for(unsigned i = 0; i < vect_count; ++i)
    {
        float f = data[i];
        fwrite(&f, sizeof(float), 1, fp);
    }

    fclose(fp);
}

/**
** \brief Entry point of the program.
** \param argc The number of arguments.
** \param argv A null terminated array of char * representing the arguments.
** \return The exit code of the program.
**/
int main(int argc, char *argv[])
{
    // Check and parse args
    if (argc != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> \
                <dim: int> <nbvec:int> <datafile> <outputClassFile>\n",
                argv[0]);
    unsigned max_iter = atoi(argv[2]);
    //double min_err = atof(argv[3]);
    unsigned K = atoi(argv[1]);
    unsigned vect_dim = atoi(argv[4]);
    unsigned vect_count = atoi(argv[5]);
    char *input = argv[6];
    char *output = argv[7];

    omp_set_num_threads(omp_get_num_procs());

    // Run K-means algorithm
    printf("Start Kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n",
            input, K, vect_dim, vect_count);
    float *data = load_data(input, vect_count, vect_dim);
    srand(time(0));
    unsigned char *res = kmeans(data, vect_count, vect_dim, K, max_iter);

    // Save output and free memory
    save_output(res, vect_count, output);
    munmap(data, vect_count * vect_dim * sizeof(float));
    free(res);

    return 0;
}
