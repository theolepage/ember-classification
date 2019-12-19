#include <err.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>
#include <unistd.h>

float *loadData(char *fileName, unsigned nbVec, unsigned dim)
{
    int fd = open(fileName, O_RDONLY);
    if (fd == -1)
        err(1, "Error while openning %s", fileName);

    struct stat st;
    if (fstat(fd, &st) != -1 && nbVec * dim * sizeof(float) > (size_t)st.st_size)
        errx(1, "Error in parameters");

    void *tab = mmap(NULL, nbVec * dim * sizeof(float), PROT_READ,
                     MAP_SHARED, fd, 0);
    if (tab == MAP_FAILED)
        err(1, "Error while mmap");
    close(fd);

    return tab;
}

void writeClassinFloatFormat(unsigned char *data, unsigned nbelt, char *fileName) 
{
    FILE *fp = fopen(fileName, "w");
    if (!fp)
        err(1, "Cannot create File: %s\n", fileName);

    for(unsigned i = 0; i < nbelt; ++i) 
    {
        float f = data[i];
        fwrite(&f, sizeof(float), 1, fp);
    }

    fclose(fp);
}

double distance(float *vec1, float *vec2, unsigned dim) 
{
    double dist = 0;
    for(unsigned i = 0; i < dim; ++i, ++vec1, ++vec2) 
    {
        double d = *vec1 - *vec2;
        dist += d * d;
    }

    return sqrt(dist);
}

unsigned char classify(float *vec, float *means, unsigned dim,
                       unsigned char K, double *e) 
{
    unsigned char min = 0;
    float dist, distMin = FLT_MAX;

    for(unsigned i = 0; i < K; ++i) 
    {
        dist = distance(vec, means + i * dim, dim);
        if(dist < distMin) 
        {
            distMin = dist;
            min = i;
        }
    }

    *e = distMin;
    return min;
}

static inline void print_result(int iter, double time, float err)
{
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"error\": \"%f\"}\n", iter, time, err);
    else
        printf("Iteration: %d, Time: %lf, Error: %f\n", iter, time, err);
}

unsigned char *Kmeans(float *data, unsigned nbVec, unsigned dim,
                      unsigned char K, double minErr, unsigned maxIter)
{
    unsigned iter = 0;
    double e = 0.;
    double diffErr = DBL_MAX;
    double err = DBL_MAX;

    float *means = malloc(sizeof(float) * dim * K);
    unsigned *card = malloc(sizeof(unsigned) * K);
    unsigned char* c = malloc(sizeof(unsigned char) * nbVec);

    // Random init of c
    for(unsigned i = 0; i < nbVec; ++i)
        c[i] = rand() / (RAND_MAX + 1.) * K;

    for(unsigned i = 0; i < dim * K; ++i)
        means[i] = 0.;

    for(unsigned i = 0; i < K; ++i)
        card[i] = 0.;

    for(unsigned i = 0; i < nbVec; ++i) 
    {
        for(unsigned j = 0; j < dim; ++j)
            means[c[i] * dim + j] += data[i * dim  + j];
        ++card[c[i]];
    }

    for(unsigned i = 0; i < K; ++i)
        for(unsigned j = 0; j < dim; ++j)
            means[i * dim + j] /= card[i];

    while ((iter < maxIter) && (diffErr > minErr)) 
    {
        double t1 = omp_get_wtime();
        diffErr = err;
        // Classify data
        err = 0.;
        for(unsigned i = 0; i < nbVec; ++i) 
        {
            c[i] = classify(data + i * dim, means, dim, K, &e);
            err += e;
        }

        // update Mean
        for(unsigned i = 0; i < dim * K; ++i)
            means[i] = 0.;

        for(unsigned i = 0; i < K; ++i)
            card[i] = 0.;

        for(unsigned i = 0; i < nbVec; ++i) 
        {
            for(unsigned j = 0; j < dim; ++j)
                means[c[i] * dim + j] += data[i * dim  + j];
            ++card[c[i]];
        }
        for(unsigned i = 0; i < K; ++i)
            for(unsigned j = 0; j < dim; ++j)
                means[i * dim + j] /= card[i];

        ++iter;
        err /= nbVec;
        double t2 = omp_get_wtime();
        diffErr = fabs(diffErr - err);

        print_result(iter, t2 - t1, err);
    }

    free(means);
    free(card);

    return c;
}

int main(int ac, char *av[])
{
    if (ac != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> <dim: int> <nbvec:int> <datafile> <outputClassFile>\n", av[0]);

    unsigned maxIter = atoi(av[2]);
    double minErr = atof(av[3]);
    unsigned K = atoi(av[1]);
    unsigned dim = atoi(av[4]);
    unsigned nbVec = atoi(av[5]);

    printf("Start Kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", av[6], K, dim, nbVec);

    float *tab = loadData(av[6], nbVec, dim);
    unsigned char * classif = Kmeans(tab, nbVec, dim, K, minErr, maxIter);

    writeClassinFloatFormat(classif, nbVec, av[7]);

    munmap(tab, nbVec * dim * sizeof(float));
    free(classif);

    return 0;
}
