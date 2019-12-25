#include <time.h>
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

static inline void print_result(int iter, double time, float err)
{
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"error\": \"%f\"}\n", iter, time, err);
    else
        printf("Iteration: %d, Time: %lf, Error: %f\n", iter, time, err);
}

struct kmeans_state
{
    unsigned vect_count;
    unsigned vect_dim;
    unsigned char K;

    unsigned char *assignment;
    float *centroids;
    float *centroids_sum;
    unsigned *centroids_count;

    float *upper_bounds;
    float *lower_bounds;
    float *p;
    float *s;
};

/**
** \brief Update upper and lower bounds of a vector and its assignment.
** \param vectors The feature vectors data.
** \param i The index of the current vector.
** \param state A pointer to the struct representing algorithm's state.
**/
static void point_all_ctrs(
        float *vectors,
        unsigned i,
        struct kmeans_state *state,
        double *e
)
{
    // Update assignment
    float min_dist = FLT_MAX;
    unsigned char min_dist_index = 0;
    for (unsigned c = 0; c < state->K; c++)
    {
        float tmp_dist = distance(vectors + i * state->vect_dim,
                state->centroids + c * state->vect_dim,
                state->vect_dim);
        if (tmp_dist < min_dist)
        {
            min_dist = tmp_dist;
            min_dist_index = c;
        }
    }
    *e = min_dist;
    state->assignment[i] = min_dist_index;

    // Update upper bound
    state->upper_bounds[i] = distance(vectors + i * state->vect_dim,
            state->centroids + min_dist_index * state->vect_dim,
            state->vect_dim);

    // Update lower bound
    min_dist = 0;
    for (unsigned c = 0; c < state->K; c++)
    {
        if (c == min_dist_index)
            continue;
float tmp_dist = distance(vectors + i * state->vect_dim,
                state->centroids + c * state->vect_dim,
                state->vect_dim);
        if (tmp_dist < min_dist)
            min_dist = tmp_dist;
    }
    state->lower_bounds[i] = min_dist;
}

static void move_centers(struct kmeans_state *state)
{
    for (unsigned c = 0; c < state->K; c++)
    {
        // Make a copy of current centroid
        float *old_centroid = calloc(state->vect_dim, sizeof(float));
        for (unsigned d = 0; d < state->vect_dim; d++)
        {
            old_centroid[d] = state->centroids[c * state->vect_dim + d];
        }

        // Compute new centroid
        for (unsigned d = 0; d < state->vect_dim; d++)
        {
            unsigned count = state->centroids_count[c];
            float value = state->centroids[c * state->vect_dim + d] / count;
            state->centroids[c * state->vect_dim + d] = value;
        }

        // Store difference between old and new centroid
        state->p[c] = distance(old_centroid,
                state->centroids + c * state->vect_dim,
                state->vect_dim);

        free(old_centroid);
    }
}

static void update_bounds(struct kmeans_state *state)
{
    // r = argmax(p(c))
    float max = 0;
    unsigned max_index = 0;
    for (unsigned c = 0; c < state->K; c++)
    {
        float max_tmp = state->p[c];
        if (max_tmp > max)
        {
            max = max_tmp;
            max_index = c;
        }
    }
    unsigned r = max_index;

    // r_prime = argmax(p(c)) c != r
    max = 0;
    max_index = 0;
    for (unsigned c = 0; c < state->K; c++)
    {
        if (c == r)
            continue;

        float max_tmp = state->p[c];
        if (max_tmp > max)
        {
            max = max_tmp;
            max_index = c;
        }
    }
    unsigned r_prime = max_index;

    for (unsigned i = 0; i < state->vect_count; i++)
    {
        state->upper_bounds[i] += state->p[state->assignment[i]];
        if (r == state->assignment[i])
            state->lower_bounds[i] -= state->p[r_prime];
        else
            state->lower_bounds[i] -= state->p[r];
    }
}

/**
** \brief K-means algorithm (Hamerly's version)
** \param vectors The feature vectors data.
** \param vect_count The number of feature vectors.
** \param vect_dim The number of features (dimension of vectors).
** \param K The number of clusters.
**/
unsigned char *Kmeans(
        float *vectors,
        unsigned vect_count,
        unsigned vect_dim,
        unsigned char K,
        double min_err,
        unsigned max_iter
)
{
    // Init state and allocate memory
    struct kmeans_state *state = calloc(1, sizeof(struct kmeans_state));
    state->K = K;
    state->vect_count = vect_count;
    state->vect_dim = vect_dim;
    state->assignment = calloc(vect_count, sizeof(unsigned char)); // a
    state->centroids = calloc(K * vect_dim, sizeof(float)); // c
    state->centroids_sum = calloc(K * vect_dim, sizeof(float)); // c'
    state->centroids_count = calloc(K, sizeof(unsigned)); // q
    state->upper_bounds = calloc(vect_count, sizeof(float)); // u
    state->lower_bounds = calloc(vect_count, sizeof(float)); // l
    state->p = calloc(K, sizeof(float)); // p
    state->s = calloc(K, sizeof(float)); // s

    unsigned iter = 0;
    double e = 0;
    double diffErr = DBL_MAX;
    double err = DBL_MAX;
    double *min_dist = calloc(vect_count, sizeof(double));


    // Question: Initialize upper_bounds (with INFINITY values)  and assignment?
    // Question: Is it a problem if I use one array of both c and c'?

    // Init randomly the centers.
    int *centroids_index = calloc(state->K, sizeof(int));
    for (int i = 0; i < state->K; i++)
    {
        centroids_index[i] = rand() / (RAND_MAX + 1.) * vect_count;
        // Check that the given index is unique in the array.
        for (int j = 0; j < i; j++)
        {
            // If the index is already used by another centroids,
            // choose another value (restart the loop from i)
            if (centroids_index[i] == centroids_index[j])
            {
                i -= 1;
                break;
            }
        }
    }
    for (int i = 0; i < state->K; i++)
    {
        float *c_vec = vectors + centroids_index[i] * state->vect_dim;
        for (unsigned j = 0; j < state->vect_dim; j++)
            state->centroids_sum[i * state->vect_dim + j] = c_vec[j];
    }
    free(centroids_index);

    // Initialize (Algorithm 2)
    for (unsigned i = 0; i < vect_count; i++)
    {
        state->upper_bounds[i] = INFINITY;
        point_all_ctrs(vectors, i, state, &e);
        min_dist[i] = e;

        unsigned char c = state->assignment[i];
        state->centroids_count[c]++;

        for (unsigned d = 0; d < vect_dim; d++)
            state->centroids[c * vect_dim + d] += vectors[i * vect_dim + d];
    }

    // Main loop
    while ((iter < max_iter) && (diffErr > min_err))
    {
        double t1 = omp_get_wtime();
        diffErr = err;
        err = 0;
        // Update s
        for (unsigned c1 = 0; c1 < K; c1++)
        {
            float min = 0;
            for (unsigned c2 = 0; c2 < K; c2++)
            {
                if (c1 == c2)
                    continue;

                float min_tmp = distance(state->centroids + c1 * vect_dim,
                        state->centroids + c2 * vect_dim,
                        vect_dim);
                min = (min_tmp < min) ? min_tmp : min;
            }
            state->s[c1] = min;
        }

        // Update centroids if necessary
        for (unsigned i = 0; i < vect_count; i++)
        {
            float m = fmax(state->s[state->assignment[i]] / 2, state->lower_bounds[i]);

            // First bound test
            if (state->upper_bounds[i] > m)
            {
                // Tighten upper bound
                state->upper_bounds[i] = distance(vectors + i * vect_dim,
                        state->centroids + state->assignment[i] * vect_dim,
                        state->vect_dim);

                // Second bound test
                if (state->upper_bounds[i] > m)
                {
                    unsigned char old_assignment = state->assignment[i];
                    point_all_ctrs(vectors, i, state, &e);
                    min_dist[i] = e;
                    unsigned char curr_assignment = state->assignment[i];

                    // Update centroids
                    if (old_assignment != curr_assignment)
                    {
                        state->centroids_count[old_assignment]--;
                        state->centroids_count[curr_assignment]++;
                        for (unsigned d = 0; d < vect_dim; d++)
                        {
                            float value = vectors[i * vect_dim + d];
                            state->centroids[old_assignment * vect_dim + d] -= value;
                            state->centroids[curr_assignment * vect_dim + d] += value;
                        }
                    }
                }
            }
        }
        for (unsigned i = 0; i < state->vect_count; i++)
            err += min_dist[i];
        err /= state->vect_count;
        double t2 = omp_get_wtime();
        diffErr = fabs(diffErr - err);
        print_result(iter, t2 - t1, err);
        move_centers(state);
        update_bounds(state);
        iter += 1;
    }

    // Free state memory
    unsigned char *res = state->assignment;
    free(state->centroids);
    free(state->centroids_count);
    free(state->upper_bounds);
    free(state->lower_bounds);
    free(state->p);
    free(state->s);
    free(state);

    return res;
}

int main(int argc, char *argv[])
{
    // Check and parse args
    if (argc != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> \
                <dim: int> <nbvec:int> <datafile> <outputClassFile>\n",
                argv[0]);
    unsigned max_iter = atoi(argv[2]);
    double min_err = atof(argv[3]);
    unsigned K = atoi(argv[1]);
    unsigned vect_dim = atoi(argv[4]);
    unsigned vect_count = atoi(argv[5]);
    char *input = argv[6];
    char *output = argv[7];

    srand(time(0));

    // Run K-means algorithm
    printf("Start Kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n",
            input, K, vect_dim, vect_count);
    float *data = loadData(input, vect_count, vect_dim);
    unsigned char *res = Kmeans(data, vect_count, vect_dim, K, min_err, max_iter);

    // Save output and free memory
    writeClassinFloatFormat(res, vect_count, output);
    munmap(data, vect_count * vect_dim * sizeof(float));
    free(res);

    return 0;
}
