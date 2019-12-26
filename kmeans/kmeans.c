#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
** \brief Compute the euclidean distance between two vectors.
** \param vec1 The first vector.
** \param vec2 The second vector.
** \param dim The dimension of both vectors.
** \return The distance between vec1 and vec2.
**/
static double distance(float *vec1, float *vec2, unsigned dim)
{
    double dist = 0;
    for(unsigned i = 0; i < dim; ++i, ++vec1, ++vec2)
    {
        double d = *vec1 - *vec2;
        dist += d * d;
    }
    return sqrt(dist);
}

/**
** \brief Print debugging information for each iteration.
** \param iter The id of the iteration.
** \param time The time of the iteration execution.
** \param change The number of changes applied during the iteration.
**/
static inline void print_result(int iter, double time, unsigned change)
{
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"change\": \
        \"%d\"}\n", iter, time, change);
    else
        printf("Iteration: %d, Time: %lf, Change: %d\n", iter, time, change);
}

/**
** \brief Update assignment and the bounds of a vector.
** \param vectors The feature vectors data.
** \param i The index of the current vector.
** \param state A pointer to the struct representing algorithm's state.
**/
static void point_all_ctrs(
        float *vectors,
        unsigned i,
        struct kmeans_state *state
)
{
    float min_dist = FLT_MAX;
    float min_dist_p = FLT_MAX;
    unsigned char min_dist_index = 0;

    // Find the two closest centroids
    for (unsigned c = 0; c < state->K; c++)
    {
        float tmp_dist = distance(vectors + i * state->vect_dim,
                state->centroids + c * state->vect_dim,
                state->vect_dim);

        if (tmp_dist < min_dist)
        {
            min_dist_p = min_dist;
            min_dist = tmp_dist;
            min_dist_index = c;
        }
        else if (tmp_dist < min_dist_p)
            min_dist_p = tmp_dist;
    }

    // Update assignment and bounds
    state->upper_bounds[i] = min_dist;
    state->assignment[i] = min_dist_index;
    state->lower_bounds[i] = min_dist_p;
}

/**
** \brief Update centroids during k-means algorithm.
** \param state A pointer to the struct representing algorithm's state.
** \return The maximum distance a centroid moved.
**/
static float move_centers(struct kmeans_state *state)
{
    float max_moved = 0;

    for (unsigned c = 0; c < state->K; c++)
    {
        // Compute new centroid centers
        for (unsigned d = 0; d < state->vect_dim; d++)
        {
            float count = state->centroids_count[c];
            float res = 0;
            if (count)
                res = state->centroids_next[c * state->vect_dim + d] / count;
            state->centroids_next[c * state->vect_dim + d] = res;
        }

        // Compute distance between old and new centroid
        state->p[c] = distance(state->centroids + c * state->vect_dim,
                state->centroids_next + c * state->vect_dim,
                state->vect_dim);

        // Update max_moved
        if (state->p[c] > max_moved)
            max_moved = state->p[c];
    }

    return max_moved;
}

/**
** \brief Update bounds during k-means algorithm.
** \param max_moved The maximum distance a centroid moved.
** \param state A pointer to the struct representing algorithm's state.
**/
static void update_bounds(struct kmeans_state *state, float max_moved)
{
    for (unsigned i = 0; i < state->vect_count; i++)
    {
        state->upper_bounds[i] += state->p[state->assignment[i]];
        state->lower_bounds[i] -= max_moved;
    }
}

/**
** \brief Initialize centroids before k-means algorithm.
** \param vectors The feature vectors data.
** \param state A pointer to the struct representing algorithm's state.
**/
static void init_centroids(float *vectors, struct kmeans_state *state)
{
    int *centroids_index = calloc(state->K, sizeof(int));
    for (int i = 0; i < state->K; i++)
    {
        centroids_index[i] = rand() / (RAND_MAX + 1.) * state->vect_count;

        // Check that the given index is unique in the array.
        for (int j = 0; j < i; j++)
        {
            if (centroids_index[i] == centroids_index[j])
            {
                i--;
                break;
            }
        }
    }
    for (int i = 0; i < state->K; i++)
    {
        float *c_vec = vectors + centroids_index[i] * state->vect_dim;
        for (unsigned j = 0; j < state->vect_dim; j++)
            state->centroids[i * state->vect_dim + j] = c_vec[j];
    }
    free(centroids_index);
}

/**
** \brief Run k-means algorithm (Hamerly's version).
** \param vectors The feature vectors data.
** \param vect_count The number of feature vectors.
** \param vect_dim The number of features (dimension of vectors).
** \param K The number of clusters.
** \param max_iter The number of maximum iterations.
**/
unsigned char *kmeans(
        float *vectors,
        unsigned vect_count,
        unsigned vect_dim,
        unsigned char K,
        unsigned max_iter
)
{
    // Init state and allocate memory
    struct kmeans_state *state = calloc(1, sizeof(struct kmeans_state));
    state->K = K;
    state->vect_count = vect_count;
    state->vect_dim = vect_dim;
    state->assignment = calloc(vect_count, sizeof(unsigned char));
    state->centroids = calloc(K * vect_dim, sizeof(float));
    state->centroids_next = calloc(K * vect_dim, sizeof(float));
    state->centroids_count = calloc(K, sizeof(unsigned));
    state->upper_bounds = malloc(vect_count * sizeof(float));
    state->lower_bounds = calloc(vect_count, sizeof(float));
    state->p = calloc(K, sizeof(float));
    state->s = calloc(K, sizeof(float));

    // Initialize
    init_centroids(vectors, state);
    state->centroids_count[0] = vect_count;
    for (unsigned i = 0; i < vect_count; i++)
        state->upper_bounds[i] = FLT_MAX;
    for (unsigned c = 0; c < K; c++)
        state->s[c] = FLT_MAX;

    unsigned iter = 0;
    unsigned change_cluster = 1;

    // Main loop
    while (iter < max_iter && change_cluster)
    {
        double t1 = omp_get_wtime();
        change_cluster = 0;

        // Update shortest distance between each two cluster
        for (unsigned c1 = 0; c1 < K; c1++)
        {
            for (unsigned c2 = c1 + 1; c2 < K; c2++)
            {
                float min_tmp = distance(state->centroids + c1 * vect_dim,
                        state->centroids + c2 * vect_dim,
                        vect_dim);
                if (min_tmp < state->s[c1])
                    state->s[c1] = min_tmp;
                if (min_tmp < state->s[c2])
                    state->s[c2] = min_tmp;
            }
        }


        // Apply k-means algorithm for each vector
        for (unsigned i = 0; i < vect_count; i++)
        {
            float m = fmax(state->s[state->assignment[i]] / 2,
                    state->lower_bounds[i]);

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
                    point_all_ctrs(vectors, i, state);
                    unsigned char curr_assignment = state->assignment[i];

                    // Update centroids
                    if (old_assignment != curr_assignment)
                    {
                        change_cluster++;
                        state->centroids_count[old_assignment]--;
                        state->centroids_count[curr_assignment]++;
                    }
                }
            }
        }

        // Update centroids
        for (unsigned i = 0; i < K * vect_dim; i++)
            state->centroids_next[i] = 0;
        for (unsigned i = 0; i < vect_count; i++)
        {
            for (unsigned d = 0; d < vect_dim; d++)
            {
                unsigned index = state->assignment[i] * vect_dim + d;
                state->centroids_next[index] += vectors[i * vect_dim + d];
            }
        }
        float max_moved = move_centers(state);
        update_bounds(state, max_moved);

        for (unsigned j = 0; j < state->K; j++)
            state->s[j] = FLT_MAX;
        memcpy(state->centroids, state->centroids_next,
                sizeof(float) * vect_dim * K);

        // Print debug
        double t2 = omp_get_wtime();
        print_result(iter, t2 - t1, change_cluster);
        iter += 1;
    }

    // Free state memory
    unsigned char *res = state->assignment;
    free(state->centroids);
    free(state->centroids_next);
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
    //double min_err = atof(argv[3]);
    unsigned K = atoi(argv[1]);
    unsigned vect_dim = atoi(argv[4]);
    unsigned vect_count = atoi(argv[5]);
    char *input = argv[6];
    char *output = argv[7];

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
