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
        err(1, "Error while opening %s", fileName);

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

static inline void print_result(int iter, double time, unsigned change)
{
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"change\": \"%d\"}\n", iter, time, change);
    else
        printf("Iteration: %d, Time: %lf, Change: %d\n", iter, time, change);
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
        struct kmeans_state *state
)
{
    // Update assignment
    float min_dist = FLT_MAX;
    float min_dist_p = FLT_MAX;

    unsigned char min_dist_index = 0;
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
    state->assignment[i] = min_dist_index;

    // Update upper bound
    state->upper_bounds[i] = min_dist;

    // Update lower bound
    state->lower_bounds[i] = min_dist_p;
}

static void move_centers(struct kmeans_state *state)
{
    // Make a copy of current centroid
    float *old_centroid = calloc(state->vect_dim, sizeof(float));
    for (unsigned c = 0; c < state->K; c++)
    {
        // Compute new centroid
        for (unsigned d = 0; d < state->vect_dim; d++)
        {
            old_centroid[d] = state->centroids[c * state->vect_dim + d];
            unsigned count = state->centroids_count[c];
            float value = state->centroids_sum[c * state->vect_dim + d] / count;
            state->centroids[c * state->vect_dim + d] = value;
        }

        // Store difference between old and new centroid
        state->p[c] = distance(old_centroid,
                state->centroids + c * state->vect_dim,
                state->vect_dim);
    }
    free(old_centroid);
}

static void update_bounds(struct kmeans_state *state)
{
    float max_shift = 0;
    float max_shift_p = 0;
    // r
    unsigned char max_shift_index = 0;
    // r'
    unsigned char max_shift_p_index = 0;

    for (unsigned c = 0; c < state->K; c++)
    {
        float tmp_shift = state->p[c];
        if (tmp_shift > max_shift)
        {
            max_shift_p_index = max_shift_index;
            max_shift_index = c;
            max_shift_p = max_shift;
            max_shift = tmp_shift;
        }
        else if (tmp_shift > max_shift_p)
        {
            max_shift_p = tmp_shift;
            max_shift_p_index = c;
        }
    }

    for (unsigned i = 0; i < state->vect_count; i++)
    {
        state->upper_bounds[i] += state->p[state->assignment[i]];
        if (max_shift_index == state->assignment[i])
            state->lower_bounds[i] -= state->p[max_shift_p_index];
        else
            state->lower_bounds[i] -= state->p[max_shift_index];
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
    min_err += 1;
    // Init state and allocate memory
    struct kmeans_state *state = calloc(1, sizeof(struct kmeans_state));
    state->K = K;
    state->vect_count = vect_count;
    state->vect_dim = vect_dim;
    state->assignment = calloc(vect_count, sizeof(unsigned char)); // a
    state->centroids = calloc(K * vect_dim, sizeof(float)); // c
    state->centroids_sum = calloc(K * vect_dim, sizeof(float)); // c'
    state->centroids_count = calloc(K, sizeof(unsigned)); // q
    state->upper_bounds = malloc(vect_count * sizeof(float)); // u
    state->lower_bounds = calloc(vect_count, sizeof(float)); // l
    state->p = calloc(K, sizeof(float)); // p
    state->s = calloc(K, sizeof(float)); // s

    unsigned iter = 0;
    unsigned change_cluster = 1;

    // Init randomly the centers.
    int *centroids_index = calloc(state->K, sizeof(int));
    for (int i = 0; i < state->K; i++)
    {
        state->s[i] = FLT_MAX;
        centroids_index[i] = rand() / (RAND_MAX + 1.) * vect_count;
        // Check that the given index is unique in the array.
        for (int j = 0; j < i; j++)
        {
            // If the index is already used by another centroids,
            // choose another value (restart the loop from i)
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

    // Initialize (Algorithm 2)
    for (unsigned i = 0; i < vect_count; i++)
    {
        state->upper_bounds[i] = FLT_MAX;
        point_all_ctrs(vectors, i, state);
        unsigned char c = state->assignment[i];
        state->centroids_count[c]++;

        for (unsigned d = 0; d < vect_dim; d++)
            state->centroids_sum[c * vect_dim + d] += vectors[i * vect_dim + d];
    }

    // Main loop
    while ((iter < max_iter))
    {
        if (iter)
            change_cluster = 0;
        double t1 = omp_get_wtime();
        // Update s
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
                    point_all_ctrs(vectors, i, state);
                    unsigned char curr_assignment = state->assignment[i];

                    // Update centroids
                    if (old_assignment != curr_assignment)
                    {
                        change_cluster++;
                        state->centroids_count[old_assignment]--;
                        state->centroids_count[curr_assignment]++;
                        for (unsigned d = 0; d < vect_dim; d++)
                        {
                            float value = vectors[i * vect_dim + d];
                            state->centroids_sum[old_assignment * vect_dim + d] -= value;
                            state->centroids_sum[curr_assignment * vect_dim + d] += value;
                        }
                    }
                }
            }
        }
        for (unsigned j = 0; j < state->K; j++)
            state->s[j] = FLT_MAX;
        double t2 = omp_get_wtime();
        print_result(iter, t2 - t1, change_cluster);
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
