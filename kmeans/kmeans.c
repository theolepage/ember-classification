#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <err.h>

#include "kmeans.h"

/**
** \brief Compute the euclidean distance between two vectors.
** \param vec1 The first vector.
** \param vec2 The second vector.
** \param dim The dimension of both vectors.
** \return The distance between vec1 and vec2.
**/
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
** \brief Initialize the kmeans_state struct.
** \param vect_count The number of feature vectors.
** \param vect_dim The number of features (dimension of vectors).
** \param K The number of clusters.
** \return state a pointer to the kmeans_state struct created.
**/
struct kmeans_state *init_state(
        unsigned vect_count,
        unsigned vect_dim,
        unsigned char K)
{
    struct kmeans_state *state = calloc(1, sizeof(struct kmeans_state));
    if (!state)
        errx(1, "Error while allocating memory.");
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
    return state;
}

/**
** \brief Free the allocated memory for state.
** \param state A pointer to the struct kmeans_state to free.
**/
void free_state(struct kmeans_state *state)
{
    free(state->centroids);
    free(state->centroids_next);
    free(state->centroids_count);
    free(state->upper_bounds);
    free(state->lower_bounds);
    free(state->p);
    free(state->s);
    free(state);
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
    // Initialize
    struct kmeans_state *state = init_state(vect_count, vect_dim, K);
    state->centroids_count[0] = vect_count;
    init_random_centroids(vectors, state);
    //kmeanspp(vectors, state);

    for (unsigned i = 0; i < vect_count; i++)
        state->upper_bounds[i] = FLT_MAX;
    for (unsigned c = 0; c < K; c++)
        state->s[c] = FLT_MAX;

    unsigned iter = 0;
    unsigned min_error = vect_count * 0.005;
    unsigned change_cluster = min_error + 1;

    // Main loop
    while (iter < max_iter && change_cluster > min_error)
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
                    // Update assignments
                    unsigned char old_assignment = state->assignment[i];
                    point_all_ctrs(vectors, i, state);
                    unsigned char curr_assignment = state->assignment[i];
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

        // Update centroids and bounds
        float max_moved = move_centers(state);
        for (unsigned i = 0; i < state->vect_count; i++)
        {
            state->upper_bounds[i] += state->p[state->assignment[i]];
            state->lower_bounds[i] -= max_moved;
        }

        // Reset / prepare for next iteration
        for (unsigned j = 0; j < state->K; j++)
            state->s[j] = FLT_MAX;
        memcpy(state->centroids, state->centroids_next,
                sizeof(float) * vect_dim * K);
        iter++;

        // Print debug
        double t2 = omp_get_wtime();
        print_result(iter, t2 - t1, change_cluster);
    }

    // Free state memory
    unsigned char *res = state->assignment;
    free_state(state);
    return res;
}
