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
** \param disable_sqrt If set to 1, return the distance without applying sqrt.
** \return The distance between vec1 and vec2.
**/
double distance(float *vec1, float *vec2, unsigned dim, int disable_sqrt)
{
    double dist = 0;
#pragma omp simd reduction(+: dist)
    for(unsigned i = 0; i < dim; ++i)
    {
        double d = *vec1 - *vec2;
        dist += d * d;
        vec1++;
        vec2++;
    }
    return disable_sqrt ? dist : sqrt(dist);
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
** \param thread Data from current thread.
** \param state A pointer to the struct representing algorithm's state.
** \param change_cluster Whether or not the cluster has been changed.
**/
static void update_assignment(
        float *vectors,
        unsigned i,
        struct kmeans_thread *thread,
        struct kmeans_state *state,
        unsigned *change_cluster
)
{
    float min_dist_first = FLT_MAX;
    float min_dist_second = FLT_MAX;
    unsigned char min_dist_index = 0;

    // Find the two closest centroids
    for (unsigned c = 0; c < state->K; c++)
    {
        float tmp_dist = distance(vectors + i * state->vect_dim,
                state->centroids + c * state->vect_dim,
                state->vect_dim, 1);

        if (tmp_dist < min_dist_first)
        {
            min_dist_second = min_dist_first;
            min_dist_first = tmp_dist;
            min_dist_index = c;
        }
        else if (tmp_dist < min_dist_second)
            min_dist_second = tmp_dist;
    }

    // Update vector i assignment
    // min_dist_index = new assignment
    if (min_dist_index != state->assignment[i])
    {
        *change_cluster = 0;
        thread->change_cluster_t[min_dist_index] = 1;
        thread->change_cluster_t[state->assignment[i]] = 1;
        thread->centroids_count_t[state->assignment[i]]--;
        thread->centroids_count_t[min_dist_index]++;

        // Update centroids
        for (unsigned d = 0; d < state->vect_dim; d++)
        {
            unsigned old_index = state->assignment[i] * state->vect_dim + d;
            unsigned new_index = min_dist_index * state->vect_dim + d;
            float val = vectors[i * state->vect_dim + d];
            thread->centroids_sum_t[old_index] -= val;
            thread->centroids_sum_t[new_index] += val;
        }

        state->assignment[i] = min_dist_index;
        state->upper_bounds[i] = sqrt(min_dist_first);
    }

    state->lower_bounds[i] = sqrt(min_dist_second);
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
        // Make a copy of old centroid
        float *old_centroid = calloc(state->vect_dim, sizeof(float));
        for (unsigned d = 0; d < state->vect_dim; d++)
            old_centroid[d] = state->centroids[c * state->vect_dim + d];

        // Compute new centroid centers
        for (unsigned d = 0; d < state->vect_dim; d++)
        {
            float count = state->centroids_count[c];
            unsigned index = c * state->vect_dim + d;
            float res = count ? state->centroids_sum[index] / count : 0;
            state->centroids[c * state->vect_dim + d] = res;
        }

        // Compute distance between old and new centroid
        state->p[c] = distance(old_centroid,
                state->centroids + c * state->vect_dim,
                state->vect_dim, 0);

        // Update max_moved
        if (state->p[c] > max_moved)
            max_moved = state->p[c];

        free(old_centroid);
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
static struct kmeans_state *init_state(
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
    state->centroids_sum = calloc(K * vect_dim, sizeof(float));
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
static void free_state(struct kmeans_state *state)
{
    free(state->centroids);
    free(state->centroids_sum);
    free(state->centroids_count);
    free(state->upper_bounds);
    free(state->lower_bounds);
    free(state->p);
    free(state->s);
    free(state);
}

/**
** \brief Main loop of k-means algorithm on each vector.
** \param vectors The feature vectors data.
** \param state A pointer to the struct representing algorithm's state.
** \return Whether or not at least a cluster has changed.
**/
static unsigned kmeans_aux(
        float *vectors,
        struct kmeans_state *state
)
{
    unsigned change_cluster = 1;

#pragma omp parallel reduction(&: change_cluster)
    {
        struct kmeans_thread *t = calloc(1, sizeof(struct kmeans_thread));
        t->change_cluster_t = calloc(state->K, sizeof(unsigned));
        t->centroids_sum_t = calloc(state->K * state->vect_dim, sizeof(float));
        t->centroids_count_t = calloc(state->K, sizeof(unsigned));

        // Apply k-means algorithm for each vector
#pragma omp for schedule(guided, 10)
        for (unsigned i = 0; i < state->vect_count; i++)
        {
            float m = fmax(state->s[state->assignment[i]] / 2,
                    state->lower_bounds[i]);

            // First bound test
            if (state->upper_bounds[i] > m)
            {
                // Tighten upper bound
                float *v1 = vectors + i * state->vect_dim;
                float *v2 = state->centroids + state->assignment[i]
                        * state->vect_dim;
                state->upper_bounds[i] = distance(v1, v2, state->vect_dim, 0);

                // Second bound test
                if (state->upper_bounds[i] > m)
                {
                    update_assignment(vectors, i, t, state, &change_cluster);
                }
            }
        }

        // Reduction
        for (unsigned c = 0; c < state->K; c++)
        {
            if (t->change_cluster_t[c])
            {
#pragma omp atomic
                state->centroids_count[c] += t->centroids_count_t[c];
                for (unsigned d = 0; d < state->vect_dim; d++)
                {
                    float value = t->centroids_sum_t[c * state->vect_dim + d];
#pragma omp atomic
                    state->centroids_sum[c * state->vect_dim + d] += value;
                }
            }
        }

        // Free thread memory
        free(t->change_cluster_t);
        free(t->centroids_sum_t);
        free(t->centroids_count_t);
        free(t);
    }

    return change_cluster;
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
    //init_random_centroids(vectors, state);
    kmeanspp(vectors, state);

#pragma omp parallel
    {
        float *sum_thread = calloc(vect_dim, sizeof(float));
#pragma omp for
        for (unsigned i = 0; i < vect_count; i++)
            for (unsigned d = 0; d < vect_dim; d++)
               sum_thread[d] += vectors[i * vect_dim + d];
        for (unsigned d = 0; d < vect_dim; d++)
#pragma omp atomic
            state->centroids_sum[d] += sum_thread[d];
        free(sum_thread);
    }
    for (unsigned i = 0; i < vect_count; i++)
        state->upper_bounds[i] = FLT_MAX;
    for (unsigned c = 0; c < K; c++)
        state->s[c] = FLT_MAX;

    unsigned iter = 0;
    unsigned change_cluster = 0;

    // Main loop
    while (iter < max_iter && !change_cluster)
    {
        double t1 = omp_get_wtime();

        // Update shortest distance between each two cluster
        for (unsigned c1 = 0; c1 < K; c1++)
        {
            for (unsigned c2 = c1 + 1; c2 < K; c2++)
            {
                float min_tmp = distance(state->centroids + c1 * vect_dim,
                        state->centroids + c2 * vect_dim,
                        vect_dim, 0);
                if (min_tmp < state->s[c1])
                    state->s[c1] = min_tmp;
                if (min_tmp < state->s[c2])
                    state->s[c2] = min_tmp;
            }
        }

        // Main loop on all vectors
        change_cluster = kmeans_aux(vectors, state);

        // Update centroids and bounds
        float max_moved = move_centers(state);

#pragma omp parallel for
        for (unsigned i = 0; i < state->vect_count; i++)
        {
            state->upper_bounds[i] += state->p[state->assignment[i]];
            state->lower_bounds[i] -= max_moved;
        }

        // Reset / prepare for next iteration
        for (unsigned j = 0; j < state->K; j++)
            state->s[j] = FLT_MAX;
        iter++;

        // Print debug
        double t2 = omp_get_wtime();
        print_result(iter, t2 - t1, change_cluster);
    }

    // Free state memory and return assignment
    unsigned char *res = state->assignment;
    free_state(state);
    return res;
}
