#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#include "kmeans.h"

void kmeanspp(float *vectors, struct kmeans_state *state)
{
    unsigned dim = state->vect_dim;

    // Choose randomly the fist center
    unsigned new_centroids = rand() / (RAND_MAX + 1.) * state->vect_count;
    float *point = vectors + new_centroids * dim;
    for (unsigned i = 0; i < dim; i++)
        state->centroids[i] = point[i];

    // Choose the furthest centers in consequence
    for (unsigned center = 1; center < state->K; center++)
    {
        double max_dist = 0;
        unsigned max_dist_index = 0;

        // For each vector of the data set
        for (unsigned i = 0; i < state->vect_count; i++)
        {
            float *point = vectors + i * dim;
            double min_dist = DBL_MAX;

            // Find the closest distance between this vector and a center
            for (unsigned j = 0; j < center; j++)
            {
                float *center_compared = state->centroids + j * dim;
                double dist = distance(point, center_compared, dim);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    if (j != state->assignment[i])
                    {
                        //state->centroids_count[state->assignment[i]]--;
                        state->assignment[i] = j;
                        //state->centroids_count[state->assignment[i]]++;
                    }
                }
            }
            if (max_dist < min_dist)
            {
                max_dist = min_dist;
                max_dist_index = i;
            }
        }

        // Copy the new center into state->centroids
        for (unsigned i = 0; i < dim; i++)
        {
            float *point = vectors + max_dist_index * dim;
            state->centroids[center * dim + i] = point[i];
        }
    }

    for (unsigned i = 0; i < state->vect_count; i++)
        state->assignment[i] = 0;
}
