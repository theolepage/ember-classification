#ifndef KMEANS_H
#define KMEANS_H

struct kmeans_state
{
    unsigned vect_count;
    unsigned vect_dim;
    unsigned char K;

    unsigned char *assignment;
    float *centroids;
    float *centroids_next;
    unsigned *centroids_count;

    float *upper_bounds;
    float *lower_bounds;
    float *p;
    float *s;
};

#endif /* ! KMEANS_H */
