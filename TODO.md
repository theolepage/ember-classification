# To-Do

## K-means

- [ ] Refactor / clean code
- [ ] Create a cache for distances
- [ ] For c random init: see for use of "rand() % K", rather than "rand() / RAND_MAX * K"
- [ ] Make our k-means work
- [ ] Refactor / clean / doc code
- [ ] Use OpenMP and SMID

## Python part

- [ ] Start working on Python part

# Draft

## Llyod's algorithm (currently implemented)

Init: random centers for clusters
For each vector: b(x) = closest cluster
Update cluster centers: for each cluster = mean of all its vectors
Repeat until centers stop changing

## Hamerly's algorithm

Aim: skipping distance calculations with lower bounds

# First filter

Un : the upper bound distance of the vector n to its assigned centroid
Ln : the lower bound distance of the vector n to its second nearest centroid.

If Un < Ln => distance computaions can be skipped

# Second filter

Uses the distance of the nearest centroid to the assigned centroid.

Suppose vector n is assigned to cluster c1, and nearest centroid to c1 is c2.
=> 
