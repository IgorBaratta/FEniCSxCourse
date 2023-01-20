# Calculate number of vertices, edges, facets, and cells for any given
# level of refinement
def num_entities(i, j, k, num_refine):
    nv = (i + 1) * (j + 1) * (k + 1)
    ne = 0
    nc = (i * j * k) * 6
    earr = [1, 3, 7]
    farr = [2, 12]
    for r in range(num_refine):
        ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i)
        + earr[2] * i * j * k
        nv += ne
        nc *= 8
        earr[0] *= 2
        earr[1] *= 4
        earr[2] *= 8
        farr[0] *= 4
        farr[1] *= 8
    ne = earr[0] * (i + j + k) + earr[1] * \
        (i * j + j * k + k * i) + earr[2] * i * j * k
    nf = farr[0] * (i * j + j * k + k * i) + farr[1] * i * j * k

    return (nv, ne, nf, nc)
