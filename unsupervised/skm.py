import random
import logging as log
import numpy as np
import sklearn.manifold
import sklearn.metrics.pairwise
from scipy.spatial.distance import cdist
from scipy.sparse import issparse

# --------------------------------------------------------------

class SphericalKMeans:
    """
    Basic implementation of Spherical K-Means (SKM), backed by a generalized K-Means implementation that
    uses cosine distances. Term rankings are produced based on cluster centroid values.
    """

    def __init__( self, max_iters = 100 ):
        self.max_iters = max_iters
        self.partition = None
        self.centroids = None

    def apply( self, X, k = 2 ):
        # we use prototype initialization here: randomly select k rows from the matrix
        init_centroid_indices = random.sample( xrange( X.shape[0] ), k )
        init_centroids = X[init_centroid_indices]
        # actually apply the custom k-means implementation
        self.centroids, self.partition, _ = kmeans( X, init_centroids, delta=0.001, maxiter=self.max_iters, metric="cosine", verbose=0 )
 
    def generate_partition( self ):
        if self.partition is None:
            raise ValueError("No results for previous run available")
        return self.partition

    def rank_terms( self, topic_index, top = -1 ):
        """
        Return the top ranked terms for the specified topic, using the centroids from the last run.
        """
        if self.centroids is None:
            raise ValueError("No results for previous run available")
        # ranks the topics, then reverse afterwards
        top_indices = np.argsort( self.centroids[topic_index,:] ).tolist()[0][::-1]
        # truncate if necessary
        if top < 1 or top > len(top_indices):
            return top_indices
        return top_indices[0:top]

# --------------------------------------------------------------

class SpectralSphericalKMeans( SphericalKMeans ):
    def __init__( self, max_iters = 100 ):
        SphericalKMeans.__init__( self, max_iters )

    def apply( self, X, k = 2 ):
        # Build Affinity Matrix
        log.debug( "Computing similarity matrix ..." )
        # TODO: can we assume rows are unit length?
        S = sklearn.metrics.pairwise.linear_kernel(X)
        # set diagonal to zero
        np.fill_diagonal( S, 0 )        
        log.debug( "Constructing spectral embedding ..." )
        E = sklearn.manifold.spectral_embedding(S, n_components = k, eigen_solver = 'arpack', drop_first = True )
        log.debug( "Constructed embedding of size", E.shape )
        # actually apply the custom k-means implementation
        init_centroid_indices = random.sample( xrange( E.shape[0] ), k )
        init_centroids = E[init_centroid_indices]        
        _, self.partition, _ = kmeans( E, init_centroids, delta=0.001, maxiter=self.max_iters, metric="cosine", verbose=0 )
        # Create empty cluster x term matrix, and then populate it with centroids from the original space
        self.centroids = np.matrix( np.zeros( (k, X.shape[1]) ) )
        for jc in range(k): 
            c = np.where( self.partition == jc )[0]
            if len(c) > 0:
                self.centroids[jc] = X[c].mean( axis=0 )        

# --------------------------------------------------------------
# Implementation of generalized k-means originally from
# http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikits-learn-k-means

def kmeans( X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 
        s running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    return centres, xtoc, distances

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d

