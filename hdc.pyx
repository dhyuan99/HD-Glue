import numpy as np
cimport numpy as np
import cython
import time
from cython.parallel import prange

from libc.stdlib cimport rand, srand
cdef extern from "limits.h":
    int INT_MAX
srand(time.time())

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def gen_levels(int dim, int n_levels):
    L = np.zeros(shape=(dim, n_levels), dtype=np.short)
    L[:, 0] = np.random.randint(2, size=[dim], dtype=np.short)
    cdef short[:,:] L_view = L
    cdef Py_ssize_t d, j
    cdef double prob = 1 / <double> n_levels
    for d in range(dim):
        for j in range(1, n_levels):
            if rand() / <double> INT_MAX < prob:
                L_view[d, j] = 1 - L_view[d, j-1]
            else:
                L_view[d, j] = L_view[d, j-1]
    return L

class RecordBased:
    def __init__(self, dim, n_bundles, n_levels, low, high):
        self.L = gen_levels(dim, n_levels+1)
        self.ID = np.random.randint(2, size=[dim, n_bundles], dtype=np.short)

        self.dim = dim
        self.n_bundles = n_bundles
        self.n_levels = n_levels
        self.low = low
        self.high = high

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def encode(self, x):
        # x is of shape (batch_size, n_bundles)
        assert x.shape[1] == self.n_bundles

        cdef int batch_size = x.shape[0]
        cdef int n_bundles = self.n_bundles
        cdef int dim = self.dim

        out = np.zeros(shape=[batch_size, dim], dtype=np.short)

        cdef short [:,:] x_view = ((x - self.low) / (self.high - self.low) * self.n_levels).astype(np.short)
        cdef short [:,:] out_view = out

        cdef short[:,:] L_view = self.L
        cdef short[:,:] ID_view = self.ID

        cdef Py_ssize_t xi, d, j

        cdef int[:] count_view = np.zeros([batch_size], dtype=np.intc)
        
        for xi in prange(batch_size, nogil=True):
            for d in range(dim):
                count_view[xi] = 0
                for j in range(n_bundles):
                    count_view[xi] += L_view[d, x_view[xi, j]] ^ ID_view[d, j]
                if count_view[xi] * 2 > n_bundles:
                    out_view[xi, d] = 1

        return out

class NGramBased:
    def __init__(self, dim, n_bundles, n_levels, low, high):
        self.L = gen_levels(dim, n_levels+1)
        self.P = np.stack([np.random.permutation(dim) for _ in range(n_bundles)], axis=1)
        self.P = self.P.astype(np.intc)

        self.dim = dim
        self.n_bundles = n_bundles
        self.n_levels = n_levels
        self.low = low
        self.high = high

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def encode(self, x):
        # x is of shape (batch_size, n_bundles)
        assert x.shape[1] == self.n_bundles

        cdef int batch_size = x.shape[0]
        cdef int n_bundles = self.n_bundles
        cdef int dim = self.dim
        cdef int tmp

        out = np.zeros(shape=[batch_size, dim], dtype=np.short)

        cdef short [:,:] x_view = ((x - self.low) / (self.high - self.low) * self.n_levels).astype(np.short)
        cdef short [:,:] out_view = out

        cdef short[:,:] L_view = self.L
        cdef int[:,:] P_view = self.P

        cdef Py_ssize_t xi, d, j

        for xi in range(batch_size):
            for d in range(dim):
                tmp = 0
                for j in range(n_bundles):
                    tmp += L_view[P_view[d, j], x_view[xi, j]]
                if tmp * 2 > n_bundles:
                    out_view[xi, d] = 1

        return out

class DistancePreserve:
    def __init__(self, n_bundles, n_levels, low, high):
        self.n_bundles = n_bundles
        self.n_levels = n_levels
        self.dim = n_bundles * (n_levels + 1)
        self.low = low
        self.high = high

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def encode(self, x):
        # x is of shape (batch_size, n_bundles)
        cdef int batch_size = x.shape[0]
        cdef int n_bundles = self.n_bundles
        cdef int n_levels = self.n_levels
        
        out = np.zeros(shape=[batch_size, n_bundles * (n_levels + 1)], dtype=np.short)
        cdef short[:,:] x_view = np.round((x - self.low) / (self.high - self.low) * self.n_levels).astype(np.short)
        cdef short[:,:] out_view = out

        cdef Py_ssize_t xi, j, k

        for xi in range(batch_size):
            for j in range(n_bundles):
                for k in range(j * (n_levels + 1), j * (n_levels + 1) + x_view[xi, j]):
                    out_view[xi, k] = 1

        return out

class Aggregator:
    def __init__(self, n_models, dim):
        self.n_models = n_models
        self.dim = dim
        # P is of shape (dim, n_models)
        self.P = np.stack([np.random.permutation(dim) for _ in range(n_models)], axis=1)
        self.P = self.P.astype(np.intc)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def aggregate(self, X, n=None):

        if n is None:
            n = self.n_models

        assert X.shape[1] == self.dim

        # X is of shape (batch_size, dim, n_models)
        if n == 1:
            return X[:, :, 0]
            
        out = np.zeros(shape=X.shape[0:2], dtype=np.short)
        cdef short[:,:] out_view = out
        cdef short[:,:,:] X_view = X
        cdef int[:,:] P_view = self.P

        cdef int batch_size = X.shape[0]
        cdef int dim = X.shape[1]
        cdef int n_ = n

        cdef int tmp

        cdef Py_ssize_t xi, d, j
        for xi in range(batch_size):
            for d in range(dim):
                tmp = 0
                for j in range(n_):
                    tmp += X_view[xi, P_view[d, j], j]
                if tmp * 2 > n_:
                    out_view[xi, d] = 1
        return out


class VectorClassifier:
    def __init__(self, n_classes, dim):
        self.n_classes = n_classes
        self.dim = dim

        self.label = np.zeros(shape=[dim, n_classes], dtype=np.short)
        self.count = np.zeros(shape=[dim, n_classes], dtype=np.short)
 
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def construct_memory(self, x, y):
        # x is of shape (batch_size, dim)
        # y is of shape (batch_size)
        assert len(x.shape) == 2, "The shape of the input should be (batch_size, dim)."
        assert x.shape[1] == self.dim, "The shape of the input should be (batch_size, dim)."
        assert x.shape[0] == y.shape[0], "The input and target have different sample size."
        assert np.all(np.logical_and(y >= 0, y < self.n_classes)), "each entry of y must be in [0, n_classes)"

        cdef int batch_size = x.shape[0]
        cdef int n_classes = self.n_classes
        cdef int dim = self.dim

        x = x.astype(np.short)
        y = y.astype(np.short)

        cdef short[:,:] x_view = x
        cdef short[:] y_view = y
        cdef short[:,:] label_view = self.label
        cdef short[:,:] count_view = self.count

        cdef int one, count
        cdef Py_ssize_t xi, i, j, k, d

        for xi in range(batch_size):
            for d in range(dim):
                label_view[d, y_view[xi]] += x_view[xi, d]
                count_view[d, y_view[xi]] += 1

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.cdivision(True)
    def optimize_label(self, train_X, correct_Y, wrong_Y):
        # train_X is of shape (batch_size, dim)
        # correct_Y is of shape (batch_size)
        # wrong_Y is of shape (batch_size)
        
        assert train_X.shape[1] == self.dim
        assert train_X.shape[0] == len(correct_Y) and train_X.shape[0] == len(wrong_Y)

        label = (self.label * 2 > self.count).astype(int)
        dist = np.sum(np.abs(np.expand_dims(train_X, axis=2) - np.expand_dims(label, axis=0)), axis=1)
        dist = (dist[np.arange(len(correct_Y), dtype=int), correct_Y] - dist[np.arange(len(correct_Y), dtype=int), wrong_Y]) / self.dim

        cdef int dim = self.dim
        cdef int batch_size = train_X.shape[0]
        cdef double[:] dist_view = dist
        cdef short[:,:] label_view = self.label
        cdef short[:,:] count_view = self.count
        cdef short[:,:] X_view = train_X
        cdef short[:] correct_Y_view = correct_Y.astype(np.short)
        cdef short[:] wrong_Y_view = wrong_Y.astype(np.short)

        for xi in range(batch_size):
            for d in range(dim):
                if rand() / <double> INT_MAX < dist_view[xi]:
                    label_view[d, correct_Y_view[xi]] += X_view[xi, d]
                    count_view[d, correct_Y_view[xi]] += 1
                    label_view[d, wrong_Y_view[xi]] -= X_view[xi, d]
                    count_view[d, wrong_Y_view[xi]] -= 1

    def predict(self, x, y=None):
        # x is of shape (batch_size, dim)
        assert x.shape[1] == self.dim
        label = (self.label * 2 > self.count).astype(np.short)

        cdef int batch_size = x.shape[0]
        cdef int dim = self.dim
        cdef int n_classes = self.n_classes

        dist = np.zeros([batch_size, n_classes], dtype=np.intc)
        cdef short[:,:] x_view = x
        cdef short[:,:] label_view = label
        cdef int[:,:] dist_view = dist

        cdef Py_ssize_t xi, d, k

        for xi in range(batch_size):
            for d in range(dim):
                for k in range(n_classes):
                    dist_view[xi, k] += x_view[xi, d] ^ label_view[d, k]
        
        yhat = np.argmin(dist, axis=1)

        if y is None:
            return yhat
        else:
            wrong_idx = np.where(yhat != y)[0]
            acc = 1 - float(len(wrong_idx)) / len(y)
            return yhat, wrong_idx, acc
    
    def get_model(self, clf, w):
        assert self.n_classes == clf.n_classes and self.dim == clf.dim
        self.label = self.label + w * clf.label
        self.count = self.count + w * clf. count

    def compress_memory(self, factor):
        self.label = self.label * factor
        self.count = self.count * factor