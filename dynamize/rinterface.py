# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy
import rpy2.robjects as robj


VECTOR_TYPES = {
    numpy.float64: robj.FloatVector,
    numpy.float32: robj.FloatVector,
    numpy.float: robj.FloatVector,
    numpy.int: robj.IntVector,
    numpy.int32: robj.IntVector,
    numpy.int64: robj.IntVector,
    numpy.object_: robj.StrVector,
    numpy.str: robj.StrVector,
    numpy.bool: robj.BoolVector,
}

NA_TYPES = {
    numpy.float64: robj.NA_Real,
    numpy.float32: robj.NA_Real,
    numpy.float: robj.NA_Real,
    numpy.int: robj.NA_Integer,
    numpy.int32: robj.NA_Integer,
    numpy.int64: robj.NA_Integer,
    numpy.object_: robj.NA_Character,
    numpy.str: robj.NA_Character,
    numpy.bool: robj.NA_Logical,
}

_VECTOR_TYPES_BY_NAME = dict((str(t), v) for (t, v) in VECTOR_TYPES.items())


def _fix_get_vector_type(t):
    return _VECTOR_TYPES_BY_NAME[str(t)]


def convert_to_r_posixct(value):
    """Should convert `numpy.datetime64` objects to R dates, but since
    we don't use such objects we just don't care :)

    NOTE: If you really need it, copy it from `pandas.rpy.common`
    """
    raise NotImplementedError


def isnull(obj):
    """Stupidly oversimplified copy of `pandas.isnull` that should
    apply to all our inputs, but probably doesn't.
    """
    return numpy.isnan(obj)


def notnull(obj):
    """Stupidly oversimplified copy of `pandas.notnull` that should
    apply to all our inputs, but probably doesn't.
    """
    return not isnull(obj)


def convert_to_r_matrix(matrix):
    """Convert a 2x2 numpy matrix to a R matrix.

    It works by iterating and converting the given *matrix* row by row.
    """
    rbind = robj.baseenv.get('rbind')

    #robj.r('library("Matrix")')
    #robj.r('library("glmnet")')
    #rmatrix = robj.r('Matrix(NA, 0, {0})'.format(matrix.shape[1]))
    rmatrix = robj.r('matrix(NA, 0, {0})'.format(matrix.shape[1]))

    for row in matrix:
    #if True:
        #row = matrix

        # FIXME I'm so stupid I don't know how to create a matrix of
        # arrays. All matrices I create with `numpy` are matrices of matrices.
        # Damn.
        if hasattr(row, 'toarray'):
            value = row.toarray()
        else:
            value = row.A

        value = value.ravel()
        value_type = value.dtype.type

        # XXX Will this really avoid memory waste?
        del row

        if value_type == numpy.datetime64:
            value = convert_to_r_posixct(value)
        else:
            value = [
                item if notnull(item) else NA_TYPES[value_type]
                for item in value
            ]

            # FIXME XXX Fuck this. At this point for god only knows what
            # reason, `value_type` will NOT BE IN THE VECTOR_TYPE keys! Even
            # if its the same type, say, numpy.int64, it'll be an numpy.int64
            # whose id() is different from the version imported from numpy.
            # FUCK THIS.
            #value = VECTOR_TYPES[value_type](value)
            value = _fix_get_vector_type(value_type)(value)

        rmatrix = rbind(rmatrix, value)

    return rmatrix
