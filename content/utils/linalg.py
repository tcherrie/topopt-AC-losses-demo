"""Linear algebra utilities

Provide utilities related to linear algebra

Functions defined here:
- sparse
- vec
- split_mat
- split_vec


A Large Language Model (GPT-5.5 from Open AI, free version) was used to help with the code and generate
the docstrings of the functions. The authors have written the initial code, carefully checked and post-edited
the content of this file, and take full responsability of its content.
This software is provided "as is" without warranty of any kind, and can be used, shared ad modified under the terms of GNU LGPL license.
"""

#%% Metadata

__author__ = "Théodore Cherrière"
__copyright__ = "Copyright 2026, CentraleSupélec, SAFRAN"
__credits__ = ["Théodore Cherrière", "Alexis Pons", "Guillaume Krebs",
                    "Adrien Mercier", "Loucif Benmamas", "Sulivan Küttler"]
__license__ = "GNU LGPL"
__version__ = "0.2"
__maintainer__ = "Théodore Cherrière"
__email__ = "theodore.cherriere@centralesupelec.fr"
__status__ = "Development"

#%% Import

from scipy.sparse import csc_matrix

#%% Vector and matrices

def sparse(bf, freedofs_rows = None, freedofs_cols=None):
    """
    Convert a bilinear form into a sparse CSC matrix, optionally restricting
    it to selected rows and columns.

    Parameters
    ----------
    bf : object
        Bilinear form providing a ``COO()`` method that returns row indices,
        column indices, and corresponding matrix values.
    freedofs_rows : array-like, optional
        Row indices to retain. If ``None``, all rows are retained.
    freedofs_cols : array-like, optional
        Column indices to retain. If ``None``, all columns are retained.

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse matrix constructed from the COO representation of ``bf``,
        optionally restricted to the specified rows and columns.
    """
    r,c,vals  = bf.COO()
    K = csc_matrix((vals,(r,c)))
    if freedofs_rows is not None:
        K = K[freedofs_rows,:]
    if freedofs_cols is not None:
        K = K[:,freedofs_cols]
    return K

def vec(lf, freedofs = None):
    """
    Convert a linear form into a column vector, optionally restricting it
    to selected degrees of freedom.

    Parameters
    ----------
    lf : object
        Linear form providing an ``Evaluate()`` method whose result can be
        converted to a NumPy array via ``FV().NumPy()``.
    freedofs : array-like, optional
        Indices of the degrees of freedom to retain. If ``None``, all
        degrees of freedom are retained.

    Returns
    -------
    numpy.ndarray
        Column vector containing the values of the linear form, optionally
        restricted to the specified degrees of freedom.
    """
    f = lf.Evaluate().FV().NumPy().reshape(-1,1)
    if freedofs is not None:
        f = f[freedofs]
    return f

#%% Condensation

def split_mat(K,
               freedofs,
               excluded_dofs = None,
               )-> tuple:
    """
    Split a matrix into submatrices corresponding to free and excluded
    degrees of freedom.

    Parameters
    ----------
    K : object
        Matrix or bilinear form to split. If not already a sparse matrix,
        it is converted using ``sparse()``.
    freedofs : array-like
        Boolean mask or indices identifying the free degrees of freedom.
    excluded_dofs : array-like, optional
        Boolean mask or indices identifying the excluded degrees of freedom.
        If ``None``, the complement of ``freedofs`` is used.

    Returns
    -------
    tuple
        Four sparse matrix blocks ``(A, B, C, D)`` corresponding to the
        partition

        ``K = [[A, B], [C, D]]``

        where ``A`` contains free-to-free entries, ``B`` free-to-excluded
        entries, ``C`` excluded-to-free entries, and ``D`` excluded-to-
        excluded entries.
    """
    if excluded_dofs is None:
        excluded_dofs = ~freedofs
        
    K = sparse(K)
    A = K[freedofs,:][:,freedofs]
    B = K[freedofs,:][:,excluded_dofs]
    C = K[excluded_dofs,:][:,freedofs]
    D = K[excluded_dofs,:][:,excluded_dofs]
    
    return A, B, C, D

def split_vec(F,
              freedofs,
              excluded_dofs = None,
              )-> tuple:
    """
    Split a vector into components corresponding to free and excluded
    degrees of freedom.

    Parameters
    ----------
    F : object
        Linear form to split. It is converted to a column vector using
        ``vec()``.
    freedofs : array-like
        Boolean mask or indices identifying the free degrees of freedom.
    excluded_dofs : array-like, optional
        Boolean mask or indices identifying the excluded degrees of freedom.
        If ``None``, the complement of ``freedofs`` is used.

    Returns
    -------
    tuple
        Two vectors ``(F1, F2)`` where ``F1`` contains the entries
        corresponding to the free degrees of freedom and ``F2`` contains
        the entries corresponding to the excluded degrees of freedom.
    """
    if excluded_dofs is None:
        excluded_dofs = ~freedofs
        
    F = vec(F)
    F1 = F[freedofs]
    F2 = F[excluded_dofs]
    
    return F1, F2
