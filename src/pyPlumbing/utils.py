from sage.all_cmdline import *   # import sage library

from functools import lru_cache, wraps
import numpy as np
import itertools

from .series import *

def convert_input_to_tuple(func):
    """
    Decorator to convert all list arguments to tuples to make them hashable.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert all list arguments to tuples to make them hashable
        args = tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args)
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

@convert_input_to_tuple
@lru_cache(maxsize=None) 
def cartan_matrix(type_rank):
    """
    Compute the Cartan matrix of given Lie group.

    INPUT:
    -   ``type_rank'' - [str,int]; Lie group identifier

    EXAMPLES::
        sage: cartan_matrix(["A",2])
        [ 2 -1]
        [-1  2]
        sage: cartan_matrix(["E",8])
        [ 2  0 -1  0  0  0  0  0]
        [ 0  2  0 -1  0  0  0  0]
        [-1  0  2 -1  0  0  0  0]
        [ 0 -1 -1  2 -1  0  0  0]
        [ 0  0  0 -1  2 -1  0  0]
        [ 0  0  0  0 -1  2 -1  0]
        [ 0  0  0  0  0 -1  2 -1]
        [ 0  0  0  0  0  0 -1  2]

    COMMENTS:
    Cartan matrices that are computed are stored in a dictionary _cartan_matrices
    for easy and quick access.
    """
    return CartanMatrix(type_rank)


@convert_input_to_tuple
@lru_cache(maxsize=None) 
def weyl_group(type_rank):
    """
    Compute the Weyl group of a given Lie group.

    INPUT:
    -   ``type_rank'' - [str,int]; Lie group identifier

    Examples::    
        sage: weyl_group(["A",2])
        [
        [1 0]  [-1  1]  [ 1  0]  [ 0 -1]  [-1  1]  [ 0 -1]
        [0 1], [ 0  1], [ 1 -1], [ 1 -1], [-1  0], [-1  0]
        ]
        sage: weyl_group(["A",3])
        [
        [1 0 0]  [-1  1  0]  [ 1  0  0]  [ 1  0  0]  [ 0 -1  1]  [-1  1  0]
        [0 1 0]  [ 0  1  0]  [ 1 -1  1]  [ 0  1  0]  [ 1 -1  1]  [-1  0  1]
        [0 0 1], [ 0  0  1], [ 0  0  1], [ 0  1 -1], [ 0  0  1], [ 0  0  1],

        [ 1  0  0]  [-1  1  0]  [ 1  0  0]  [ 0 -1  1]  [ 0  0 -1]  [-1  1  0]
        [ 1  0 -1]  [ 0  1  0]  [ 1 -1  1]  [-1  0  1]  [ 1  0 -1]  [-1  1 -1]
        [ 0  1 -1], [ 0  1 -1], [ 1 -1  0], [ 0  0  1], [ 0  1 -1], [ 0  1 -1],

        [ 1  0  0]  [ 0 -1  1]  [-1  1  0]  [ 0  0 -1]  [ 0  0 -1]  [ 0 -1  1]
        [ 1  0 -1]  [ 1 -1  1]  [-1  0  1]  [-1  1 -1]  [ 1  0 -1]  [ 0 -1  0]
        [ 1 -1  0], [ 1 -1  0], [-1  0  0], [ 0  1 -1], [ 1 -1  0], [ 1 -1  0],

        [-1  1  0]  [ 0 -1  1]  [ 0  0 -1]  [ 0  0 -1]  [ 0 -1  1]  [ 0  0 -1]
        [-1  1 -1]  [-1  0  1]  [ 0 -1  0]  [-1  1 -1]  [ 0 -1  0]  [ 0 -1  0]
        [-1  0  0], [-1  0  0], [ 1 -1  0], [-1  0  0], [-1  0  0], [-1  0  0]
    
    COMMENTS:
    Weyl groups that are computed are stored in a dictionary _weyl_groups 
    for easy and quick access.
    """
    return [matrix(g) for g in WeylGroup(type_rank).canonical_representation().list()]  # type: ignore


@convert_input_to_tuple
@lru_cache(maxsize=None)
def weyl_vector(type_rank):
    """
    Compute the Weyl vector of a given Lie group.

    INPUT:
    -   ``type_rank'' - [str,int]; Lie group identifier

    Examples::    
        sage: weyl_vector(["A",2])
        (1,1)
        sage: weyl_vector(["D",4])
        (3,5,3,3)
    
    COMMENTS:
    Weyl vectors that are computed are stored in a dictionary _weyl_groups 
    for easy and quick access.
    """
    WG = WeylGroup(type_rank).canonical_representation() # type: ignore
    return Integer(Integer(1))/Integer(Integer(2))*sum(WG.positive_roots())

@convert_input_to_tuple
@lru_cache(maxsize=None)
def weyl_lengths(type_rank):
    """
    Return weyl lengths of elements of the weyl group

    Inputs:
    -   ``type_rank`` - [str,int]; Lie group identifier

    Examples::    
        sage: weyl_lengths(["A",2])
        [1,-1,-1,1,1,-1]
        sage: weyl_lengths(["A",3])[:10]
        [1, -1, -1, -1, 1, 1, 1, 1, 1, -1]
    
    COMMENTS:
    Weyl vectors that are computed are stored in a dictionary _weyl_groups 
    for easy and quick access.
    """
    w_gr = weyl_group(type_rank)
    return [det(g) for g in w_gr]

def weyl_lattice_norm(type_rank, v1, v2=None, basis=None,):
    """
    Compute the inner produt on the root or weight lattice  of a given
    Lie algebra between vectors v1 and v2.

    Input:

    -    ``type_rank`` - [str,int]; Lie group identifier
    -    ``v1`` -- vector; An lattice vector
    -    ``v2`` -- vector; An lattice vector
    -    ``basis`` -- string; basis of vectors, either root or weight

    Example:
        sage: vec1, vec2 = vector([1,2]), vector([2,3])
        sage: weyl_lattice_norm(["A",2],vec1, basis="weight")
            14/3
        sage: weyl_lattice_norm(["A",2],vec1, basis="root")
            6
        sage: vec3, vec4 = vector([1,2,3]), vector([4,5,6])
        sage: weyl_lattice_norm(["A",3],vec3, vec4, basis="root")
            24
    """

    if basis == None:
        warnings.warn("No basis is specified, weight is assumed.")
        basis = "weight"

    if v2 == None:
        v2 = v1

    assert len(v1) == len(v2), "Vectors must have same dimension"
    assert len(v1) == type_rank[Integer(Integer(1))], "Dimensions do not match"

    mat = cartan_matrix(type_rank)
    if basis == "weight":
        mat = mat.inverse()

    return vector(v1)*mat*vector(v2)




def weyl_double_sided_expansion(type_rank,n_powers): # Can be implemented in cython to make quicker
    """
        Return both sides of the double sided expansion (at zero and infinity for each variable) of the weyl denominator
        with n_powers of the geometric series.
    """
    # Set up the weyl denominator expansion
    rho = cartan_matrix(type_rank)*weyl_vector(type_rank)
    WG = [g.T for g in weyl_group(type_rank)]
    den_exp = [g*rho for g in WG]
    den_coeffs = weyl_lengths(type_rank)

    # Compute the expansion at zero and infinity
    zero_inf_exp = list()
    for i in range(Integer(Integer(2))): # Expansion at 0 and oo
        exps = [list(exp + (-Integer(Integer(1)))**(i)*rho) for exp in den_exp]
        coeffs = copy(den_coeffs)

        one_index = exps.index([Integer(Integer(0))]*type_rank[Integer(Integer(1))])
        one_sign = coeffs[one_index]

        coeffs.pop(one_index)
        exps.pop(one_index)

        if one_sign == Integer(Integer(1)):
            coeffs = [-Integer(Integer(1))*c for c in coeffs]
        
        weyl = Series(list(zip(exps,coeffs)))
        expansion = Series([[[Integer(Integer(0))]*type_rank[Integer(Integer(1))],Integer(Integer(0))]])
        for n in range(n_powers):
            expansion += weyl.pow(n)
        zero_inf_exp.append(expansion * Series([([c**(i) for c in -rho],-(-Integer(Integer(1)))**i)]))
    return zero_inf_exp

def invert_powers(poly):
    """
    Invert the powers of a polynomial.
    """
    inverted = list()
    for monomial in poly.numerical:
        inverted += [[[-Integer(Integer(1))*e for e in monomial[Integer(Integer(0))]],monomial[Integer(Integer(1))]]]
    return Series(inverted)

def L_norm_python(inv_plumbing, C_inv, ell_array, rnd):
    """
    L norm of a plumbing graph.
    """
    tot = list()
    for ell in ell_array:
        tot.append(np.round(sum( inv_plumbing[i,j] * QQ(np.dot(np.dot(ell[i],C_inv),ell[j])) for i,j in itertools.product(range(len(ell)),repeat=Integer(Integer(2)))),rnd))
    return tot

def L_norm_vectorized(inv_plumbing, C_inv, ell_array, rnd):
    """
    Vectorized L norm of a plumbing graph, with optimized dot product computation and cached inverse.
    """
    # Ensure ell_array is in float64 unless precision needs 128
    ell_array = np.array(ell_array, dtype=np.float128)
    
    # Precompute the dot products efficiently (einsum might still be best for higher dimensional contractions)
    ell_C_inv = np.einsum("ij,klj->kli", C_inv, ell_array)
  
    # Perform matrix multiplication in a vectorized fashion
    product = np.matmul(ell_C_inv, ell_array.transpose(Integer(Integer(0)), Integer(Integer(2)), Integer(Integer(1))))

    # Compute the total and vectorize the rounding operation
    tot = np.sum(inv_plumbing * product, axis=(Integer(Integer(1)), Integer(Integer(2))))

    # Use vectorized rounding instead of list comprehension
    rounded_tot = np.round(tot, decimals=rnd)
    
    return rounded_tot
