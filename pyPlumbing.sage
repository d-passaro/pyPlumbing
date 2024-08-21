from sage.all_cmdline import *   # import sage library
from sage.graphs.graph_plot import GraphPlot
from collections import Counter, defaultdict
from typing import List
from functools import lru_cache, wraps
import numpy as np
import itertools
from cython_l_norm import *

def convert_input_to_tuple(func):
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
    return [matrix(g) for g in WeylGroup(type_rank).canonical_representation().list()]


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
    WG = WeylGroup(type_rank).canonical_representation()
    return 1/2*sum(WG.positive_roots())

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
    assert len(v1) == type_rank[1], "Dimensions do not match"

    mat = cartan_matrix(type_rank)
    if basis == "weight":
        mat = mat.inverse()

    return vector(v1)*mat*vector(v2)


_known_multiplicities = dict()
def freudenthal_multiplicity(type_rank, L, m):
    """
        Compute multiplicity of weight m within rep of highest weight L. 
        Uses Freudenthal\'s recursive formula.

        Input:
        -    ``type_rank`` - [str,int]; Lie group identifier
        -   ``Lambda`` -- vector; Highest weight vector
        -   ``m`` -- vector; Weight vector

        Example::
            sage: freudenthal_multiplicity(["A",2],vector([3,3]),vector([-1,2]))
            3
            sage: freudenthal_multiplicity(["A",3],vector([2,0,2]),vector([0,0,0]))
            6
    """

    cart = cartan_matrix(type_rank)
    cart_i = cart.inverse()
    rho = cart*weyl_vector(type_rank)

    if str([type_rank,L,m]) in _known_multiplicities.keys():
        return _known_multiplicities[str([type_rank,L,m])]
    mult = 0
    if np.logical_or.reduce(np.less(cart_i*(L-m), 0)):
        # m is a higher weight that Lambda
        mult = 0
    elif L == m:
        # m is Lambda
        mult = 1
    else:
        # m is a lower weight than Lambda
        num = 0
        p_roots = WeylGroup(type_rank).canonical_representation().positive_roots()
        for pr in p_roots:
            k = 1
            v = m+k*cart*pr
            while all(c >= 0 for c in cart_i*(L-v)):
                num += 2*freudenthal_multiplicity(type_rank,L,v) * \
                    weyl_lattice_norm(type_rank,v,cart*pr,basis="weight")
                k += 1
                v = m+k*cart*pr
        
        den = weyl_lattice_norm(type_rank,L+rho, basis="weight") - \
            weyl_lattice_norm(type_rank,m+rho, basis="weight")

        if den == 0:
            mult = 0
        else:
            mult = num / den
    _known_multiplicities[str([type_rank,L,m])] = mult
    return mult


def q_series(fprefexp, expMax, dim, qvar=None, rejection=3):
    """
        Compute the q series with prefactors and exponents computed by fprefexp.

        INPUT:
        -   ``fprefexp`` -- function; function which computes prefactors and exponents. Must take as input a dim-dimensional lattice vector only, and return prefactors and exponents as a tuple: (pref,exp), where pref and exp are lists.
        -   ``expMax`` -- Integer; maximum power of expansion
        -   ``qvar`` -- sage.symbolic.expression.Expression (default=var('q')); expansion variable
        -   ``rejection`` -- Integer; rejection parameter. After no acceptable terms are found in centered taxi-cab circles of increasing radius a number of times specified by the rejection parameter the function concludes.

        EXAMPLES::
            sage: fprefexp = lambda n: ([1/2*abs(n[0])],[n[0]^2]); q_series(fprefexp,120,1)
            10*q^100 + 9*q^81 + 8*q^64 + 7*q^49 + 6*q^36 + 5*q^25 + 4*q^16 + 3*q^9 + 2*q^4 + q

            sage: fprefexp = lambda n: ([1/2*abs(n[0])],[abs(n[0])]); q_series(fprefexp,10,1,qvar=var("z"))
            9*z^9 + 8*z^8 + 7*z^7 + 6*z^6 + 5*z^5 + 4*z^4 + 3*z^3 + 2*z^2 + z

            sage: fprefexp = lambda n: ([1/2*abs(n[0]+n[1])],[n[0]^2+n[0]*n[1]+n[1]^2]); q_series(fprefexp,20,2,rejection=5)
            20*q^19 + 8*q^16 + 16*q^13 + 8*q^12 + 6*q^9 + 12*q^7 + 4*q^4 + 4*q^3 + 2*q
    """
    amax = 0
    rejected = 0
    q_series = 0
    allNs = set()
    if qvar == None:
        qvar = var("q")

    while rejected < rejection:
        newNs = set(itertools.product(range(-amax, amax+1), repeat=dim))-allNs
        allNs = allNs.union(newNs)
        pref = list()
        exp = list()
        for n in newNs:
            temp = fprefexp(vector(n))
            pref += temp[0]
            exp += temp[1]

        new_terms_found = False
        for e, p, in zip(exp, pref):
            if abs(e) < expMax:
                new_terms_found = True
                rejected = 0
                q_series += p*qvar ** e
        # If there are no terms to keep (goodTerms is a list of False) increase rejected counter
        if not new_terms_found:
            rejected += 1
        amax += 1
    return q_series


def weyl_cycle(type_rank,v, f, z=None, basis=None):
    """
        Compute the Weyl Cycled z polynomial associated to v:

        MATH::
            \\sum_{w \\in W} (-1)^{f l(w)} \\exp^{\\langle  \\vec{\\xi}, w(v) \\rangle} =   0

        where z_i are defined in equations (3.20) and (3.21) of [1].

        INPUT:
        -    ``type_rank`` - [str,int]; Lie group identifier
        -   ``v`` -- vector; lattice vector
        -   ``f`` -- Integer; Weyl length factor
        -   ``z`` -- variable (Optional); symbolic expressions of z_i. If none are given then z_i is chosen as default.
        -   ``basis`` -- string; basis in which v is given.

        EXAMPLES::
            sage: weyl_cycle(["A",2],vector([1,2]),3,basis = "weight")
            z0*z1^2 - z1^3/z0 - z0^3/z1^2 + z0^2/z1^3 + z1/z0^3 - 1/(z0^2*z1)
            sage: weyl_cycle(["A",3],vector([1,2,1]),3,basis = "weight")
            z0*z1^2*z2 - z0^3*z2^3/z1^2 - z0*z1^3/z2 - z1^3*z2/z0 + z0^4*z2^2/z1^3 
            +z0^2*z2^4/z1^3 + z1^4/(z0*z2) - z0^3*z2^3/z1^4 + z0^3*z1/z2^3 
            -z0^4/(z1*z2^2) + z1*z2^3/z0^3 - z2^4/(z0^2*z1) - z0^2*z1/z2^4 
            +z0^3/(z1*z2^3) - z1*z2^2/z0^4 + z2^3/(z0^3*z1) - z1^4/(z0^3*z2^3) 
            +z0*z2/z1^4 + z1^3/(z0^2*z2^4) + z1^3/(z0^4*z2^2) - z0/(z1^3*z2) 
            -z2/(z0*z1^3) - z1^2/(z0^3*z2^3) + 1/(z0*z1^2*z2)   
    """

    if basis == None:
        warnings.warn("No basis is specified, weight is assumed.")
        basis = "weight"

    assert basis == "root" or basis == "weight", "basis must be root or weight"

    rk = type_rank[1]    
    if z == None:
        varstr = ""
        for i in range(rk):
            varstr += f", z{i}"
        z = var(varstr)

    if basis == "root":
        v = cartan_matrix(type_rank)*v
        basis = "weight"

    WG = [g.transpose() for g in weyl_group(type_rank)] # get to weight basis
    WL = weyl_lengths(type_rank)

    v = vector(v)
    WGv = list()
    for g, l in zip(WG, WL):
        WGv.append([g*v, l])

    cycle = list()
    for gv in WGv:
        cycle.append(gv[1]**(f) *
                     product([x**y for x, y in zip(z, gv[0])]))
    return sum(cycle)


def const_term(expr):
    """
    Extract the constant term of a polyonmial.

    INPUT:
    -   ``expr`` -- polynomial

    EXAMPLES::
        sage: x = var("x");const_term(x^2+x+3)
        3
    """
    q = var("q")
    zeroTerm = expr
    for v in expr.variables():
        if v == q:
            continue
        for coeff, exp in zeroTerm.series(v,3).coefficients(v):
            if exp == 0:
                zeroTerm = coeff
                break
    return zeroTerm


def weyl_exp(N):
    """
    Expand the inverse of the A_2 Weyl determinant in powers of z_i, for large z_i.

    INPUT:
    -   ``N`` -- 2 dimensional vector; expansion order

    EXAMPLES::
        sage: weyl_exp(vector([2,3]))
        (z0/z1^2 + z0^2/z1^4 + 1)*(z1/z0^2 + z1^2/z0^4 + z1^3/z0^6 + 1)*(1/(z0*z1) + 1/(z0^2*z1^2) + 1)/(z0*z1)

    """

    z0, z1 = var("z0,z1")
    return 1/(z0*z1) * sum([(z0/z1 ^ 2) ^ n for n in range(N[0]+1)]) * sum([(z1/z0 ^ 2) ^ n for n in range(N[1]+1)])*sum([(1/(z0*z1)) ^ n for n in range(min(N)+1)])


def triplet_character(type_rank, lmbd, mu, m, f, expMax,  basis="weight", qvar=var("q")):
    """
    Compute the triplet character with specified parameters, up to the inverse Dedekind eta function to the
    power of the rank of the lattice. Argument descriptions refer to equation (3.15) of [1].

    INPUTS:
    -    ``type_rank`` - [str,int]; Lie group identifier
    -   ``lmbd`` -- Vector; lambda parameter in equation (3.15)
    -   ``mu`` -- Vector; mu parameter in equatoin (3.13)
    -   ``m`` -- Integer; m parameter in equatoin (3.13)
    -   ``f`` -- Integer; number of fibers of Seifert manifold
    -   ``expMax`` -- Integer; Maximum exponent in q series expansion
    -   ``basis`` -- String; basis in which wh and b are given
    -   ``qvar`` -- Variable (default=None); variable in which to expand. If None qvar = var("q")

    EXAMPLES::
            sage: lmbd,mu,m,f,expMax = vector([0,0]),1/sqrt(30)*vector([3,3]),30,2,20
            sage: triplet_character(["A",2],lmbd,mu,m,f,expMax)
            6*q^(2/15)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1))
    """

    rk = type_rank[1]
    cart_i = cartan_matrix(type_rank).inverse()
    if basis == "weight":
        mu = cart_i*mu
        lmbd = cart_i*lmbd
        basis = "root"

    rho = weyl_vector(type_rank) # Basis is root

    def Delta(v): 
        return weyl_cycle(type_rank, v, f, basis=basis)

    def fprefexp(lt):
        expl = [1/(2*m)*weyl_lattice_norm(type_rank, m*(lmbd+lt) +
                                     sqrt(m)*mu+(m-1)*rho, basis=basis)]
        prefl = [Delta(lt+lmbd+rho)/Delta(rho)]
        return prefl, expl

    return q_series(fprefexp, expMax, rk, qvar=qvar)


def singlet_character(type_rank, lmbdt, lmbd, mu, m, f, expMax, basis="root", qvar=var("q")):
    """
    Compute the singlet character with specified parameters, up to the inverse Dedekind eta function to the
    power of the rank of the lattice. Argument descriptions refer to equation (3.20) of [1].

    INPUT:
    -   ``type_rank`` - [str,int]; Lie group identifier
    -   ``lmbd`` -- Vector; lambda-tilde parameter in equation (3.20)
    -   ``lmbd`` -- Vector; lambda parameter in equation (3.20)
    -   ``mu`` -- Vector; mu parameter in equatoin (3.20)
    -   ``m`` -- Integer; m parameter in equatoin (3.20)
    -   ``f`` -- Integer; number of fibers of Seifert manifold
    -   ``expMax`` -- Integer; Maximum exponent in q series expansion
    -   ``basis`` -- String; basis in which wh and b are given
    -   ``qvar`` -- Variable (default=None); variable in which to expand. If None qvar = var("q")

    EXAMPLES::
        sage: lmbd,lmbdt,mu,expMax = vector([0,0]),vector([0,0]),1/sqrt(S.m)*vector([3,3]),100
        ....: singlet_character(lmbdt,lmbd,mu,m,f,expMax)
        -4*q^(1352/15) - 4*q^(1262/15) - q^(512/15) - 2*q^(482/15) - 2*q^(422/15) - q^(392/15)
    """

    rk = type_rank[1]
    cart_i = cartan_matrix(type_rank).inverse()
    if basis == "weight":
        mu = cart_i*mu
        lmbd = cart_i*lmbd
        basis = "root"

    rho = weyl_vector(type_rank) # Basis is root

    varstr = ""
    for i in range(rk):
        varstr += f", z{i}"
    z = var(varstr)

    def Delta(v): 
        return weyl_cycle(type_rank, v, f, basis=basis)

    def fprefexp(lt):
        expl = [1/(2*m)*weyl_lattice_norm(type_rank, m*(lmbd+lt) +
                                     sqrt(m)*mu+(m-1)*rho, basis=basis)]
        prefl = [const_term(Delta(lt+lmbd+rho)/Delta(rho)*product([zz**l for zz, l in zip(z,lmbdt)]))]
        return prefl, expl

    return q_series(fprefexp, expMax, rk, qvar=qvar)



def triplet_character_p_pprime(p, pp, s, r, expMax, qvar=var("q"), z=var("z")):
    """
    Compute the p,p' triplet character with specified parameters, up to the inverse Dedekind eta function.
    Argument descriptions refer to equation (3.45) of [1].

    INPUT:
    -   ``p`` -- Vector; lambda parameter in equation (3.20)
    -   ``pp`` -- Vector; mu parameter in equatoin (3.20)
    -   ``s`` -- Integer; m parameter in equatoin (3.20)
    -   ``r`` -- Integer; number of fibers of Seifert manifold
    -   ``expMax`` -- Integer; Maximum exponent in q series expansion
    -   ``basis`` -- String; basis in which wh and b are given
    -   ``qvar`` -- Variable (default=None); variable in which to expand.

    EXAMPLES::
        sage: p, pp, r, s, expMax = 2, 105, 29, 1,200
        ....: triplet_character_p_pprime(p,pp,r,s,expMax)
        -(z^2 + 1/z^2 - 2)*q^(139129/840)/(z - 1/z)^2 + (z^2 + 1/z^2 - 2)*q^(66049/840)/(z - 1/z)^2

    """

    def fprefexp(n):
        k = n[0]
        prefl = [x*(z ^ (2*k)-2+z ^ (-2*k))*(z-z ^ (-1)) ^ (-2)
                 for x in [1, -1]]
        expl = [(2*p*pp*k+p*s+pp*r) ^ 2/(4*p*pp),
                (2*p*pp*k+p*s-pp*r) ^ 2/(4*p*pp)]
        return prefl, expl

    return q_series(fprefexp, expMax, 1, qvar=qvar)

def multi_poly_coeffs(multi_poly):
    """
        Get coefficients and powers of a multivariable poly expansion.
    """
    vrs = multi_poly.variables()    
    if is_monomial(multi_poly):
        coeff = multi_poly.substitute(*(v==1 for v in vrs))
        exps = list()
        for v in vrs:
            exps.append((multi_poly.degree(v)))
        return [[exps, coeff]]
    else:
        coeffs = list()
        powrs = list()
        for mono in multi_poly.iterator():
            coeffs.append(Integer(mono.substitute(*(v==1 for v in vrs))))
            exps = list()
            for v in vrs:
                exps.append(mono.degree(v))
            powrs.append(list(exps))
        return list(zip(powrs,coeffs))

def is_monomial(multi_poly):
    """
        Check if a multivariable polynomial is a monomial.
    """
    try:
        if multi_poly.is_numeric():
            return True
        if len(multi_poly.variables()) == 1 and multi_poly == multi_poly.variables()[0]:
            return True
        if (multi_poly == product(multi_poly.op)):
            return True
        if len(list(multi_poly.op)) == 2 and multi_poly.op[1].is_constant() and (multi_poly == multi_poly.op[0]**multi_poly.op[1]):
            return True
        return False
    except Exception as e:
        print("Warning: could not determine if expression is a monomial.")
        print("Polynomial:" + str(multi_poly))
        print("Exception: "+ str(e))
        return False

def _continued_fraction(x, iterMax=1000):
    """
    Computes the continued fraction of x with parameter a up to iterMax iterations.
        [
    INPUT:
    -   ``x`` --  Rational; fraction to expand
    -   ``iterMax`` - Integer (default = 1000); iteration maximum, default = 1000

    EXAMPLES::
        sage: _continued_fraction(3/5)
        [1,3,2]

    Some code taken from
    https://share.cocalc.com/share/d1efa37e-be6a-40f6-80c7-8a34201d7c4e/PlumbGraph.sagews?viewer=share
    on July 4th 2021

    """
    assert x in QQ, "x must be a rational number"
    assert iterMax in NN, "iterMax must be a positive integer"

    n = 0
    r = list()
    while n < iterMax:
        r.append(ceil(x))
        if x == ceil(x):
            break
        else:
            x = 1/(ceil(x)-x)
            n += 1
    if n < iterMax:
        return r
    else:
        return r

def join_dicts_with_function(dict1, dict2, conflict_resolver):
    """
    Join two dictionaries using a specified function to resolve conflicts for overlapping keys.

    Parameters:
    - dict1: First dictionary.
    - dict2: Second dictionary.
    - conflict_resolver: Function to apply in case of overlapping keys. Should accept two arguments (values from both dictionaries for the overlapping key) and return a single value.

    Returns:
    - A new dictionary with combined key-value pairs.
    """
    # Start with a shallow copy of dict1 to avoid modifying the original
    result = dict1.copy()

    # Iterate through dict2, adding or updating keys in the result
    for key, value in dict2.items():
        if key in result:
            # If the key exists in both, use the conflict_resolver function to determine the value
            result[key] = conflict_resolver(result[key], value)
        else:
            # If the key is unique to dict2, add it to the result
            result[key] = value

    return result

def weyl_double_sided_expansion(type_rank,n_powers):
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
    for i in range(2):
        exps = [list(exp + (-1)**(i)*rho) for exp in den_exp]
        coeffs = copy(den_coeffs)

        one_index = exps.index([0]*type_rank[1])
        one_sign = coeffs[one_index]

        coeffs.pop(one_index)
        exps.pop(one_index)

        if one_sign == 1:
            coeffs = [-1*c for c in coeffs]
        
        weyl = Series(list(zip(exps,coeffs)))
        expansion = Series([[[0]*type_rank[1],0]])
        for n in range(n_powers):
            expansion += weyl.pow(n)
        zero_inf_exp.append(expansion * Series([([c**(i) for c in -rho],-(-1)**i)]))
    return zero_inf_exp

def invert_powers(poly):
    inverted = list()
    for monomial in poly.numerical:
        inverted += [[[-1*e for e in monomial[0]],monomial[1]]]
    return Series(inverted)

def L_norm_python(inv_plumbing, C_inv, ell_array, rnd):
    """
    L norm of a plumbing graph.
    """
    tot = list()
    for ell in ell_array:
        tot.append(sum( inv_plumbing[i,j] * QQ(np.dot(np.dot(ell[i],C_inv),ell[j])) for i,j in itertools.product(range(len(ell)),repeat=2)))
    return tot

def L_norm_vectorized(inv_plumbing, C_inv, ell_array, rnd):
    """
    Vectorized L norm of a plumbing graph, with optimized dot product computation and cached inverse.
    """
    # Ensure ell_array is in float64 unless precision needs 128
    ell_array = np.array(ell_array, dtype=np.float64)
    
    # Precompute the dot products efficiently (einsum might still be best for higher dimensional contractions)
    ell_C_inv = np.einsum("ij,klj->kli", C_inv, ell_array)
  
    # Perform matrix multiplication in a vectorized fashion
    product = np.matmul(ell_C_inv, ell_array.transpose(0, 2, 1))

    # Compute the total and vectorize the rounding operation
    tot = np.sum(inv_plumbing * product, axis=(1, 2))

    # Use vectorized rounding instead of list comprehension
    rounded_tot = np.round(tot, decimals=rnd)
    
    return rounded_tot



class WeightedGraph():
    def __init__(self, vertices_dict, edges):
        try:
            self._vertices_dict = vertices_dict
            self._edges = edges

            self._vertex_count = len(vertices_dict)
            self._edge_count = len(edges)

            self._graph = Graph()
            self._vertex_list = ['$v_{' + str(i) + '}\\hspace{2} '
                                  + str(vertices_dict[i])
                                  + '$' for i in range(0, self._vertex_count)]

            self._edge_list = [(self._vertex_list[x[0]], self._vertex_list[x[1]])
                               for x in edges]

            self._graph.add_vertices(self._vertex_list)
            self._graph.add_edges(self._edge_list)
            self._plot_options = {'vertex_color': 'black',
                                  'vertex_size': 20,
                                  'layout': 'tree'}

            self._graph_plot = GraphPlot(self._graph, self._plot_options)

            self._weight_vector = Matrix(list(vertices_dict.values())).T
            self._degree_vector = [self._graph.degree(x) for x in
                                   self._vertex_list]
            self._degree_vector = Matrix(self._degree_vector).T

            self._adjacency_matrix = None

            self._adjacency_matrix_inverse = None
            self._adjacency_matrix_determinant = None
            

        except Exception as e:
            print(f"Error: Weighted graph entered incorrectly. Please check input. Details {e}")
            # Print error message

    @property
    def vertex_count(self):
        """int: The number of vertices in the plumbing."""
        return self._vertex_count

    @property
    def edge_count(self):
        """int: The number of edges in the plumbing"""
        return self._edge_count

    @property
    def weight_vector(self):
        """Matrix: An sx1 matrix of the form [[m_1],...,[m_s]] where m_i is the
        weight of vertex i and s is the number of vertices of the plumbing."""
        return self._weight_vector

    @property
    def degree_vector(self):
        """Matrix: An sx1 matrix of the form [[d_1],...,[d_s]] where d_i is the
        degree (or valence) of vertex i and s is the number of vertices of the
        plumbing."""
        return self._degree_vector

    @property
    def adjacency_matrix(self):
        """Matrix: The plumbing matrix of the plumbing manifold."""
        if self._plumbing_matrix is None:
            adjacency_matrix = self._graph.adjacency_matrix()
            for i in range(0, self._vertex_count):
                adjacency_matrix[i, i] = self._weight_vector[i,0]
            self._adjacency_matrix = adjacency_matrix
        return self._adjacency_matrix
    
    @property
    def graph(self):
        """Graph: The graph representation of the plumbing manifold."""
        return self._graph
    
    def __repr__(self):
        self.graph.show()
        return f"Weighted Graph with {self._vertex_count} vertices and {self._edge_count} edges."
    
class Plumbing():
    def __init__(self, vertices_dict, edges):
        try:
            self._vertices_dict = vertices_dict
            self._edges = edges

            self._vertex_count = len(vertices_dict)
            self._edge_count = len(edges)

            self._graph = Graph()
            self._vertex_list = ['$v_{' + str(i) + '}\\hspace{2} '
                                  + str(vertices_dict[i])
                                  + '$' for i in range(0, self._vertex_count)]

            self._edge_list = [(self._vertex_list[x[0]], self._vertex_list[x[1]])
                               for x in edges]

            self._graph.add_vertices(self._vertex_list)
            self._graph.add_edges(self._edge_list)
            self._plot_options = {'vertex_color': 'black',
                                            'vertex_size': 20,
                                            'layout': 'tree'}

            self._graph_plot = GraphPlot(self._graph, self._plot_options)

            self._weight_vector = Matrix(list(vertices_dict.values())).T
            self._degree_vector = [self._graph.degree(x) for x in
                                   self._vertex_list]
            self._degree_vector = Matrix(self._degree_vector).T

            self._plumbing_matrix = None

            self._plumbing_matrix_inverse = None
            self._plumbing_matrix_determinant = None
            self._lattice_dilation_factor = None
            self._orbifold_euler = None
            self._norm_fac_euler = None
            self._eff_inv_euler = None
            self._coker= None

            self._is_plumbing_non_singular = None
            self._is_unimodular = None
            self._is_tree = None
            self._definiteness_type = None
            self._weak_definiteness_type = None
            self._is_Seifert = None
            self._Seifert_fiber_count = None
            self._Seifert_data = None

        except Exception as e:
            print(f"Error: Plumbing entered incorrectly. Please check input. Details {e}")
            # Print error message
    
    @classmethod
    def from_Seifert_data(cls, Seifert_Data):
        """
        Create a plumbing manifold from Seifert data.
        """
        # Initialize vertices and edges
        vertices = {}
        edges = []

        def continued_fraction(x):
            """
            Compute the continued fraction of a rational number x.
            """
            assert x in QQ, "x must be a rational number"
            x = -x  # Invert the sign of x for the calculation
            fractions = []
            while True:
                ceil_x = ceil(x)
                fractions.append(-ceil_x)
                if x == ceil_x:
                    break
                x = 1 / (ceil_x - x)
            return fractions

        # Process the Seifert data
        vertices[0] = Seifert_Data[0]
        vertex_id = 1  # Start from 1 since the first vertex is already added

        for seifert_value in Seifert_Data[1:]:
            leg_data = continued_fraction(-1 / seifert_value)
            for i, fraction in enumerate(leg_data):
                vertices[vertex_id] = fraction
                # Connect the first fraction to the central vertex (0), others to the previous vertex
                edge_start = 0 if i == 0 else vertex_id - 1
                edges.append((edge_start, vertex_id))
                vertex_id += 1
        return cls(vertices, edges)
    
    def invert_orientation(self):
        return Plumbing({ k: -v for k, v in self._vertices_dict.items()}, [(x[1], x[0]) for x in self._edges])

    @property
    def vertex_count(self):
        """int: The number of vertices in the plumbing."""
        return self._vertex_count

    @property
    def edge_count(self):
        """int: The number of edges in the plumbing"""
        return self._edge_count

    @property
    def weight_vector(self):
        """Matrix: An sx1 matrix of the form [[m_1],...,[m_s]] where m_i is the
        weight of vertex i and s is the number of vertices of the plumbing."""
        return self._weight_vector

    @property
    def degree_vector(self):
        """Matrix: An sx1 matrix of the form [[d_1],...,[d_s]] where d_i is the
        degree (or valence) of vertex i and s is the number of vertices of the
        plumbing."""
        return self._degree_vector

    @property
    def plumbing_matrix(self):
        """Matrix: The plumbing matrix of the plumbing manifold."""
        if self._plumbing_matrix is None:
            plumbing_matrix = self._graph.adjacency_matrix()
            for i in range(0, self._vertex_count):
                plumbing_matrix[i, i] = self._weight_vector[i,0]
            self._plumbing_matrix = plumbing_matrix
        return self._plumbing_matrix

    @property
    def is_plumbing_non_singular(self):
        "bool: True if the plumbing matrix is non-singular, False otherwise."
        if self._is_plumbing_non_singular is None:
            self._is_plumbing_non_singular = self.plumbing_matrix.det() != 0
        return self._is_plumbing_non_singular

    @property
    def is_unimodular(self):
        "bool: True if the plumbing matrix is unimodular, False otherwise."
        if self._is_unimodular is None:
            self._is_unimodular = (self.plumbing_matrix.det())**2 == 1 
        return self._is_unimodular

    
    @property
    def definiteness_type(self, threshold=1e-8):
        """str: The definiteness type of the plumbing matrix of the plumbing manifold.

        Warnings
        --------
        Since the eigenvalues are computed numerically, they may contain small
        error terms. Therefore, to check the sign of an eigenvalue, we have
        chosen a small error threshold (default 1e-8). This potentially could lead to
        incorrect answers in some edge cases when the true eigenvalues are very
        close to zero, but non-zero.

        """
        if self._definiteness_type is None:
            eigenvalues = self.plumbing_matrix.eigenvalues()
            if all(i < -1*threshold for i in eigenvalues):
                self._definiteness_type = "negative definite"
            elif all(i > threshold for i in eigenvalues):
                self._definiteness_type = "positive definite"
            elif all(-1*threshold <= i <= threshold for i in eigenvalues):
                self._definiteness_type = "zero matrix"
            elif all(i <= threshold for i in eigenvalues):
                self._definiteness_type = "negative semi-definite"
            elif all(i >= -1*threshold for i in eigenvalues):
                self._definiteness_type = "positive semi-definite"
            else:
                return "positive and negative eigenvalues"
        return self._definiteness_type

    @property
    def weak_definiteness_type(self, threshold=1e-8):
        """str: The definiteness type of the plumbing matrix minor corresponding to high valency nodes.

        Warnings
        --------
        Since the eigenvalues are computed numerically, they may contain small
        error terms. Therefore, to check the sign of an eigenvalue, we have
        chosen a small error threshold (default 1e-8). This potentially could lead to
        incorrect answers in some edge cases when the true eigenvalues are very
        close to zero, but non-zero.
        """

        if self._weak_definiteness_type is None:
            minor = copy(self.plumbing_matrix.inverse())
            to_remove = list()
            for index, degree in enumerate(self.degree_vector):
                if degree[0] <= 2:
                    to_remove.append(index)
            to_remove = sorted(to_remove, reverse=True)
            for index in to_remove:
                minor = minor.delete_rows([index])
                minor = minor.delete_columns([index])

            eigenvalues = minor.eigenvalues()
            if all(i < -1*threshold for i in eigenvalues):
                self._weak_definiteness_type = "negative weakly definite"
            elif all(i > threshold for i in eigenvalues):
                self._weak_definiteness_type = "positive weakly definite"
            elif all(-1*threshold <= i <= threshold for i in eigenvalues):
                self._weak_definiteness_type = "zero matrix"
            elif all(i <= threshold for i in eigenvalues):
                self._weak_definiteness_type = "negative weakly semi-definite"
            elif all(i >= -1*threshold for i in eigenvalues):
                self._weak_definiteness_type = "positive weakly semi-definite"
            else:
                return "positive and negative eigenvalues"
        return self._weak_definiteness_type

    @property
    def is_tree(self):
        """Bool: True if the plumbing graph is a tree, False otherwise."""
        if self._is_tree is None:
            self._is_tree = self._graph.is_tree()
        return self._is_tree

    @property
    def is_Seifert(self):
        """Bool: True if the manifold is a Seifert manifold, False otherwise. """
        if self._is_Seifert is None:
            if self.is_tree == False:
               return False
            else:
                high_valency_node_count = sum(1 for d in self._degree_vector if d[0] > 2)
                self._is_Seifert =  high_valency_node_count == 1
        return self._is_Seifert
    
    @property
    def Seifert_fiber_count(self):
        """Integer: Return the number of Seifert manifolds. """
        if self._Seifert_fiber_count is None:
            if not self.is_Seifert:
                print("The plumbing graph is not a Seifert manifold.")
                self._Seifert_fiber_count    
            else:
                self._Seifert_fiber_count = max(d[0] for d in self._degree_vector)
        return self._Seifert_fiber_count
    
    @property
    def graph(self):
        """Graph: The graph representation of the plumbing manifold."""
        return self._graph
    
    @property
    def Seifert_data(self):
        if self._Seifert_data is None:
            if not self.is_Seifert:
                print("The plumbing graph is not a Seifert manifold.")
                return -1
        
            # Flatten the list of edges
            edges_flat = [vertex for edge in self._edges for vertex in edge]

            # Count the occurrences of each vertex
            vertex_counts = Counter(edges_flat)

            # Find the vertex with the highest valency
            high_valency_vertex = max(vertex_counts, key=vertex_counts.get)

            # Initialize the list to store legs
            legs = []

            # Function to find the next vertex in the leg
            def find_next_vertex(current_vertex, edges, visited):
                for edge in edges:
                    if current_vertex in edge:
                        next_vertex = [v for v in edge if v != current_vertex][0]
                        if next_vertex not in visited:
                            return next_vertex
                return None

            # Find legs connected to the high valency vertex
            for edge in self._edges:
                if high_valency_vertex in edge:
                    leg = []
                    last_vertex = [v for v in edge if v != high_valency_vertex][0]
                    leg.append(last_vertex)
                    visited = set(leg).union({high_valency_vertex})
                    
                    while vertex_counts[last_vertex] > 1:
                        next_vertex = find_next_vertex(last_vertex, self._edges, visited)
                        if next_vertex:
                            leg.append(next_vertex)
                            visited.add(next_vertex)
                            last_vertex = next_vertex
                        else:
                            break

                    legs.append(leg)

            # Calculate weights for each leg
            legs_weights = [[self._vertices_dict[v] for v in leg] for leg in legs]

            # Calculate the Seifert data
            seif_data = list()
            seif_data.append(self._vertices_dict[high_valency_vertex])
            for leg in legs_weights:
                seif_coeff = leg[0]
                for ai in leg[1:]:
                    seif_coeff = ai - 1/seif_coeff
                seif_coeff = -1/seif_coeff
                seif_data.append(seif_coeff)
            self._Seifert_data = seif_data
        return self._Seifert_data

    @property
    def plumbing_matrix_inverse(self):
        """
        Return the inverse of the plumbing matrix
        """
        if self._plumbing_matrix_inverse == None:
            self._plumbing_matrix_inverse = self.plumbing_matrix.inverse()
        return self._plumbing_matrix_inverse
    
    @property
    def plumbing_matrix_determinant(self):
        """
        Return the determinant of the plumbing matrix
        """
        if self._plumbing_matrix_determinant == None:
            self._plumbing_matrix_determinant = self.plumbing_matrix.det()
        return self._plumbing_matrix_determinant

    @property
    def orbifold_euler(self):
        """
        Return the normalization Orbifold Euler characteristic. This is denoted by D in [1].
        """
        if not self.is_Seifert:
            raise NotImplementedError("The Orbifold Euler Characteristic is only implemented for Seifert manifolds. This Plumbing manifold is not a Seifert manifold")
        if self._orbifold_euler == None:
            self._orbifold_euler = sum(self.Seifert_data)
        return self._orbifold_euler

    @property
    def norm_fac_euler(self):
        """
        Return the normalization factor for the Orbifold's Euler characteristic. This is denoted by D in [1].
        """
        if not self.is_Seifert:
            raise NotImplementedError("This function is only implemented for Seifert manifolds. This Plumbing manifold is not a Seifert manifold")
        if self._norm_fac_euler == None:
            p_coefficients = [d.denominator() for d in self.Seifert_data[1:]]
            factors = [(1/(self.orbifold_euler * p)).denominator() for p in p_coefficients]
            self._norm_fac_euler = lcm(factors)
        return self._norm_fac_euler


    @property
    def eff_inv_euler(self):
        """
        Return the effective inverse Euler characteristic. This is denoted by m in [1].
        """
        if self._eff_inv_euler == None:
            self._eff_inv_euler = -self.plumbing_matrix_inverse[0, 0]*self.norm_fac_euler
        return self._eff_inv_euler
            
    @property
    def coker(self):
        """
        Compute the cokernel of the plumbing matrix.
        """
        if self._coker == None:
            self._coker = [ a.lift() for a in ZZ**(self.vertex_count) / (self.plumbing_matrix*ZZ**(self.vertex_count))]
        return self._coker
    
    def L_norm(self, type_rank, ell):
        C_inv = np.array(cartan_matrix(type_rank).inverse())
        ell = np.array(ell)
        
        # Precompute the dot products
        ell_C_inv = np.dot(ell, C_inv)
        
        # Compute the total using matrix multiplication
        tot = np.sum(self.plumbing_matrix_inverse * np.dot(ell_C_inv, ell.T))
        
        return QQ(tot)
    
    def fugacity_variables(self,type_rank):
        """
        Compute the fugacity variables associated to a gauge group
        """
        fug_var = []
        for i in range(self._vertex_count):
            fug_var.append(matrix(var(f"z_{i}_{j+1}") for j in range(type_rank[1])).T)
        return fug_var
        
    def spin_c(self, type_rank, basis="weight"):
        """
            Returns the a list of inequivalent spin_c structures
        """
        
        type_rank = tuple(type_rank)
        # Construct b0 
        C = cartan_matrix(type_rank)
        rho = weyl_vector(type_rank)
        b0 = matrix([((a[0] % 2) * rho) for a in self.degree_vector])

        # Tensor coker with Lambda
        rk = type_rank[1]
        e = identity_matrix(rk)
        spin_c = list()
        for ei, v in itertools.product(e, self.coker):
            spin_c.append([x*ei for x in vector(v)])

        # Identify duplicates based on Weyl group transformations
        toRemove = set()
        for i, b in enumerate(spin_c):
            for g in weyl_group(type_rank):
                gb = [g*vector(x) for x in b]
                remove = False
                for bb in spin_c[:i]:
                    if vector(self.plumbing_matrix_inverse*(matrix(gb)-matrix(bb))) in ZZ^(self.vertex_count * rk):
                        remove = True
                        toRemove.add(i)
                        break
                if remove:
                    break
                    
        # Remove duplicates
        for i in reversed(sorted(toRemove)):
            del spin_c[i]
        spin_c = sorted([(matrix(b)+matrix(b0)) for b in spin_c])
        
        if basis == "weight":
            C = cartan_matrix(type_rank)
            spin_c = [matrix([C*v_r for v_r in v.rows()]) for v in spin_c]
                
        return spin_c

    
    def display(self):
        "Displays the plumbing graph."
        self._graph_plot.show()

    def _zhat_prefactor(self,type_rank):
        """
        Compute the prefactor for the Zhat invariant of A_rk.
        """
        pos_eigen = sum(np.greater(self.plumbing_matrix.eigenvalues(),0))    
        Phi = len(WeylGroup(type_rank).canonical_representation().positive_roots())
        sig = 2*pos_eigen - self.vertex_count
        norm_rho2 = weyl_lattice_norm(type_rank,weyl_vector(type_rank),basis="root")

        return Series.from_symbolic((-1)^(Phi*pos_eigen)*q^((3*sig-self.plumbing_matrix.trace())/2*norm_rho2))

    '''
    def zhat(self, type_rank, spin_c, n_powers = 2, basis="weight"):
        if basis == "root":
            raise NotImplementedError("Root basis not implemented yet")
        #if not self.is_unimodular:
            #raise NotImplementedError("Zhat only implemented for brieskorn spheres for now")
            
        # Collect data for the weyl group
        WG = [g.T for g in weyl_group(type_rank)]
        rho = cartan_matrix(type_rank) * weyl_vector(type_rank)
        
        
        # Set up the q-power contributions for leaves if leaves exist
        if 1 in list(self.degree_vector.T)[0]:
            leaf_exp  = [list(g * rho) for g in WG]
            leaf_pref = [det(g)        for g in WG]
        
        # Compute the weyl denominator expansion, if high degree nodes exists 
        if any([x > 2 for x in self.degree_vector.T[0]]):
            weyl_expansion = weyl_double_sided_expansion(type_rank, n_powers)
        
        # Compute the node contributions
        node_contributions_exponents = list()
        node_contributions_prefactors = list()
        
        for degree in self.degree_vector.T[0]:
            if degree == 0: # If the degree is zero, the contribution is just one
                node_contributions_exponents.append([[0]*rk])
                node_contributions_prefactors.append([1])
            elif degree == 1: # If the degree is one we have a leaf
                node_contributions_exponents.append(leaf_exp)
                node_contributions_prefactors.append(leaf_pref)
            else: #dddd If the degree is greater than one, use 
                new_powrs = list()
                new_coeffs = list()
                for expansion in weyl_expansion:
                    tot_exp = invert_powers(expansion.pow(degree-2))
                    powrs,coeffs = list(zip(*tot_exp.numerical))
                    new_powrs += powrs
                    new_coeffs += coeffs
                node_contributions_exponents.append(new_powrs)
                node_contributions_prefactors.append(new_coeffs)
        # Iterate over cartesian products
        zhat = list()
        make_products = lambda x: itertools.product(*x)
        for coeffs, ell in zip(*list(map(make_products,[node_contributions_prefactors,node_contributions_exponents]))):
            if matrix(self.plumbing_matrix_inverse * (matrix(ell)-matrix(spin_c))) in MatrixSpace(ZZ,self.vertex_count,type_rank[1]):
                zhat += [[[(-1/2*self.L_norm(type_rank,ell))],product(coeffs)]]
        return Series(zhat,variables=[q]) *self._zhat_prefactor(type_rank)
    '''

    def zhat_vec_np(self, type_rank, spin_c, n_powers = 2, basis="weight"):
        if basis == "root":
            raise NotImplementedError("Root basis not implemented yet")

        # Collect data for the weyl group
        WG = [g.T for g in weyl_group(type_rank)]
        rho = cartan_matrix(type_rank) * weyl_vector(type_rank)


        # Set up the q-power contributions for leaves if leaves exist
        if 1 in list(self.degree_vector.T)[0]:
            leaf_exp  = [tuple(g * rho) for g in WG]
            leaf_pref = [det(g)        for g in WG]

        # Compute the weyl denominator expansion, if high degree nodes exists 
        if any([x > 2 for x in self.degree_vector.T[0]]):
            weyl_expansion = weyl_double_sided_expansion(type_rank, n_powers)

        # Compute the node contributions
        node_contributions_exponents = list()
        node_contributions_prefactors = list()

        for degree in self.degree_vector.T[0]:
            if degree == 0 or degree == 2: # If the degree is zero or two, the contribution is just one
                node_contributions_exponents.append(([0]*type_rank[1]))
                node_contributions_prefactors.append([1])
            elif degree == 1: # If the degree is one we have a leaf
                node_contributions_exponents.append(leaf_exp)
                node_contributions_prefactors.append(leaf_pref)
            else: # If the degree is greater than one, use 
                new_powrs = list()
                new_coeffs = list()
                for expansion in weyl_expansion:
                    tot_exp = invert_powers(expansion.pow(degree-2))
                    powrs,coeffs = list(zip(*tot_exp.numerical))
                    new_powrs += powrs
                    new_coeffs += coeffs
                node_contributions_exponents.append(new_powrs)
                node_contributions_prefactors.append(new_coeffs)

        # Compute exponents for all cartesian products
        exponent_products = np.array(list(itertools.product(*node_contributions_exponents))).astype(np.float128)
        prefactor_products = np.array(list(itertools.product(*node_contributions_prefactors)))

        condition = np.all(np.mod((np.array(self.plumbing_matrix_inverse) @ (exponent_products - np.array(spin_c))),1) == 0,(1,2))

        exponent_contributing = exponent_products[condition]
        prefactor_contributing = prefactor_products[condition]
        
        dec_approx = len(str(np.max(np.abs(self.plumbing_matrix_inverse))))
        q_powers = self._compute_zhat(spin_c,type_rank,exponent_contributing,prefactor_contributing,dec_approx)
        return Series([[tuple(-1/2*p),product(c)] for p,c in zip(q_powers,prefactor_contributing)],variables=[var("q")])*self._zhat_prefactor(type_rank)*self._zhat_prefactor(type_rank)
    
            
    def _compute_zhat(self,spin_c, type_rank, exponent_products, prefactor_products, L_norm = L_norm_vectorized):
        
        condition = np.all(np.mod((np.array(self.plumbing_matrix_inverse) @ (exponent_products.astype(float) - np.array(spin_c))),1) == 0,(1,2))

        exponent_contributing = exponent_products[condition]
        prefactor_contributing = prefactor_products[condition]

        # Compute L_norms and prefactors
        dec_approx = len(str(np.max(np.abs(self.plumbing_matrix_inverse))))
        C_inv = np.array(cartan_matrix(type_rank).inverse(),dtype=np.float64)
        L_norms = L_norm(np.array(self.plumbing_matrix_inverse,dtype=np.float64),C_inv,exponent_contributing,dec_approx)
        prefactor_contributing = np.prod(prefactor_contributing,axis=1)

        # Convert to higher precision result if necessary
        q_powers = [QQ(-1/2*t) for t in L_norms]
        series_numerical = [[tuple(p),c] for p,c in zip(q_powers,prefactor_contributing)]

        return Series(series_numerical,variables=[var("q")])
    
    def _ell_setup(self, type_rank, n_powers):
        rk = type_rank[1]
        C = cartan_matrix(type_rank)
        rho = C*weyl_vector(type_rank)
        WG = [g.T for g in weyl_group(type_rank)]
        node_contributions_exponents = list()
        node_contributions_prefactors = list()
        # Compute the weyl denominator expansion, if high degree nodes exists 
        if any([x > 2 for x in self.degree_vector.T[0]]):
            weyl_expansion = weyl_double_sided_expansion(type_rank, n_powers)
        # Compute the node contributions
        for degree in self.degree_vector.T[0]:
            if degree == 0: # If the degree is zero, the contribution is just one
                node_contributions_exponents.append(([0]*rk))
                node_contributions_prefactors.append([1])
            elif degree == 1: # If the degree is one we have a leaf
                node_contributions_exponents.append([tuple(g * rho) for g in WG])
                node_contributions_prefactors.append([det(g)        for g in WG])
            else: # If the degree is greater than one, use 
                new_powrs = list()
                new_coeffs = list()
                for expansion in weyl_expansion:
                    tot_exp = invert_powers(expansion.pow(degree-2))
                    powrs,coeffs = list(zip(*tot_exp.numerical))
                    new_powrs += powrs
                    new_coeffs += coeffs
                node_contributions_exponents.append(new_powrs)
                node_contributions_prefactors.append(new_coeffs)
        exponent_products = np.array(list(itertools.product(*node_contributions_exponents))).astype(np.float64)
        prefactor_products = np.array(list(itertools.product(*node_contributions_prefactors)))
        return exponent_products, prefactor_products


    def zhat(self, type_rank, spin_c, order = 10, basis="weight",n_powers_start = 1, div_factor=100, method = "cython"):
        
        match method:
            case "vectorized":
                L_norm = L_norm_vectorized
            case "python":
                L_norm = L_norm_python
            case "cython":
                L_norm = L_norm_cython
            case _:
                raise ValueError("Method must be one of 'vectorized', 'python' or 'cython'")

        if basis == "root":
            raise NotImplementedError("Root basis not implemented yet")

        n_powers = n_powers_start
        zhat_A = Series([])
        zhat_B = Series([])
        max_power_computed = 0
        while max_power_computed <= order:
            # Compute the exponents, prefactors and the zhat invariant
            exponent_products, prefactor_products = self._ell_setup(type_rank, n_powers)
            zhat_A = self._compute_zhat(spin_c, type_rank, exponent_products, prefactor_products, L_norm=L_norm)*self._zhat_prefactor(type_rank)
            # Assess the order of the computed series
            exponent_products2, prefactor_products2 = self._ell_setup(type_rank, n_powers+1)
            new_terms = np.logical_not(np.all(np.isin(exponent_products2,exponent_products),axis=(1,2)))
            exponent_products2 = exponent_products2[new_terms]
            prefactor_products2 = prefactor_products2[new_terms]
            zhat_B = self._compute_zhat(spin_c, type_rank, exponent_products2, prefactor_products2, L_norm=L_norm)*self._zhat_prefactor(type_rank)
            max_power_computed = zhat_B.min_degree()
            n_powers += 1 * (int((order - max_power_computed)/div_factor)+1)

        return zhat_A.truncate(max_power_computed)


class Series():
    def __init__(self, numerical, variables=None):

        # Ensure numerical is properly constructed. It should be a list of monomials.
        # Each monomial should contain a list of powers and a coefficient.

        for monomial in numerical:
            error_message = "Series incorrectly constructed. Please check input."
            example = "Example: [[[1,2,3], 4], [0,3,4], -3] represents 4*z0^1*z1^2*z2^3 -3*z1^3*z2^4"
            assert type(monomial) in [list,tuple], error_message + f" Each monomial should be a list or a tuple." + example
            assert len(monomial) == 2, error_message + f" Each monomial should contain a list of powers and a coefficient." + example
            assert len(monomial[0]) == len(numerical[0][0]), error_message + " All monomials should have the same number of powers." + example
            assert type(monomial[0]) in [list,tuple], error_message + " The first element of each monomial should be a list or a tuple of powers." + example
            assert (not monomial in [Integer, Rational]) or monomial[1].is_constant(), error_message + " The second element of each monomial should be a number. " + example
            assert all([type(power) in [int,Rational] or type for power in monomial[0]]), error_message + " The powers of each monomial should be integers." + example
        
        self._n_variables = len(numerical[0][0]) if len(numerical) > 0 else 0
        self._numerical = sorted([[tuple(p),c] for p,c in numerical if c != 0], key=lambda x: sum(x[0]))
        self._variables = variables if self._n_variables > 0 else []
        self._dictionary = None
        self._symbolic = None
        self._powers = None

    @classmethod
    def from_symbolic(cls, series):
        """
            Create a Series object from a symbolic expression.
        """
        
        try:
            variables = series.variables()
            series = series.expand()
        except:
            variables = None
    
        numerical = multi_poly_coeffs(series)
        return cls(numerical, variables)
    
    @classmethod
    def from_dictionary(cls, dictionary,variables=None):
        """
            Create a Series object from a dictionary.
        """
        numerical = list(dictionary.items())
        return cls(numerical, variables)

    @classmethod
    def from_function(cls, function, dimension, range, variables=None):
        """
            Create a Series object from a function.
        """
        raise NotImplementedError
        return cls(series, variables)

    @property
    def variables(self):
        """
            Return the variables of the series.
        """
        if self._variables == None:
            self._variables = [var(f"z_{i}") for i in range(self.n_variables)]
        return self._variables

    @property
    def n_variables(self):
        """
            Return the number of variables in the series.
        """
        return self._n_variables
    
    @property
    def max_degree(self, key=None):
        # Return the maximum order of the series for series with one variable
        # If more variables and key=None throw error
        if self.n_variables > 1 and key == None:
            raise ValueError("Please specify a key for the maximum order.")
        if key == None and self.n_variables == 1:
            key = max

        max_order = max([key(monomial[0]) for monomial in self.numerical])    
        return max_order
    
    @property
    def numerical(self):
        """
        Return the numerical representation of the series.
        """
        return self._numerical

    @property
    def dictionary(self):
        """
        Return the dictionary representation of the series.
        """
        if self._dictionary == None:
            self._dictionary = dict()
            for powers, coeff in self.numerical:
                self._dictionary[tuple(powers)] = coeff
        return self._dictionary
    
    @property
    def symbolic(self):
        """
        Return the symbolic representation of the series.
        """
        if self._symbolic == None:
            self._symbolic = 0
            for powers,coeff in self.numerical:
                self._symbolic += coeff * product([v**p for v,p in zip(self.variables,powers)])
        return self._symbolic
    
    @property
    def n_terms(self):
        return len(self.numerical)
    
    @property
    def powers(self):
        if self._powers == None:
            self._powers = [monomial[0] for monomial in self.numerical]
        return self._powers
    
    @property
    def min_degree(self, key=None):
        # Return the minimum order of the series for series with one variable
        # If more variables and key=None throw error
        if self.n_variables > 1 and key == None:
            raise ValueError("Please specify a key for the minimum order.")
        if len(self.numerical) == 0:
            return 0
        if key == None and self.n_variables == 1:
            key = min
        min_order = min([key(monomial[0]) for monomial in self.numerical])    
        return min_order

    
    def _make_none(self):
        """
        Make all properties None.
        """
        self._numerical = None
        self._dictionary = None
        self._symbolic = None
        self._n_variables = None
        self._powers = None
    
    def _multiply_series(self, other):
        """
        Multiply two multivariate polynomials using NumPy operations.
        Each polynomial is represented as a list of (exponents, coefficient) tuples,
        where exponents are arrays representing the powers of each variable.
        """
        exps1, coeffs1 = zip(*self.numerical)
        exps2, coeffs2 = zip(*other.numerical)
        
        exps1 = np.array(exps1)
        coeffs1 = np.array(coeffs1)
        exps2 = np.array(exps2)
        coeffs2 = np.array(coeffs2)
        
        # Broadcasting addition of exponents
        new_exps = exps1[:, np.newaxis, :] + exps2[np.newaxis, :, :]
        new_coeffs = coeffs1[:, np.newaxis] * coeffs2[np.newaxis, :]
        
        # Reshape the arrays to be a list of tuples
        new_exps = new_exps.reshape(-1, exps1.shape[1])
        new_coeffs = new_coeffs.flatten()
        
        # Sum the coefficients of the same exponents
        result_dict = {}
        for exp, coeff in zip(map(tuple, new_exps), new_coeffs):
            if exp in result_dict:
                result_dict[exp] += coeff
            else:
                result_dict[exp] = coeff
        
        # Convert the result to the required format, filtering out zero coefficients
        result = [(list(exp), coeff) for exp, coeff in result_dict.items() if coeff != 0]
        return Series(result,self.variables)
    

    def _fft_multiply_series(self, other):
        """
        Multiply two multivariate polynomials using FFT.
        Each polynomial is represented as a list of (exponents, coefficient) tuples,
        where exponents are arrays representing the powers of each variable.
        The shape parameter defines the size of the FFT grid.
        """
        # Define the shape of the FFT grid (based on the maximum exponents +1)
        shape = tuple(max(max(e) for e in zip(*self.powers, *other.powers)) + 1 for _ in range(self.n_variables))
        
        # Create coefficient grids
        coeff_grid1 = np.zeros(shape, dtype=complex)
        coeff_grid2 = np.zeros(shape, dtype=complex)
        
        for exps, coeff in self.numerical:
            coeff_grid1[tuple(exps)] = coeff
        
        for exps, coeff in other.numerical:
            coeff_grid2[tuple(exps)] = coeff
        
        # Perform FFT on both grids
        fft_grid1 = np.fft.fftn(coeff_grid1)
        fft_grid2 = np.fft.fftn(coeff_grid2)
        
        # Point-wise multiply the FFT results
        fft_result = fft_grid1 * fft_grid2
        
        # Perform inverse FFT to get the result polynomial coefficients
        result_grid = np.fft.ifftn(fft_result).real
        
        # Extract non-zero coefficients and their exponents
        result = [(list(index), coeff) for index, coeff in np.ndenumerate(result_grid) if coeff != 0]
        
        return result

    
    def __add__(self, other):
        if self.n_variables != other.n_variables:
            raise ValueError("The number of variables of the two series do not match.")
        
        # Combine the two series
        new_dict = join_dicts_with_function(self.dictionary, other.dictionary, lambda x,y: x+y)
        new_numerical = list(new_dict.items())
        return Series(new_numerical, self.variables)
    
    def __sub__(self, other):
        if self.n_variables != other.n_variables and 0 not in [self.n_variables, other.n_variables]:
            raise ValueError("The number of variables of the two series do not match.")
        
        # Combine the two series
        new_dict = join_dicts_with_function(self.dictionary, other.dictionary, lambda x,y: x-y)
        new_numerical = [(k,v) for k,v in new_dict.items() if v != 0]
        return Series(new_numerical, self.variables)
    
    def __mul__(self, other):
        if self.n_variables != other.n_variables:
            return Series.from_symbolic(self.symbolic * other.symbolic)
            raise ValueError("The number of variables of the two series do not match.")
        
        # Decide which multiplication method to use based on the number of terms
        if True or self.n_terms * other.n_terms < 1000:
            #print("Using NumPy multiplication")
            res = Series.from_symbolic(self.symbolic * other.symbolic)
        else:
            # For some reason fft is super slow
            print("Using FFT multiplication")
            res = self._fft_multiply_series(other)
        return res

    def __repr__(self):
        # Check if all powers are equal mod 1
        return self.show(max_terms = 10)
    
    def show(self,max_terms=10):
        # Check if all powers are equal mod 1
        res = str("")
        if self.variables == (var("q"),):
            rational_part = lambda x: x - floor(x)
            if all(rational_part(self.powers[0][0]) == rational_part(x[0]) for x in self.powers):
                overall = self.powers[0][0]
                if overall != 0:
                    res += f"q^({overall})(" 
                n_added = 0
                for powers, coeff in self.numerical:
                    if coeff == 0:
                        continue
                    if n_added >= max_terms:
                        break
                    if coeff < 0 and powers != self.powers[0]:
                        res = res[:-3] + f" - "
                    if powers[0] == overall:
                        res +=str(coeff) + " + "
                    else:
                        res += f"{abs(coeff)}q^({powers[0]-overall}) + "
                    n_added += 1
                try:
                    res += f"O(q^{self.numerical[max_terms][0][0]-overall})"
                except:
                    res += f"O(q^{self.numerical[n_added-1][0][0]+1-overall})"
                if overall != 0:
                    res += f")"
                return res
        return str(self.symbolic)
    
    def pow(self,power):
        assert type(power) in [int, np.int64, Integer], "The power should be a positive integer."
        assert power >= 0, "The power should be a positive integer."
        
        if power == 0:
            return Series.from_dictionary({tuple([0]*self.n_variables):1},self.variables)
        if power == 1:
            return self
        return Series.from_symbolic(self.symbolic**power)
    
    def truncate(self, max_order):
        """
        Truncate the series to a maximum order.
        """
        new_numerical = [monomial for monomial in self.numerical if sum(monomial[0]) < max_order]
        return Series(new_numerical, self.variables)
    

        
        

class Seifert():
    
    #
    def __init__(self, SeifertData, qvar=var('q')):
        """
        Create a Seifert manifold.

        Implements the "Seifert" constructor. Currently working for Seifert manifolds with three and four exceptional fibers.

        INPUT:
        -   ``SeifertData`` -- list; List containing the Seifert manifold data, as [b,q_1,p_1,...,p_n,q_n]
        -   ``qvar`` -- q variable for q series

        EXAMPLES::
            sage: S = Seifert([-1, 1, 2, 1, 3, 1, 5]);S
            Seifert manifold with 3 exceptional fibers.
            Seifert data:
            [-1, 1, 2, 1, 3, 1, 5]
            Plumbing Matrix:
            [-1  1  1  1]
            [ 1 -2  0  0]
            [ 1  0 -3  0]
            [ 1  0  0 -5]
            D: 1, m: 30, det(M): -1

            sage: S = Seifert([-1, 1, 2, 1, 3, 1, 5, 1, 7]);S
            Seifert manifold with 4 exceptional fibers.
            Seifert data:
            [-1, 1, 2, 1, 3, 1, 5, 1, 7]
            Plumbing Matrix:
            [-1  1  1  1  1]
            [ 1 -2  0  0  0]
            [ 1  0 -3  0  0]
            [ 1  0  0 -5  0]
            [ 1  0  0  0 -7]
            D: 37, m: 210, det(M): 37
        """
        self.SeifertData = SeifertData
        self.b = SeifertData[0]
        self.q = [SeifertData[x] for x in range(1, len(SeifertData), 2)]
        # p is the order of singular fibers
        self.p = [SeifertData[x] for x in range(2, len(SeifertData), 2)]
        self.f = len(self.q)
        assert self.f == 3 or self.f == 4, "pySeifert only currently supports three or four fibers"
        self.M = self._plumbing_matrix()
        self.L = self.M.ncols()
        self.Mdet = self.M.det()
        self.Minv = self.M.inverse()
        self.d = 1/GCD(self.Minv[0])
        self.m = abs(self.Minv[0, 0]*self.d)
        self.deg = self._degrees()  # returns 2-deg(v) of plumbing matrix
        self._legs()
        self.A = vector([x if y == 1 else 0 for x,
                        y in zip(self.Minv[0], self.deg)])
        self.qvar = qvar
        self.Coker = self._coker()

    def __repr__(self):
        """
        INPUT:
        -   ``self`` -- Seifert; Seifert namifold

        EXAMPLE::
            sage: S = Seifert([-1, 1, 2, 1, 3, 1, 5, 1, 7]);S
            Seifert manifold with 4 exceptional fibers.
            Seifert data:
            [-1, 1, 2, 1, 3, 1, 5, 1, 7]
            Plumbing Matrix:
            [-1  1  1  1  1]
            [ 1 -2  0  0  0]
            [ 1  0 -3  0  0]
            [ 1  0  0 -5  0]
            [ 1  0  0  0 -7]
            D: 37, m: 210, det(M): 37
        """
        return 'Seifert manifold with {} exceptional fibers.\nSeifert data:\n{}\nPlumbing Matrix:\n{}\nD: {}, m: {}, det(M): {}'.format(self.f, self.SeifertData, self.M, self.d, self.m, self.Mdet)

    def _latex_(self):
        """
        Print latex name
        """
        latex_name = f"M\\left({self.b};"
        for i in range(self.f):
            latex_name += "\\frac{"+str(self.q[i])+"}{"+str(self.p[i])+"}, "
        latex_name = latex_name[:-2] + "\\right)"
        return latex_name

    def _plumbing_matrix(self):
        r"""
        Compute the plumbing matrix of self.

        Some code taken from
        https://share.cocalc.com/share/d1efa37e-be6a-40f6-80c7-8a34201d7c4e/PlumbGraph.sagews?viewer=share
        on July 4th 2021
        """
        l = [len(_continued_fraction(self.SeifertData[2*i]/self.SeifertData[2*i-1]))
             for i in range(1, self.f+1)]
        M = matrix(1+sum(l))
        M[0, 0] = self.b
        for j in range(len(l)):
            for k in range(l[j]):
                if k == 0:
                    M[0, 1+sum(l[:j])] = 1
                    M[1+sum(l[:j]), 0] = 1
                    M[1+k+sum(l[:j]), 1+k+sum(l[:j])] = (-1)*_continued_fraction(
                        self.SeifertData[2*j+2]/self.SeifertData[2*j+1])[k]
                else:
                    M[1+k+sum(l[:j]), k+sum(l[:j])] = 1
                    M[k+sum(l[:j]), 1+k+sum(l[:j])] = 1
                    M[1+k+sum(l[:j]), 1+k+sum(l[:j])] = (-1)*_continued_fraction(
                        self.SeifertData[2*j+2]/self.SeifertData[2*j+1])[k]
        return M

    def _degrees(self):
        """
        Compute the degrees of the vertices in the order that they appear in the plumbing matrix of self. Returns 2-deg(v)
        """
        deg = list()
        for i, row in enumerate(list(self.M)):
            deg.append(2-(sum(row)-row[i]))
        return deg

    def _coker(self):
        """
        Compute the cokernel of the plumbing matrix (of self).
        """

        if self.Mdet**2 == 1:
            return [[0]*self.L]
        Coker = []
        for v in range(self.L):
            vec = vector([0]*self.L)
            for i in range(abs(self.Mdet)):
                vec[v] = i+1
                new = [x-floor(x) for x in self.Minv*vec]
                if new not in Coker:
                    Coker.append(new)
        return Coker

    def _legs(self):
        """
        Compute the lengths of the legs of the plumbing graph.
        """
        llen = 1
        self.legs = list()
        for d in reversed(self.deg[1:-1]):
            if d == 1:
                self.legs.append(llen)
                llen = 0
            llen += 1
        self.legs.append(llen)
        self.legs.reverse()

    def _zhat_prefactor(self,type_rank, qvar=None):
        """
        Compute the prefactor for the Zhat invariant of A_rk.
        """
        if qvar == None:
            qvar = self.qvar
        pos_eigen = sum(np.greater(self.M.eigenvalues(), 0))
        Phi = len(WeylGroup(type_rank).canonical_representation().positive_roots())
        sig = 2*pos_eigen - (self.L)
        norm_rho2 = weyl_lattice_norm(type_rank,weyl_vector(type_rank),basis="root")

        return (-1) ^ (Phi*pos_eigen)*q^((3*sig-self.M.trace())/2*norm_rho2)

    def delta(self, type_rank, wilson=None, basis="weight"):
        """
        Compute the q-exponent delta as in equation (4.12) of [1]. Works for rk=1,2.

        INPUT:
        -   ``type_rank`` - [str,int]; Lie group identifier

        EXAMPLES::
            sage: S = Seifert([-1,1,2,1,3,1,5]);S.delta(["A",2])
            31/30
            sage: S.delta(["D",6])
            341/12

        """ 
        if basis == "root":
            cart = cartan_matrix(type_rank)
            wilson = [ cart * nui for nui in wilson]
        if wilson == None:
            rhov = [cartan_matrix(type_rank)*weyl_vector(type_rank) if d==1 else 0 for d in self.deg ]
        else:
            wil_iter = iter(wilson)
            rhov = [cartan_matrix(type_rank)*weyl_vector(type_rank)  + vector(next(wil_iter)) if d == 1 else 0 for d in self.deg]

        return sum([weyl_lattice_norm(type_rank,rhov[i], basis="weight")*1/2*(self.Minv[0, i] ^ 2/self.Minv[0, 0]-self.Minv[i, i]) for i, d in enumerate(self.deg) if d == 1])
        
    def boundary_conditions(self, type_rank, basis=None):
        """
        Compute representatives for the set of boundary conditions for the Seifert manifold with respect to A_rk group, as in equation (4.4) of [1]. Works for rk=1,2.

        INPUT:
        -   ``type_rank`` - [str,int]; Lie group identifier
        -   ``basis`` -- str; basis in which the bs are to be outputted.

        EXAMPLES::
            sage: S = Seifert([-1,1,2,1,3,1,5])
            ....: S.boundary_conditions(["A",2], basis = "weight")
            [[(-1, -1), (1, 1), (1, 1), (1, 1)]]

            sage: S = Seifert([-1,1,2,1,2,1,2])
            ....: S.boundary_conditions(["A",2], basis = "root")
            [[(-1, -1), (1, 1), (1, 1), (1, 1)],
             [(0, -1), (0, 1), (0, 1), (1, 1)],
             [(0, -1), (0, 1), (1, 1), (0, 1)],
             [(0, -1), (1, 1), (0, 1), (0, 1)]]

            sage: S = Seifert([-1,1,2,1,2,1,2])
            ....: S.boundary_conditions(["D",4],basis="root")
            [[(-3, -5, -3, -3), (3, 5, 3, 3), (3, 5, 3, 3), (3, 5, 3, 3)],
             [(-2, -5, -3, -3), (2, 5, 3, 3), (2, 5, 3, 3), (3, 5, 3, 3)],
             [(-2, -5, -3, -3), (2, 5, 3, 3), (3, 5, 3, 3), (2, 5, 3, 3)],
             [(-2, -5, -3, -3), (3, 5, 3, 3), (2, 5, 3, 3), (2, 5, 3, 3)]]

        """

        if basis == None:
            warnings.warn("No basis is specified, weight is assumed")
            basis = "weight"
        
        # If Brieskorn sphere, just return the b0
        rho = weyl_vector(type_rank)
        b0 = [d*rho for d in self.deg]
        if self.Mdet**2 == 1:
            if basis == "weight":
                cart = cartan_matrix(type_rank)
                b0 = [d*cart*rho for d in self.deg]
            return [b0]

        # Tensor (Coker otimes Lambda)
        rk = type_rank[1]
        e = identity_matrix(rk)

        boundary_conditions = list()
        for ei, v in itertools.product(e, self.Coker):
            boundary_conditions.append([x*ei for x in self.M*vector(v)])
        # Calculate the orbit of each connection w/r. to the Weyl group.
        # If an orbit element is already present in boundary_conditions, then remove connection.
        toRemove = set()
        for i, b in enumerate(boundary_conditions):
            for g in weyl_group(type_rank):
                gb = [g*vector(x) for x in b]
                remove = False
                for bb in boundary_conditions[:i]:
                    if vector(self.Minv*(matrix(gb)-matrix(bb))) in ZZ ^ (self.L*rk):
                        remove = True
                        toRemove.add(i)
                        break
                if remove:
                    break
        
        for i in reversed(sorted(toRemove)):
            del boundary_conditions[i]

        boundary_conditions = sorted(
            [list(matrix(b)+matrix(b0)) for b in boundary_conditions])


        if basis == "weight":
            C = cartan_matrix(type_rank)
            for i in range(len(boundary_conditions)):
                b = [C*v for v in boundary_conditions[i]]
                boundary_conditions[i] = b
        return boundary_conditions

    def S_set(self, type_rank, whr, b, basis="root"):
        """
        Compute the set \\kappa_{\\hat{w};\\vec{\\underline{b}}}, as in equation (4.28) of [1].

        INPUTS:
        -   ``type_rank`` - [str,int]; Lie group identifier
        -   ``whr`` -- list of length self.L of vectors of rank rk; list of lattice vectors. whr[i] should be the zero vector if self.deg[i] is different than 1, w_i*rho otherwise
        -   ``b`` --  list of length self.L of vectors of rank rk; boundary condition
        -   ``basis`` -- basis in which whr and b are given

        EXAMPLES::
        sage: S = Seifert([0,1,3,1,2,1,6])
        ....: b = S.boundary_conditions(["A",2], basis = "root")[1]
        ....: rho = vector([1,1])
        ....: whr = [identity_matrix(2)*rho if d == 1 else matrix(2)*rho for d in S.deg]
        ....: S.S_set(["A",2],whr,b)
        [(1, 5)]

        """
        if basis == "weight":
            cart_i = cartan_matrix(type_rank).inverse()
            whr = [cart_i*vector(v) for v in whr]
            b = [cart_i*vector(v) for v in b]

        rk = type_rank[1]
        rho = weyl_vector(type_rank)
        whr = matrix(whr)
        k_list = list()
        MS = MatrixSpace(ZZ, whr.nrows(), whr.ncols())
        eps = self.f % 2
        for k in itertools.product(range(self.d),repeat=rk):
            kappa = matrix([vector(k)+eps*rho]+[[0]*rk]*(whr.nrows()-1))
            if self.Minv*(kappa+whr-matrix(b)) in MS:
                k_list += [vector(k)]
        if basis == "weight":
            cart = cartan_matrix(type_rank)
            k_list = [cart*v for v in k_list]
        return k_list

    def s_values(self, type_rank, b, basis="weight", nu=None, wilVert=0):
        """
        Compute the set of \\vec s values, and their respective \\kappa_{\\hat w; \\vec{\\underline b}} and Weyl length, as in equations (4.28) (\\kappa) and (4.36) (\\vec s) of [1].

        INPUT:
        -   ``type_rank`` - [str,int]; Lie group identifier
        -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
        -   ``basis`` -- String; basis in which b is given.
        -   ``WGsimb`` -- Bool; if True, print a hat-w in symbolic form for every s value
        -   ``nu`` -- Vector; Highest weight of Wilson line operator to be attached at an end node. If None, no Wilson operator is attached.
        -   ``wilVert`` -- Integer; leg at which to attach the Wilson operator

        Example:

        
        sage: S = Seifert([0,1,3,1,2,1,6])
        ....: b = S.boundary_conditions(["A",2],basis = "root")[0]
        ....: S.s_values(["A",2],b,"weight")
        [[-1, (-6, -6), (5, 5)],
         [1, (0, 0), (4, 4)],
         [1, (6, -12), (5, 5)],
         [-1, (0, 0), (6, 3)],
         [1, (-12, 6), (5, 5)],
         [-1, (0, 0), (3, 6)],
         [1, (0, 0), (10, -5)],
         [-1, (-6, 12), (5, 5)],
         [1, (0, 0), (-5, 10)],
         [-1, (12, -6), (5, 5)],
         [-1, (0, 0), (0, 0)],
         [1, (6, 6), (5, 5)]]

        sage: S = Seifert([0,1,3,1,2,1,6])
        ....: b = S.boundary_conditions(["A",1],basis = "weight")[0]
        ....: S.s_values(["A",1],b,"weight")
        [[-1, (-6), (10)], [1, (6), (10)]]

        """

        rk = type_rank[1]

        if nu == None:
            nu = vector([0]*rk)

        WG = weyl_group(type_rank)
        if basis == "weight":
            WG = [g.transpose() for g in WG]
        WL = weyl_lengths(type_rank)
        it = iter([[1]*len(WG) if d == wilVert else [0]*len(WG)
                  for d in range(self.f)])
        
        rho = weyl_vector(type_rank)
        if basis == "weight":
            rho = cartan_matrix(type_rank)*rho
        
        WGr = [[g*(rho+i*nu) for g, i in zip(WG, next(it))] if d ==
               1 else [vector([0]*rk)] for d in self.deg]

        whrl = list(itertools.product(*WGr))
        whlenl = itertools.product(reversed(WL), repeat=self.f)
        alls_values = list()
        i = 1
        tl = len(whrl)
        for whr, whlen in zip(whrl, whlenl):
            i+=1
            S_set = self.S_set(type_rank, whr, b, basis=basis)
            if S_set != []:
                new = [product(whlen),
                       (-self.d*self.Minv[0]*matrix(whr)), S_set[0]]
                alls_values.append(new)
        return alls_values

    def chi_tilde(self, type_rank, wh, b, expMax, basis="weight", qvar=None):
        """
            Compute the chi_tilde q-series like in equation (4.34) of [1].

            INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- q variable for q series

            EXAMPLES::
                sage: S = Seifert([0,1,3,1,2,1,6]);
                ....: wh = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg];
                ....: b = S.boundary_conditions(2,basis = "weight")[0];
                ....: expMax = 20;
                ....: S.chi_tilde(["A",2],wh, b, expMax, basis = "weight")
                (z0^5*z1^5 - z0^10/z1^5 - z1^10/z0^5 + z0^5/z1^10 + z1^5/z0^10 - 1/(z0^5*z1^5))*q^17/(z0*z1 - z0^2/z1 - z1^2/z0 + z0/z1^2 + z1/z0^2 - 1/(z0*z1)) - q^5

                sage: S = Seifert([0,1,3,1,2,1,6]);
                ....:  wh = [identity_matrix(1) if d == 1 else matrix(1) for d in S.deg];
                ....:  b = S.boundary_conditions(["A",1],basis = "weight")[0];
                ....:  expMax = 20;
                ....:  S.chi_tilde(["A",1],wh, b, expMax, basis = "weight")
                -q^(5/4)

                sage: S = Seifert([0,1,3,1,2,1,6]);
                ....: wh = [identity_matrix(3) if d == 1 else matrix(3) for d in S.deg];
                ....: b = S.boundary_conditions(["A",3],basis = "weight")[0];
                ....: expMax = 20;
                ....: S.chi_tilde(["A",3],wh, b, expMax, basis = "weight")
                -(z0^4*z1*z2^4 - z0^5*z2^5/z1 - z0^4*z1^5/z2^4 + z0^9*z2/z1^5 - z1^5*z2^4/z0^4 + z0*z2^9/z1^5 + z0^5*z1^4/z2^5 - z0^9/(z1^4*z2) + z1^4*z2^5/z0^5 - z2^9/(z0*z1^4) + z1^9/(z0^4*z2^4) - z0^5*z2^5/z1^9 - z1^9/(z0^5*z2^5) + z0^4*z2^4/z1^9 - z0*z1^4/z2^9 + z0^5/(z1^4*z2^5) - z1^4*z2/z0^9 + z2^5/(z0^5*z1^4) + z1^5/(z0*z2^9) - z0^4/(z1^5*z2^4) + z1^5/(z0^9*z2) - z2^4/(z0^4*z1^5) - z1/(z0^5*z2^5) + 1/(z0^4*z1*z2^4))*q^(25/2)/(z0*z1*z2 - z0^2*z2^2/z1 - z0*z1^2/z2 + z0^3*z2/z1^2 - z1^2*z2/z0 + z0*z2^3/z1^2 + z0^2*z1/z2^2 - z0^3/(z1*z2) + z1^3/(z0*z2) - z0^2*z2^2/z1^3 + z1*z2^2/z0^2 - z2^3/(z0*z1) - z0*z1/z2^3 + z0^2/(z1*z2^2) - z1^3/(z0^2*z2^2) + z0*z2/z1^3 - z1*z2/z0^3 + z2^2/(z0^2*z1) + z1^2/(z0*z2^3) - z0/(z1^2*z2) + z1^2/(z0^3*z2) - z2/(z0*z1^2) - z1/(z0^2*z2^2) + 1/(z0*z1*z2)) + q^(25/2)
        """

        rk = type_rank[1]
        if qvar == None:
            qvar = self.qvar

        cart_i = cartan_matrix(type_rank).inverse()
        if basis == "weight":
            wh = [w.transpose() for w in wh]
            b = [cart_i*v for v in b]
            basis = "root"
        rho = weyl_vector(type_rank)
        whr = [w*rho for w in wh]

        Aw = 1/self.Minv[0, 0]*self.A*matrix(whr)
        kappa = self.S_set(type_rank, whr, b, basis=basis)
        if kappa == []:
            return 0
        else:
            kappa = kappa[0]
        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)

        D_r = Delta(rho)
        eps = self.f % 2
        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [self.delta(type_rank)+1/(2*self.d*self.m) *
                    weyl_lattice_norm(type_rank, self.m*(v+Aw), basis=basis)]
            prefl = [Delta(v)/D_r^(self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)

    def chi_tilde_wilson_end_old(self, type_rank, wh, b, expMax, nu, leg, basis="weight", qvar=None):
        """
        Compute the tilde_chi q-series with Wilson operator insertion at an end node, as described in section 4.3 of [1].

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``nu`` -- Vector; highest weight of representation
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- Variable; q variable for q series

        EXAMPLES::
            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: wh = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg];
            ....: b = S.boundary_conditions(["A",2],basis = "weight")[0];
            ....: expMax = 100;
            ....: nu, leg = vector([3,3]), 0;
            ....: S.chi_tilde_wilson_end(["A",2],wh, b, expMax, nu, leg, basis = "weight", qvar=None)
            (z0^4*z1^4 + z0^8/z1^4 + z1^8/z0^4 + z0^4/z1^8 + z1^4/z0^8 + 1/(z0^4*z1^4))*q^(175/9)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1)) + (z0^2*z1^2 + z0^4/z1^2 + z1^4/z0^2 + z0^2/z1^4 + z1^2/z0^4 + 1/(z0^2*z1^2))*q^(103/9)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1))

            sage: S = Seifert([-1,1,3,1,4,1,5]);
            ....: wh = [identity_matrix(3) if d == 1 else matrix(3) for d in S.deg];
            ....: b = S.boundary_conditions(["A",3],basis = "weight")[0];
            ....: expMax = 200;
            ....: nu, leg = vector([1,2,1]), 0;
            ....: S.chi_tilde_wilson_end(["A",3],wh, b, expMax, nu, leg, basis = "weight", qvar=None)
            (z0*z1^5*z2^5 - z0^6*z2^10/z1^5 - z1^6*z2^5/z0 + z0^5*z2^11/z1^6 - z0*z1^10/z2^5 + z0^11*z2^5/z1^10 + z1^11/(z0*z2^5) - z0^10*z2^6/z1^11 + z1*z2^10/z0^6 - z2^11/(z0^5*z1) + z0^6*z1^5/z2^10 - z0^11/(z1^5*z2^5) - z0^5*z1^5/z2^11 + z0^10/(z1^5*z2^6) - z1^11/(z0^6*z2^10) + z0^5*z2/z1^11 - z1*z2^5/z0^11 + z2^6/(z0^10*z1) + z1^10/(z0^5*z2^11) - z0^5/(z1^10*z2) + z1^6/(z0^11*z2^5) - z2/(z0^5*z1^6) - z1^5/(z0^10*z2^6) + 1/(z0^5*z1^5*z2))*q^(822927/4394)/(z0*z1*z2 - z0^2*z2^2/z1 - z0*z1^2/z2 + z0^3*z2/z1^2 - z1^2*z2/z0 + z0*z2^3/z1^2 + z0^2*z1/z2^2 - z0^3/(z1*z2) + z1^3/(z0*z2) - z0^2*z2^2/z1^3 + z1*z2^2/z0^2 - z2^3/(z0*z1) - z0*z1/z2^3 + z0^2/(z1*z2^2) - z1^3/(z0^2*z2^2) + z0*z2/z1^3 - z1*z2/z0^3 + z2^2/(z0^2*z1) + z1^2/(z0*z2^3) - z0/(z1^2*z2) + z1^2/(z0^3*z2) - z2/(z0*z1^2) - z1/(z0^2*z2^2) + 1/(z0*z1*z2))
        """

        rk = type_rank[1]
        if qvar == None:
            qvar = self.qvar

        if basis == "weight":
            wh = [w.transpose() for w in wh]
            cart_i = cartan_matrix(type_rank).inverse()
            nu = cart_i*nu
            b = [cart_i*v for v in b]
            basis = "root"

        rho = weyl_vector(type_rank)
        whr = [w*rho for w in wh]
        ends = [i for i,d in enumerate(S.deg) if d == 1]
        end_ind = ends[leg]
        whr[end_ind] += wh[end_ind]*nu

        Aw = -1/self.m*self.A*matrix(whr)
        mod_b = copy(b)
        mod_b[end_ind] += nu
        kappa = self.S_set(type_rank, whr, mod_b, basis=basis)
        if kappa == []:
            return 0
        else:
            kappa = kappa[0]
        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)
        D_r = Delta(rho)

        eps = self.f % 2
        d = self.delta(type_rank) + (weyl_lattice_norm(type_rank,rho+nu, basis=basis)- \
            weyl_lattice_norm(type_rank, rho, basis=basis)) / \
            2*(self.Minv[0, end_ind] ^ 2 /
               self.Minv[0, 0]-self.Minv[end_ind, end_ind])


        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [d+self.m/(2*self.d)*weyl_lattice_norm(type_rank, v+Aw, basis=basis)]
            prefl = [Delta(v)/D_r ^ (self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)


    def chi_tilde_wilson_end(self, type_rank, wh, b, expMax, nu, basis="weight", qvar=None):
        """
        Compute the tilde_chi q-series with Wilson operator insertion at an end node, as described in section 4.3 of [1].

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``nu`` -- Vector; highest weight of representation
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- Variable; q variable for q series

        EXAMPLES::
            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: wh = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg];
            ....: b = S.boundary_conditions(["A",2],basis = "weight")[0];
            ....: expMax = 100;
            ....: nu, leg = vector([3,3]), 0;
            ....: S.chi_tilde_wilson_end(["A",2],wh, b, expMax, nu, leg, basis = "weight", qvar=None)
            (z0^4*z1^4 + z0^8/z1^4 + z1^8/z0^4 + z0^4/z1^8 + z1^4/z0^8 + 1/(z0^4*z1^4))*q^(175/9)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1)) + (z0^2*z1^2 + z0^4/z1^2 + z1^4/z0^2 + z0^2/z1^4 + z1^2/z0^4 + 1/(z0^2*z1^2))*q^(103/9)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1))

            sage: S = Seifert([-1,1,3,1,4,1,5]);
            ....: wh = [identity_matrix(3) if d == 1 else matrix(3) for d in S.deg];
            ....: b = S.boundary_conditions(["A",3],basis = "weight")[0];
            ....: expMax = 200;
            ....: nu, leg = vector([1,2,1]), 0;
            ....: S.chi_tilde_wilson_end(["A",3],wh, b, expMax, nu, leg, basis = "weight", qvar=None)
            (z0*z1^5*z2^5 - z0^6*z2^10/z1^5 - z1^6*z2^5/z0 + z0^5*z2^11/z1^6 - z0*z1^10/z2^5 + z0^11*z2^5/z1^10 + z1^11/(z0*z2^5) - z0^10*z2^6/z1^11 + z1*z2^10/z0^6 - z2^11/(z0^5*z1) + z0^6*z1^5/z2^10 - z0^11/(z1^5*z2^5) - z0^5*z1^5/z2^11 + z0^10/(z1^5*z2^6) - z1^11/(z0^6*z2^10) + z0^5*z2/z1^11 - z1*z2^5/z0^11 + z2^6/(z0^10*z1) + z1^10/(z0^5*z2^11) - z0^5/(z1^10*z2) + z1^6/(z0^11*z2^5) - z2/(z0^5*z1^6) - z1^5/(z0^10*z2^6) + 1/(z0^5*z1^5*z2))*q^(822927/4394)/(z0*z1*z2 - z0^2*z2^2/z1 - z0*z1^2/z2 + z0^3*z2/z1^2 - z1^2*z2/z0 + z0*z2^3/z1^2 + z0^2*z1/z2^2 - z0^3/(z1*z2) + z1^3/(z0*z2) - z0^2*z2^2/z1^3 + z1*z2^2/z0^2 - z2^3/(z0*z1) - z0*z1/z2^3 + z0^2/(z1*z2^2) - z1^3/(z0^2*z2^2) + z0*z2/z1^3 - z1*z2/z0^3 + z2^2/(z0^2*z1) + z1^2/(z0*z2^3) - z0/(z1^2*z2) + z1^2/(z0^3*z2) - z2/(z0*z1^2) - z1/(z0^2*z2^2) + 1/(z0*z1*z2))
        """

        rk = type_rank[1]
        if qvar == None:
            qvar = self.qvar
        cart_i = cartan_matrix(type_rank).inverse()
        if basis == "weight":
            wh = [w.transpose() for w in wh]
            nu = [cart_i*nu_i for nu_i in nu]
            b = [cart_i*v for v in b]
            basis = "root"
        rho = weyl_vector(type_rank)
        whr = [w*rho for w in wh]

        ends = [i for i,d in enumerate(S.deg) if d == 1]
        mod_b = copy(b)
        for end,nu_i in zip(ends,nu):
            whr[end] += wh[end]*nu_i
            mod_b[end] += nu_i

        Aw = -1/self.Minv[0,0]*self.A*matrix(whr)
        kappa = self.S_set(type_rank, whr, mod_b, basis=basis)
        if kappa == []:
            return 0
        else:
            kappa = kappa[0]
        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)
        D_r = Delta(rho)
        eps = self.f % 2
        d = self.delta(type_rank,wilson=nu,basis=basis)

        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [d+self.m/(2*self.d)*weyl_lattice_norm(type_rank, v+Aw, basis=basis)]
            prefl = [Delta(v)/D_r ^ (self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)

    def chi_tilde_wilson_mid(self, type_rank, wh, wp, b, expMax, sig, leg, step, basis="weight", qvar=None):
        """
        Compute the chi_tilde q-series with Wilson operator insertion at an intermediate node as in equation (4.81) of [1].

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``wp`` -- Matrix; w' Weyl group element
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``sig`` -- Vector; weight of representation
            -   ``leg`` -- Integer; leg at which to attach the Wilson operator
            -   ``nu`` -- Vector; highest weight of representation
            -   ``step`` -- Integer; step in leg at which to attach the Wilson operator
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- Variable (optional, default = self.qvar); q variable for q series

        EXAMPLES::
            sage: S = Seifert([-1,2,3,-1,2,-1,2]);
            ....: wh,wp = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg], identity_matrix(2)
            ....: b = S.boundary_conditions(["A",2],basis = "weight")[0];
            ....: expMax = 19;
            ....: sig, leg,step = vector([1,1]), 0, 1;
            ....: S.chi_tilde_wilson_mid(["A",2], wh, wp, b, expMax, sig, leg, step, basis = "weight", qvar=None)
            q^(67/64)
        """
        assert not (step == 0 or step ==
                    self.legs[leg]), "No center or endpoint"

        rk = type_rank[1]
        
        if qvar == None:
            qvar = self.qvar

        if basis == "weight":
            wh = [w.transpose() for w in wh]
            wp = wp.transpose()
            cart_i = cartan_matrix(type_rank).inverse()
            sig = cart_i*sig
            b = [cart_i*v for v in b]
            basis = "root"
        
        rho = weyl_vector(type_rank)

        whr = [w*rho for w in wh]
        wil_ind = sum([self.legs[i] for i in range(leg-1)])+step
        whr[wil_ind] += wp*sig

        Aw = -1/self.m*self.A * \
            matrix(whr)-self.Minv[0, wil_ind]/self.Minv[0, 0]*wp*sig

        kappa = self.S_set(type_rank, whr, b, basis=basis)
        if kappa == []:
            return 0
        else:
            kappa = kappa[0]

        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)
        D_r = Delta(rho)

        eps = self.f % 2
        d = self.delta(type_rank) + sum([weyl_lattice_norm(type_rank, wr, wp*sig, basis=basis)*(self.Minv[0, wil_ind]*self.Minv[0, i]/self.Minv[0, 0]-self.Minv[i, wil_ind])
                                  for i, wr in enumerate(whr)])-1/2*weyl_lattice_norm(type_rank, sig, basis=basis)*(self.Minv[0, wil_ind] ^ 2/self.Minv[0, 0]-self.Minv[wil_ind, wil_ind])

        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [d+self.m/(2*self.d)*weyl_lattice_norm(type_rank, v+Aw, basis=basis)]
            prefl = [Delta(v)/D_r ^ (self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)

    def chi_prime_4f_sph(self, wh, expMax, basis=None, qvar=None):
        """
        Compute the hat Z integrand for four fibered sphereical and pseudospherical examples with pecified parameters. Argument descriptions refer to equation (4.53) of [1].

        INPUTS:
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``basis`` -- String (optional, default = weight); basis in which wh is given.
            -   ``qvar`` -- Variable (optional, default = self.qvar); q variable for q series
        EXAMPLES::
            sage: S = Seifert([-2, 1, 2, 2, 3, 2, 5, 3, 7]);
            ....: wh, expMax =  [identity_matrix(1) if d == 1 else matrix(1) for d in S.deg], 100;
            ....: S.chi_prime_4f_sph(wh,expMax, basis = "weight")
        """

        if qvar == None:
            qvar = self.qvar

        if basis == None:
            warnings.warn("No basis is specified, weight is assumed")
            basis = "weight"


        rho = weyl_vector(["A",1])
        mu = sum([a*(w*rho) for a, w in zip(self.A, wh)])

        delta = self.delta(["A",1])

        def fprefexp(n):
            prefl = list()
            expl = list()
            k = n[0]
            prefl.append((z^(2*k*self.d)-2+z^(-2*k*self.d))
                         * (z-z^(-1))^(-2))
            expl.append(delta+(self.d*self.m*k+mu[0]) ^ 2/(self.d*self.m))
            return prefl, expl

        return expand(q_series(fprefexp, expMax-delta, 1, qvar=qvar))

    def z_hat(self,type_rank,b,expMax,nu = None, basis="weight"):
        """
        Compute the z_hat invariant.

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``basis`` -- String; basis in which b is given.

        EXAMPLES::

            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: b = S.boundary_conditions(["A",1],basis = "root")[0];
            ....: expMax = 100;
            ....: S.z_hat(["A",1], b, expMax, basis = "root")
            -q^(197/4) + q^(101/4) - q^(5/4) + q^(1/4)

            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: b = S.boundary_conditions(["A",2],basis = "root")[0];
            ....: expMax = 100;
            ....: S.z_hat(["A",2], b, expMax, basis = "root")
            -2*q^94 - 2*q^92 + 2*q^77 + 2*q^76 + q^65 + 4*q^58 + 2*q^53 - 4*q^50 - 2*q^44 - 2*q^40 + 4*q^32 - 4*q^29 + 2*q^26 - 2*q^22 + q^17 + q^5 + 2*q^4 - 4*q^2 + q
        """
        if basis == None:
            warnings.warn("No basis is specified, weight is assumed")
            basis = "weight"
        
        WG = weyl_group(type_rank)
        if basis == "weight":
            WG = [g.transpose() for g in WG]

        WL = weyl_lengths(type_rank)
        wh_l = list(itertools.product(*[WG if d==1 else [matrix(type_rank[1])] for d in self.deg]))
        wl_l = list(itertools.product(WL,repeat=3))
        Zhat_integrand = 0
        for wh,l in zip(wh_l,wl_l):
            if nu == None:
                chit = product(l)*self.chi_tilde(type_rank, wh, b,  expMax, basis=basis)
                Zhat_integrand += chit 
            else:
                chit = product(l)*self.chi_tilde_wilson_end(type_rank, wh, b,  expMax,nu, basis=basis)
                Zhat_integrand += chit
        Zhat_integrand *= self._zhat_prefactor(type_rank)
        try:
            return const_term(Zhat_integrand)
        except:
            return Zhat_integrand

"""
References:
    [1] Cheng, M.C., Chun, S., Feigin, B., Ferrari, F., Gukov, S., Harrison, S.M. and Passaro, D., 2022. 3-Manifolds and VOA Characters. arXiv preprint arXiv:2201.04640.
"""


