from sage.all import *
from sage.all_cmdline import *  # import sage library
from sage.graphs.graph_plot import GraphPlot

from collections import Counter
import numpy as np
import itertools

from .cython_l_norm import L_norm_cython
from .utils import *
from .weightedgraph import *
from .series import *

def cokernel_reps(M):
    """
    Return up to `limit` coset representatives for coker(M)=Z^n/Im(M),
    assuming it is finite.  Raises ValueError if there's a free part.
    """
    SP = (ZZ**(M.nrows())/(M * ZZ**(M.nrows())))
    return [i.lift() for i in SP]

class Plumbing:
    """
    A class to represent a plumbing manifold.
    """

    def __init__(self, vertices_dict, edges, edge_signs=None):
        try:
            self._vertices_dict = vertices_dict
            self._edges = edges

            self._vertex_count = len(vertices_dict)
            self._edge_count = len(edges)

            self._graph = Graph(multiedges=True)
            self._vertex_list = [
                "$v_{" + str(i) + "}\\hspace{2} " + str(vertices_dict[i]) + "$"
                for i in range(0, self._vertex_count)
            ]

            self._edge_list = [
                (self._vertex_list[x[0]], self._vertex_list[x[1]]) for x in edges
            ]

            self._graph.add_vertices(self._vertex_list)
            self._graph.add_edges(self._edge_list)
            self._plot_options = {"vertex_color": "black", "vertex_size": 20}
            if self._graph.is_tree():
                self._plot_options["layout"] = "tree"

            if edge_signs == None:
                edge_signs = [0] * self._edge_count
            else:
                assert (
                    len(edge_signs) == self._edge_count
                ), "The number of edge signs must be equal to the number of edges."

            self._edge_signs = edge_signs

            self._graph_plot = GraphPlot(self._graph, self._plot_options)

            self._weight_vector = Matrix(list(vertices_dict.values())).T
            self._degree_vector = [self._graph.degree(x) for x in self._vertex_list]
            self._degree_vector = Matrix(self._degree_vector).T

            self._plumbing_matrix = None

            self._plumbing_matrix_inverse = None
            self._plumbing_matrix_determinant = None
            self._lattice_dilation_factor = None
            self._orbifold_euler = None
            self._norm_fac_euler = None
            self._eff_inv_euler = None
            self._coker = None

            self._is_plumbing_non_singular = None
            self._is_unimodular = None
            self._is_tree = None
            self._definiteness_type = None
            self._weak_definiteness_type = None
            self._is_Seifert = None
            self._Seifert_fiber_count = None
            self._Seifert_data = None

        except Exception as e:
            print(
                f"Error: Plumbing entered incorrectly. Please check input. Details {e}"
            )

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
                ceil_x = ceil(round(x, 5))
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

    @classmethod
    def from_plumbing_matrix(cls, plumbing_matrix):
        """
        Create a plumbing manifold from a plumbing matrix.
        """
        # Ensure the plumbing matrix has the shape of a square matrix
        assert (
            len(set(list(map(len, plumbing_matrix)) + [len(plumbing_matrix)])) == 1
        ), "The plumbing matrix must be a square matrix."

        # Initialize vertices and edges
        vertices = {}
        edges = []

        # Process the plumbing matrix
        for i, row in enumerate(plumbing_matrix):
            vertices[i] = row[i]
            for j, entry in enumerate(row[i + 1 :]):
                if entry != 0:
                    edges.append((i, i + j + 1))
        return cls(vertices, edges)

    @classmethod
    def from_Brieskorn(cls, p_list):
        """
        Given a list of multiplicities p_list = [p1, p2, ..., p_r], compute the Seifert data
        associated to this Brieskorn sphere and return the corresponding plumbing manifold.
        """
        assert all(gcd(pi,pj)==1 for pi,pj in itertools.combinations(p_list,r=2)), f"Integers need to be relatively coprime for Brieskorn spere definition, got {p_list}"

        # 1) Compute the product P of all p_i
        P = product(p_list) 

        # 2) For each p_i, define P_i = P / p_i
        #    and compute q_i such that q_i * P_i = 1 (mod p_i).
        #    That is, q_i = P_i_inverse (mod p_i).
        q_list = []
        for p in p_list:
            P_i = P // p
            # Find inverse of P_i modulo p
            q_i = -inverse_mod(P_i, p)  # solves P_i * inv_Pi_mod_p == -1 (mod p)
            q_list.append(q_i)
        # 3) Solve for b using Pb - sum_i q_i/p_i = 1
        #    Note that b is an integer, so we force it to be stored as such
        b = int(1/P - sum(q_i/p_i for q_i,p_i in zip(q_list,p_list)))
        seif_data = (b, *[p / q for q, p in zip(p_list, q_list)])
        S = cls.from_Seifert_data(seif_data)
        return S if S.plumbing_matrix_inverse[0,0]<0 else S.invert_orientation()

    def invert_orientation(self):
        return Plumbing(
            {k: -v for k, v in self._vertices_dict.items()},
            [(x[1], x[0]) for x in self._edges],
        )

    @property
    def vertex_count(self):
        """
        The number of vertices in the plumbing.
        """
        return self._vertex_count

    @property
    def edge_count(self):
        """
        The number of edges in the plumbing
        """
        return self._edge_count

    @property
    def weight_vector(self):
        """
        A Sage matrix of the form [[m_1],...,[m_s]] where m_i is the
        weight of vertex i and s is the number of vertices of the plumbing.
        """
        return self._weight_vector

    @property
    def degree_vector(self):
        """
        A Sage matrix of the form [[d_1],...,[d_s]] where d_i is the
        degree (or valence) of vertex i and s is the number of vertices of the
        plumbing.
        """
        return self._degree_vector

    @property
    def plumbing_matrix(self):
        """
        The plumbing matrix of the plumbing manifold.
        """
        if self._plumbing_matrix is None:
            plumbing_matrix = matrix(self.vertex_count, self.vertex_count)
            for i in range(self.vertex_count):
                plumbing_matrix[i, i] = self._weight_vector[i, 0]
            for edge, sign in zip(self._edges, self._edge_signs):
                i, j = edge
                if i != j:
                    plumbing_matrix[edge[0], edge[1]] += (-1) ** (sign)
                    plumbing_matrix[edge[1], edge[0]] += (-1) ** (sign)
                else:
                    plumbing_matrix[edge[0], edge[1]] += 2 * (-1) ** (sign)
            self._plumbing_matrix = plumbing_matrix
        return self._plumbing_matrix

    @property
    def is_plumbing_non_singular(self):
        """
        True if the plumbing matrix is non-singular, False otherwise.
        """
        if self._is_plumbing_non_singular is None:
            self._is_plumbing_non_singular = self.plumbing_matrix.det() != 0
        return self._is_plumbing_non_singular

    @property
    def is_unimodular(self):
        """
        True if the plumbing matrix is unimodular, False otherwise.
        """
        if self._is_unimodular is None:
            self._is_unimodular = (self.plumbing_matrix.det()) ** 2 == 1
        return self._is_unimodular

    @property
    def definiteness_type(self, threshold=1e-8):
        """
        The definiteness type of the plumbing matrix of the plumbing manifold.

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
            if all(i < -1 * threshold for i in eigenvalues):
                self._definiteness_type = "negative definite"
            elif all(i > threshold for i in eigenvalues):
                self._definiteness_type = "positive definite"
            elif all(-1 * threshold <= i <= threshold for i in eigenvalues):
                self._definiteness_type = "zero matrix"
            elif all(i <= threshold for i in eigenvalues):
                self._definiteness_type = "negative semi-definite"
            elif all(i >= -1 * threshold for i in eigenvalues):
                self._definiteness_type = "positive semi-definite"
            else:
                return "positive and negative eigenvalues"
        return self._definiteness_type

    @property
    def weak_definiteness_type(self, threshold=1e-8):
        """
        The definiteness type of the plumbing matrix minor corresponding to high valency nodes.

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
            if all(i < -1 * threshold for i in eigenvalues):
                self._weak_definiteness_type = "negative weakly definite"
            elif all(i > threshold for i in eigenvalues):
                self._weak_definiteness_type = "positive weakly definite"
            elif all(-1 * threshold <= i <= threshold for i in eigenvalues):
                self._weak_definiteness_type = "zero matrix"
            elif all(i <= threshold for i in eigenvalues):
                self._weak_definiteness_type = "negative weakly semi-definite"
            elif all(i >= -1 * threshold for i in eigenvalues):
                self._weak_definiteness_type = "positive weakly semi-definite"
            else:
                return "positive and negative eigenvalues"
        return self._weak_definiteness_type

    @property
    def is_tree(self):
        """
        True if the plumbing graph is a tree, False otherwise.
        """
        if self._is_tree is None:
            self._is_tree = self._graph.is_tree()
        return self._is_tree

    @property
    def is_Seifert(self):
        """
        True if the manifold is a Seifert manifold, False otherwise.
        """
        if self._is_Seifert is None:
            if self.is_tree == False:
                return False
            else:
                high_valency_node_count = sum(
                    1 for d in self._degree_vector if d[0] > 2
                )
                self._is_Seifert = high_valency_node_count == 1
        return self._is_Seifert

    @property
    def Seifert_fiber_count(self):
        """
        Number of exceptional fibers if Plumbing is Seifert.
        """
        if self._Seifert_fiber_count is None:
            if not self.is_Seifert:
                print("The plumbing graph is not a Seifert manifold.")
                self._Seifert_fiber_count
            else:
                self._Seifert_fiber_count = max(d[0] for d in self._degree_vector)
        return self._Seifert_fiber_count

    @property
    def graph(self):
        """
        The graph representation of the plumbing manifold.
        """
        return self._graph

    @property
    def Seifert_data(self):
        """
        return the seifert data of the plumbing manifold if it is a seifert manifold.
        """
        if not self.is_Seifert:
            print("the plumbing graph is not a seifert manifold.")
            return -1
        if self._Seifert_data is None:
            # flatten the list of edges
            edges_flat = [vertex for edge in self._edges for vertex in edge]
            # count the occurrences of each vertex
            vertex_counts = Counter(edges_flat)
            # find the vertex with the highest valency
            high_valency_vertex = max(vertex_counts, key=vertex_counts.get)

            # 2. Identify legs
            # initialize the list to store legs
            legs = []

            # function to find the next vertex in the leg
            def find_next_vertex(current_vertex, edges, visited):
                for edge in edges:
                    if current_vertex in edge:
                        next_vertex = [v for v in edge if v != current_vertex][0]
                        if next_vertex not in visited:
                            return next_vertex
                return None

            # find legs connected to the high valency vertex
            for edge in self._edges:
                if high_valency_vertex in edge:
                    leg = []
                    last_vertex = [v for v in edge if v != high_valency_vertex][0]
                    leg.append(last_vertex)
                    visited = set(leg).union({high_valency_vertex})

                    while vertex_counts[last_vertex] > 1:
                        next_vertex = find_next_vertex(
                            last_vertex, self._edges, visited
                        )
                        if next_vertex:
                            leg.append(next_vertex)
                            visited.add(next_vertex)
                            last_vertex = next_vertex
                        else:
                            break
                    legs.append(leg)

            # calculate weights for each leg
            legs_weights = [[self._vertices_dict[v] for v in leg] for leg in legs]

            # calculate the seifert data
            seif_data = list()
            seif_data.append(self._vertices_dict[high_valency_vertex])
            for leg in legs_weights:
                leg = list(reversed(leg))
                seif_coeff = leg[0]
                for ai in leg[1:]:
                    seif_coeff = ai - 1 / seif_coeff
                seif_coeff = -1 / seif_coeff
                seif_data.append(seif_coeff)
            self._Seifert_data = seif_data
        return self._Seifert_data

    @property
    def plumbing_matrix_inverse(self):
        """
        Inverse of the plumbing matrix.
        """
        if self._plumbing_matrix_inverse == None:
            self._plumbing_matrix_inverse = self.plumbing_matrix.inverse()
        return self._plumbing_matrix_inverse

    @property
    def plumbing_matrix_determinant(self):
        """
        Determinant of the plumbing matrix
        """
        if self._plumbing_matrix_determinant == None:
            self._plumbing_matrix_determinant = self.plumbing_matrix.det()
        return self._plumbing_matrix_determinant

    @property
    def orbifold_euler(self):
        """
        Orbifold Euler characteristic.
        """
        if not self.is_Seifert:
            raise NotImplementedError(
                "The Orbifold Euler Characteristic is only implemented for Seifert manifolds. This Plumbing manifold is not a Seifert manifold"
            )
        if self._orbifold_euler == None:
            self._orbifold_euler = sum(self.Seifert_data)
        return self._orbifold_euler

    @property
    def norm_fac_euler(self):
        """
        Normalization factor for the Orbifold's Euler characteristic. This is denoted by D in [1].
        """
        if not self.is_Seifert:
            raise NotImplementedError(
                "This function is only implemented for Seifert manifolds. This Plumbing manifold is not a Seifert manifold"
            )
        if self._norm_fac_euler == None:
            p_coefficients = [d.denominator() for d in self.Seifert_data[1:]]
            factors = [
                (1 / (self.orbifold_euler * p)).denominator() for p in p_coefficients
            ]
            self._norm_fac_euler = abs(lcm(factors))
        return self._norm_fac_euler

    @property
    def eff_inv_euler(self):
        """
        Effective inverse Euler characteristic. This is denoted by m in [1].
        """
        if self._eff_inv_euler == None:
            self._eff_inv_euler = (
                -self.plumbing_matrix_inverse[0, 0] * self.norm_fac_euler
            )
        return self._eff_inv_euler

    @property
    def coker(self):
        """
        Cokernel of the plumbing matrix.
        """
        if self._coker == None:
            if self.plumbing_matrix_determinant**2 == 1:
                return [[0] * self.vertex_count]
            Coker = cokernel_reps(self.plumbing_matrix)
            """
            for v in range(self.vertex_count):
                vec = vector([0] * self.vertex_count)
                for i in range(abs(self.plumbing_matrix_determinant)):
                    vec[v] = i + 1
                    new = [x - floor(x) for x in self.plumbing_matrix_inverse * vec]
                    if new not in Coker:
                        Coker.append(vector(new))
            """
            return Coker

    def fugacity_variables(self, type_rank):
        """
        Fugacity variables associated to a gauge group. These are z_{i,j} where i is the vertex and j is the fugacity index
        ranging from 0 to rk-1.
        """
        fug_var = []
        for i in range(self._vertex_count):
            fug_var.append(matrix(var(f"z_{i}_{j+1}") for j in range(type_rank[1])).T)
        return fug_var

    def trivial_spin_c(self, type_rank, basis="weight"):
        """
        Trivial spin_c structure.
        """

        type_rank = tuple(type_rank)
        # Construct b0
        C = cartan_matrix(type_rank)
        rho = weyl_vector(type_rank)
        b0 = matrix([((a[0] - 2) * rho) for a in self.degree_vector])
        if basis == "weight":
            b0 = matrix([C * v_r for v_r in b0.rows()])
        return b0

    def spin_c(self, type_rank, basis="weight"):
        """
        List of inequivalent spin_c structures
        """
        type_rank = tuple(type_rank)
        # Construct b0
        C = cartan_matrix(type_rank)
        rho = weyl_vector(type_rank)
        b0 = matrix([((a[0] - 2) * rho) for a in self.degree_vector])

        if self.plumbing_matrix_determinant**2 == 1:
            spin_c = [b0]
        else:
            # Tensor coker with Lambda
            rk = type_rank[1]
            e = identity_matrix(rk)
            spin_c = list()
            for ei, v in itertools.product(e, self.coker):
                spin_c.append([x * ei for x in  vector(v)])

            # Identify duplicates based on Weyl group transformations
            toRemove = set()
            for i, b in enumerate(spin_c):
                for g in weyl_group(type_rank):
                    gb = [g * vector(x) for x in b]
                    remove = False
                    for bb in spin_c[:i]:
                        if vector(
                            self.plumbing_matrix_inverse * (matrix(gb) - matrix(bb))
                        ) in ZZ ^ (self.vertex_count * rk):
                            remove = True
                            toRemove.add(i)
                            break
                    if remove:
                        break
            
            toRemove = []
            # Remove duplicates
            for i in reversed(sorted(toRemove)):
                del spin_c[i]
            spin_c = sorted([(matrix(b) + matrix(b0)) for b in spin_c])
        if basis == "weight":
            C = cartan_matrix(type_rank)
            spin_c = [matrix([C * v_r for v_r in v.rows()]) for v in spin_c]

        return spin_c

    def display(self):
        """
        Displays the plumbing graph.
        """
        self._graph_plot.show()

    def _zhat_prefactor(self, type_rank):
        """
        Prefactor for the Zhat invariant for a give ADE group.
        """
        q = var("q")
        pos_eigen = sum(np.greater(self.plumbing_matrix.eigenvalues(), 0))
        Phi = len(WeylGroup(type_rank).canonical_representation().positive_roots())  # type: ignore
        sig = 2 * pos_eigen - self.vertex_count
        norm_rho2 = weyl_lattice_norm(type_rank, weyl_vector(type_rank), basis="root")

        return Series.from_symbolic(
            (-1)
            ^ (Phi * pos_eigen) * q
            ^ ((3 * sig - self.plumbing_matrix.trace()) / 2 * norm_rho2)
        )

    def zhat_unbounded(
        self,
        type_rank,
        spin_c,
        n_powers=2,
        wilson=None,
        basis="weight",
        method="cython",
    ):
        """
        Compute the Zhat invariant for a plumbing manifold, without cutting off the series.

        Warning
        --------
        This algorithm does not ensure convergence of the series.
        It is recommended to compare the results with other methods to ensure convergence.

        Parameters
        ----------
        type_rank : tuple
            A tuple representing the type and rank of the Lie algebra.
        spin_c : list
            A list representing the Spin^c structure in the chosen basis.
        n_powers : int, optional
            The number of powers in the double sided Weyl expansion to consider in the series (default is 2).
        wilson : list of vectors, optional
            List of wilson line insertions. This should be of length equal to the number of nodes, each item should be a vector of same dimension as the rank of the group.
        basis : str, optional
            The basis in which the Spin^c structure is given. Can be "weight" or "root" (default is "weight").
        method : str, optional
            The method to use for computing the L-norm. Can be "vectorized", "python", or "cython" (default is "cython").
        Returns
        -------
        Series
            The computed Zhat invariant for the given plumbing manifold.
        -------
        """

        # We work in the weight basis, so we need to convert the spin_c to the weight basis
        if basis == "root":
            C = cartan_matrix(type_rank)
            spin_c = matrix([C * v_r for v_r in spin_c])

        # Select the method to compute the L_norm
        match method:
            case "vectorized":
                L_norm = L_norm_vectorized
            case "python":
                L_norm = L_norm_python
            case "cython":
                L_norm = L_norm_cython
            case _:
                raise ValueError(
                    "Method must be one of 'vectorized', 'python' or 'cython'"
                )

        if wilson is None:
            wilson = [vector([0] * type_rank[1]) for _ in range(self.vertex_count)]

        # Compute the exponents and prefactors
        exponent_products, prefactor_products = self._ell_setup(
            type_rank, n_powers, wilson=wilson
        )
        # Compute the L_norms and prefactors
        q_powers = self._compute_zhat(
            spin_c,
            type_rank,
            exponent_products,
            prefactor_products,
            wilson=wilson,
            L_norm=L_norm,
        )
        return q_powers * self._zhat_prefactor(type_rank)

    def zhat(
        self,
        type_rank,
        spin_c,
        order=10,
        wilson=None,
        basis="weight",
        n_powers_start=1,
        div_factor=100,
        method="cython",
        info=False,
    ):
        """
        Compute the zhat invariant for a given type rank and spin configuration.

        Parameters:
        -----------
        type_rank : tuple
            The type and rank of the Lie algebra (e.g., ('A', 2) for A2).
        spin_c : list
            The spin configuration vector.
        order : int, optional
            The order up to which the series is computed (default is 10).
        wilson : list of vectors, optional
            List of wilson line insertions. This should be of length equal to the number of nodes, each item should be a vector of same dimension as the rank of the group.
        basis : str, optional
            The basis to use, either "weight" or "root" (default is "weight").
        n_powers_start : int, optional
            The starting number of powers for the double sided Weyl expansion in the computation (default is 1).
        div_factor : int, optional
            The division factor to adjust the increment of powers (default is 100).
        method : str, optional
            The method to use for computation, one of "vectorized", "python", or "cython" (default is "cython").
        info : bool, optional
            If True, prints additional information during computation (default is False).

        Returns:
        --------
        Series
            The truncated zhat invariant series up to the computed maximum power.
        """

        if basis == "root":
            C = cartan_matrix(type_rank)
            spin_c = matrix([C * v_r for v_r in spin_c])

        match method:
            case "vectorized":
                L_norm = L_norm_vectorized
            case "python":
                L_norm = L_norm_python
            case "cython":
                L_norm = L_norm_cython
            case _:
                raise ValueError(
                    "Method must be one of 'vectorized', 'python' or 'cython'"
                )

        if wilson is None:
            wilson = [vector([0] * type_rank[1]) for _ in range(self.vertex_count)]
        n_powers = n_powers_start
        zhat_A = Series([])
        zhat_B = Series([])
        max_power_computed = 0
        while max_power_computed <= order:
            if info:
                print(f"Computing {n_powers}")
            # Compute the exponents, prefactors and the zhat invariant
            exponent_products, prefactor_products = self._ell_setup(
                type_rank, n_powers, wilson=wilson
            ) 
            zhat_A = self._compute_zhat(
                spin_c,
                type_rank,
                exponent_products,
                prefactor_products,
                wilson=wilson,
                L_norm=L_norm,
            ) * self._zhat_prefactor(type_rank)

            # Assess the order of the computed series, has room for improvement
            exponent_products2, prefactor_products2 = self._ell_setup(
                type_rank, n_powers + 1, wilson=wilson
            )
            # Remove terms that are already computed
            new_terms = np.logical_not(
                np.all(np.isin(exponent_products2, exponent_products), axis=(1, 2))
            )
            exponent_products2 = exponent_products2[new_terms]
            prefactor_products2 = prefactor_products2[new_terms]
            zhat_B = self._compute_zhat(
                spin_c,
                type_rank,
                exponent_products2,
                prefactor_products2,
                wilson=wilson,
                L_norm=L_norm,
            ) * self._zhat_prefactor(type_rank)
            max_power_computed = zhat_B.min_degree - 1
            if info:
                print(f"Maximum power computed {max_power_computed}")
                print(f"zhat_A: {zhat_A}")
                print(f"zhat_B: {zhat_B}")
            n_powers += (
                int((order - max_power_computed) / div_factor) + 1
            )  # This has room for improvement

        return zhat_A.truncate(max_power_computed)

    def _ell_setup(self, type_rank, n_powers, wilson=None):
        """
        Construct the set of exponents and prefactors for the Zhat invariant computation.
        """
        rk = type_rank[1]
        C = cartan_matrix(type_rank)  # Uses sage
        rho = C * weyl_vector(type_rank)  # Uses sage
        WG = [g.T for g in weyl_group(type_rank)]  # Uses sage
        WL = weyl_lengths(type_rank)
        if wilson is None:
            wilson = [vector([0] * type_rank[1]) for _ in range(self.vertex_count)]

        node_contributions_exponents = list()
        node_contributions_prefactors = list()
        # Compute the weyl denominator expansion, if high degree nodes exists
        # if any([x > 2 for x in self.degree_vector.T[0]]):
        weyl_expansion = weyl_double_sided_expansion(
            type_rank, n_powers
        )  # Does not use sage
        # Compute the node contributions
        for degree, wil_rep in zip(self.degree_vector.T[0], wilson):
            if degree == 0:  # If the degree is zero, the contribution is just one
                node_contributions_exponents.append(
                    [
                        tuple(-g1 * rho - g2 * (rho + wil_rep))
                        for g1, g2 in itertools.product(WG, repeat=2)
                    ]
                )
                node_contributions_prefactors.append(
                    [l1 * l2 for l1, l2 in itertools.product(WL, repeat=2)]
                )
            elif degree == 1:  # If the degree is one we have a leaf
                node_contributions_exponents.append(
                    [tuple(-g * (rho + wil_rep)) for g in WG]
                )
                node_contributions_prefactors.append(WL)
            else:  # If the degree is greater than one
                new_powrs = list()
                new_coeffs = list()
                if wil_rep == vector([0] * rk):
                    for (
                        expansion
                    ) in weyl_expansion:  # Selects for expansion at 0 and oo
                        tot_exp = invert_powers(expansion.pow(degree - 2))
                        powrs, coeffs = list(zip(*tot_exp.numerical))
                        new_powrs += powrs
                        new_coeffs += coeffs
                    node_contributions_exponents.append(new_powrs)
                    node_contributions_prefactors.append(new_coeffs)
                else:
                    for (
                        expansion
                    ) in weyl_expansion:  # Selects for expansion at 0 and oo
                        tot_exp = invert_powers(expansion.pow(degree - 2))
                        powrs, coeffs = list(zip(*tot_exp.numerical))
                        for powr, coeff in zip(powrs, coeffs):
                            for l, g in zip(WL, WG):
                                new_powrs.append(
                                    tuple(-g * (rho + wil_rep) + vector(powr))
                                )
                                new_coeffs.append(coeff * l)
                    node_contributions_exponents.append(new_powrs)
                    node_contributions_prefactors.append(new_coeffs)

        # Create the cartesian product for the exponents and prefactors
        exponent_products = np.array(
            list(itertools.product(*node_contributions_exponents)), dtype=np.float64
        )
        prefactor_products = np.array(
            list(itertools.product(*node_contributions_prefactors))
        )
        return exponent_products, prefactor_products

    def _compute_zhat(
        self,
        spin_c,
        type_rank,
        exponent_products,
        prefactor_products,
        wilson=None,
        L_norm=L_norm_vectorized,
    ):
        """
        With the exponent and prefactor products assembled, compute the zhat.
        """
        WG = [g.T for g in weyl_group(type_rank)]
        cartan_i = cartan_matrix(type_rank).inverse()
        if wilson is None:
            wilson = [vector([0] * type_rank[1]) for _ in range(self.vertex_count)]

        matrix_products = (
            np.array(self.plumbing_matrix_inverse)
            @ (
                exponent_products[np.newaxis, :]
                - (
                    (
                        WG @ (np.array(spin_c).T - np.array(wilson).T)[np.newaxis, :]
                    ).transpose(0, 2, 1)
                )[:, np.newaxis, :]
            )
            @ cartan_i
        )
        non_int_part, _ = np.modf(np.round(matrix_products, 9))
        condition = np.concatenate(np.all((np.abs(non_int_part) < 1e-8), axis=(2, 3)))
        exponent_contributing = np.tile(exponent_products, (len(WG), 1, 1))[condition]
        prefactor_contributing = np.tile(prefactor_products, (len(WG), 1))[condition]

        # Compute L_norms and prefactors
        dec_approx = len(str(np.max(np.abs(self.plumbing_matrix_inverse))))
        C_inv = np.array(cartan_matrix(type_rank).inverse(), dtype=np.float64)
        L_norms = L_norm(
            np.array(self.plumbing_matrix_inverse, dtype=np.float64),
            C_inv,
            exponent_contributing.astype(np.float64),
            dec_approx,
        )
        prefactor_contributing = np.prod(prefactor_contributing, axis=1)

        # Convert to higher precision result if necessar
        # q_powers = [QQ(round(-1 / 2 * t,12)) for t in L_norms]
        q_powers = [QQ(-1 / 2 * t) for t in L_norms]
        series_numerical = [
            [tuple(p), 1 / (2 * len(WG)) * c]
            for p, c in zip(q_powers, prefactor_contributing)
        ]
        return Series(series_numerical, variables=[var("q")])


"""
References:
    [1] Cheng, M.C., Chun, S., Feigin, B., Ferrari, F., Gukov, S., Harrison, S.M. and Passaro, D., 2022. 3-Manifolds and VOA Characters. arXiv preprint arXiv:2201.04640.
"""
