from sage.all_cmdline import *   # import sage library
from sage.graphs.graph_plot import GraphPlot

from collections import Counter
from functools import lru_cache, wraps
import numpy as np
import itertools

from cython_l_norm import *
load("utils.sage")
load("weightedgraph.sage")

class Plumbing():
    """
    A class to represent a plumbing manifold. 
    """
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
                                            'vertex_size': 20}
            if self._graph.is_tree():
                self._plot_options['layout'] = 'tree'

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
                ceil_x = ceil(round(x,5))
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
        assert len(set(list(map(len, plumbing_matrix)) + [len(plumbing_matrix)])) == 1, "The plumbing matrix must be a square matrix."

        # Initialize vertices and edges
        vertices = {}
        edges = []

        # Process the plumbing matrix
        for i,row in enumerate(plumbing_matrix):
            vertices[i] = row[i]
            for j,entry in enumerate(row[i+1:]):
                if entry != 0:
                    edges.append((i, i+j+1))
        return cls(vertices, edges)
    
    def invert_orientation(self):
        return Plumbing({ k: -v for k, v in self._vertices_dict.items()}, [(x[1], x[0]) for x in self._edges])

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
            plumbing_matrix = self._graph.adjacency_matrix()
            for i in range(0, self._vertex_count):
                plumbing_matrix[i, i] = self._weight_vector[i,0]
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
            self._is_unimodular = (self.plumbing_matrix.det())**2 == 1 
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
                high_valency_node_count = sum(1 for d in self._degree_vector if d[0] > 2)
                self._is_Seifert =  high_valency_node_count == 1
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
        Return the Seifert data of the plumbing manifold if it is a Seifert manifold.
        """
        if not self.is_Seifert:
            print("The plumbing graph is not a Seifert manifold.")
            return -1
        if self._Seifert_data is None:
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
            raise NotImplementedError("The Orbifold Euler Characteristic is only implemented for Seifert manifolds. This Plumbing manifold is not a Seifert manifold")
        if self._orbifold_euler == None:
            self._orbifold_euler = sum(self.Seifert_data)
        return self._orbifold_euler

    @property
    def norm_fac_euler(self):
        """
        Normalization factor for the Orbifold's Euler characteristic. This is denoted by D in [1].
        """
        if not self.is_Seifert:
            raise NotImplementedError("This function is only implemented for Seifert manifolds. This Plumbing manifold is not a Seifert manifold")
        if self._norm_fac_euler == None:
            p_coefficients = [d.denominator() for d in self.Seifert_data[1:]]
            factors = [(1/(self.orbifold_euler * p)).denominator() for p in p_coefficients]
            self._norm_fac_euler = abs(lcm(factors))
        return self._norm_fac_euler


    @property
    def eff_inv_euler(self):
        """
        Effective inverse Euler characteristic. This is denoted by m in [1].
        """
        if self._eff_inv_euler == None:
            self._eff_inv_euler = -self.plumbing_matrix_inverse[0, 0]*self.norm_fac_euler
        return self._eff_inv_euler
            
    @property
    def coker(self):
        """
        Cokernel of the plumbing matrix.
        """
        if self._coker == None:
            if self.plumbing_matrix_determinant**2 == 1:
                return [[0]*self.vertex_count]
            Coker = []
            for v in range(self.vertex_count):
                vec = vector([0]*self.vertex_count)
                for i in range(abs(self.plumbing_matrix_determinant)):
                    vec[v] = i+1
                    new = [x-floor(x) for x in self.plumbing_matrix_inverse*vec]
                    if new not in Coker:
                        Coker.append(vector(new))
            return Coker
    
        
    def fugacity_variables(self,type_rank):
        """
        Fugacity variables associated to a gauge group. These are z_{i,j} where i is the vertex and j is the fugacity index
        ranging from 0 to rk-1.
        """
        fug_var = []
        for i in range(self._vertex_count):
            fug_var.append(matrix(var(f"z_{i}_{j+1}") for j in range(type_rank[1])).T)
        return fug_var


    def trivial_spin_c(self, type_rank, basis = "weight"):
        """
        Trivial spin_c structure.
        """

        type_rank = tuple(type_rank)
        # Construct b0 
        C = cartan_matrix(type_rank)
        rho = weyl_vector(type_rank)
        b0 = matrix([((a[0] - 2) * rho) for a in self.degree_vector])
        if basis == "weight":
            C = cartan_matrix(type_rank)
            b0 = matrix([C*v_r for v_r in b0.rows()])
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
                spin_c.append([x*ei for x in self.plumbing_matrix*vector(v)])
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
        """
        Displays the plumbing graph.
        """
        self._graph_plot.show()

    def _zhat_prefactor(self,type_rank):
        """
        Prefactor for the Zhat invariant for a give ADE group.
        """
        pos_eigen = sum(np.greater(self.plumbing_matrix.eigenvalues(),0))    
        Phi = len(WeylGroup(type_rank).canonical_representation().positive_roots())
        sig = 2*pos_eigen - self.vertex_count
        norm_rho2 = weyl_lattice_norm(type_rank,weyl_vector(type_rank),basis="root")

        return Series.from_symbolic((-1)^(Phi*pos_eigen)*q^((3*sig-self.plumbing_matrix.trace())/2*norm_rho2))

    def zhat_np(self, type_rank, spin_c, n_powers = 2, basis="weight", L_norm = L_norm_vectorized):
        """
            Compute the Zhat invariant for a plumbing manifold, without cutting off the series.
            
            Warning
            --------
            This algorithm does not ensure convergence of the series. 
            It is recommended to compare the results with other methods to ensure convergence.

        """
        if basis == "root":
            C = cartan_matrix(type_rank)
            spin_c = matrix([C*v_r for v_r in spin_c])

        exponent_products, prefactor_products = self._ell_setup(type_rank, n_powers)
        condition = np.all(np.mod(np.array(self.plumbing_matrix_inverse) @ ((exponent_products - np.array(spin_c))),1) == 0,(1,2))

        exponent_contributing = exponent_products[condition]
        prefactor_contributing = prefactor_products[condition]
        
        q_powers = self._compute_zhat(spin_c,type_rank,exponent_contributing,prefactor_contributing, L_norm=L_norm_vectorized)
        return q_powers * self._zhat_prefactor(type_rank)*self._zhat_prefactor(type_rank)
    
            
    def _compute_zhat(self,spin_c, type_rank, exponent_products, prefactor_products, L_norm = L_norm_vectorized):
        """
        With the exponent and prefactor products assembled, compute the zhat. 
        """
        WG = [g.T for g in weyl_group(type_rank)]
        cartan_i = cartan_matrix(type_rank).inverse()
        matrix_products = (np.array(self.plumbing_matrix_inverse) @ (exponent_products[np.newaxis,:] - ((WG @ (np.array(spin_c).T)[np.newaxis,:]).transpose(0,2,1))[:,np.newaxis,:]) @ cartan_i)
        non_int_part, _ = np.modf(np.round(matrix_products, 6))
        condition = np.concatenate(np.all((np.abs(non_int_part) < 1e-5),axis=(2,3)))
        exponent_contributing = np.tile(exponent_products, (len(WG),1,1))[condition]
        prefactor_contributing = np.tile(prefactor_products, (len(WG),1))[condition]

        # Compute L_norms and prefactors
        dec_approx = len(str(np.max(np.abs(self.plumbing_matrix_inverse))))
        C_inv = np.array(cartan_matrix(type_rank).inverse(),dtype=np.float64)
        L_norms = L_norm(np.array(self.plumbing_matrix_inverse,dtype=np.float64),C_inv,exponent_contributing,dec_approx)
        prefactor_contributing = np.prod(prefactor_contributing,axis=1)
        
        # Convert to higher precision result if necessar
        q_powers = [QQ(round(-1/2*t,12)) for t in L_norms]
        series_numerical = [[tuple(p),c] for p,c in zip(q_powers,prefactor_contributing)]
        return Series(series_numerical,variables=[var("q")])
    
    def _ell_setup(self, type_rank, n_powers):
        """
        Construct the set of exponents and prefactors for the Zhat invariant computation.
        """
        rk = type_rank[1]
        C = cartan_matrix(type_rank) # Uses sage
        rho = C*weyl_vector(type_rank) # Uses sage
        WG = [g.T for g in weyl_group(type_rank)] # Uses sage
        WL = weyl_lengths(type_rank)
        node_contributions_exponents = list()
        node_contributions_prefactors = list()
        # Compute the weyl denominator expansion, if high degree nodes exists 
        if any([x > 2 for x in self.degree_vector.T[0]]):
            weyl_expansion = weyl_double_sided_expansion(type_rank, n_powers) # Does not use sage
        # Compute the node contributions
        for degree in self.degree_vector.T[0]:
            if degree == 0: # If the degree is zero, the contribution is just one
                node_contributions_exponents.append(([0]*rk)) 
                node_contributions_prefactors.append([1])
            elif degree == 1: # If the degree is one we have a leaf
                node_contributions_exponents.append([tuple(g * rho) for g in WG])
                node_contributions_prefactors.append(WL)
            else: # If the degree is greater than one, use (add a degree = 2 case which should be trivial)
                new_powrs = list()
                new_coeffs = list()
                for expansion in weyl_expansion: # Selects for expansion at 0 and oo
                    tot_exp = invert_powers(expansion.pow(degree-2))
                    powrs,coeffs = list(zip(*tot_exp.numerical))
                    new_powrs += powrs
                    new_coeffs += coeffs
                node_contributions_exponents.append(new_powrs)
                node_contributions_prefactors.append(new_coeffs)
        exponent_products = np.array(list(itertools.product(*node_contributions_exponents))).astype(np.float64) # This here is the bottleneck right now
        prefactor_products = np.array(list(itertools.product(*node_contributions_prefactors))) # This here is the bottleneck right now
        return exponent_products, prefactor_products


    def zhat(self, type_rank, spin_c, order = 10, basis="weight", n_powers_start = 1, div_factor=100, method = "cython", info = False):
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
            if info:
                print(f"Computing {n_powers}")
            # Compute the exponents, prefactors and the zhat invariant
            exponent_products, prefactor_products = self._ell_setup(type_rank, n_powers) # This has room for improvement
            zhat_A = self._compute_zhat(spin_c, type_rank, exponent_products, prefactor_products, L_norm=L_norm)*self._zhat_prefactor(type_rank)

            # Assess the order of the computed series, has room for improvement
            exponent_products2, prefactor_products2 = self._ell_setup(type_rank, n_powers + 1)
            # Remove terms that are already computed
            new_terms = np.logical_not(np.all(np.isin(exponent_products2,exponent_products),axis=(1,2)))
            exponent_products2 = exponent_products2[new_terms]
            prefactor_products2 = prefactor_products2[new_terms]
            zhat_B = self._compute_zhat(spin_c, type_rank, exponent_products2, prefactor_products2, L_norm=L_norm)*self._zhat_prefactor(type_rank)
            max_power_computed = zhat_B.min_degree-1
            if info:
                print(f"Maximum power computed {max_power_computed}")
                print(f"zhat_A: {zhat_A}")
                print(f"zhat_B: {zhat_B}")
            n_powers += (int((order - max_power_computed)/div_factor)+1) # This has room for improvement

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
        self._coefficients = None



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
    def coefficients(self):
        if self._coefficients == None:
            self._coefficients = [monomial[1] for monomial in self.numerical]
        return self._coefficients

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

    def collect_like_terms(self):
        """
        Collect like terms in the series.
        """
        result_dict = {}
        for powers, coeff in self.numerical:
            if powers in result_dict:
                powers = round(powers*1.,10)
                result_dict[powers] += coeff
            else:
                result_dict[powers] = coeff
        return Series.from_dictionary(result_dict,self.variables)
    
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
    
"""
References:
    [1] Cheng, M.C., Chun, S., Feigin, B., Ferrari, F., Gukov, S., Harrison, S.M. and Passaro, D., 2022. 3-Manifolds and VOA Characters. arXiv preprint arXiv:2201.04640.
"""


