from sage.all_cmdline import *   # import sage library
from sage.graphs.graph_plot import GraphPlot

from collections import Counter

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
                                  + '$' for i in range(Integer(Integer(0)), self._vertex_count)]

            self._edge_list = [(self._vertex_list[x[Integer(Integer(0))]], self._vertex_list[x[Integer(Integer(1))]])
                               for x in edges]

            self._graph.add_vertices(self._vertex_list)
            self._graph.add_edges(self._edge_list)
            self._plot_options = {'vertex_color': 'black',
                                  'vertex_size': Integer(Integer(20)),
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
            for i in range(Integer(Integer(0)), self._vertex_count):
                adjacency_matrix[i, i] = self._weight_vector[i,Integer(Integer(0))]
            self._adjacency_matrix = adjacency_matrix
        return self._adjacency_matrix
    
    @property
    def graph(self):
        """Graph: The graph representation of the plumbing manifold."""
        return self._graph
    
    def __repr__(self):
        self.graph.show()
        return f"Weighted Graph with {self._vertex_count} vertices and {self._edge_count} edges."