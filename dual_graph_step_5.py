import numpy as np
import maxflow
import meshHelpers
import fileWriter
from mesh import Mesh
from gudhi.simplex_tree import SimplexTree

class dual_step_5():

    def build_mj_from_cj(self, c_j):
        self.triangle_indices = {}
        self.M_j = SimplexTree()

        index = 0
        for simplex in c_j:
            self.triangle_indices[simplex] = index
            self.M_j.insert(simplex)
            index += 1

    def __init__(self, c_j, mesh, birth, death) -> None:

        self.build_mj_from_cj(c_j)

        allSimplices = meshHelpers.removeFiltration(self.M_j.get_simplices())
        (edges_in_mj, triangles_in_mj) = meshHelpers.getEdgesAndTriangles(allSimplices)
        numTriangles = len(triangles_in_mj)

        self.graph = maxflow.Graph[float](numTriangles, 2 * numTriangles)

        # Number of vertices in dual graph for each triangle = numTriangles
        # + 1 vertex for dummy vertex in the dual graph
        # (numTriangles + 1) vertices in dual graph in total
        # self.nodes[numTriangles] is the last vertex and this corresponds to the dummy vertex
        self.nodes = self.graph.add_nodes(numTriangles + 1)

        # Add edges in the dual graph based on its cofaces in mj
        self.add_edges_in_dual_graph(edges_in_mj, mesh)

        # Based on the triangles in mj and whether or not they contain a critical value
        # select the sources and targets based on i-b being even/odd
        self.addSourcesAndTargets(triangles_in_mj, mesh, birth)

############################################################################################        

    def add_edges_in_dual_graph(self, edges_in_mj, mesh):

        self.edges = np.empty_like(edges_in_mj)
        index = 0

        for edge in edges_in_mj:

            # get cofaces of codimension 1 (i.e. triangles adjacent to edge)
            adjTriangles = meshHelpers.removeFiltration(self.M_j.get_cofaces(edge, 1))

            t1 = np.array(adjTriangles[0])
            t_i1 = self.triangle_indices[tuple(t1)]

            if len(adjTriangles) == 1:
                #t2 = np.array(adjTriangles[1])       # This should be a dummy edge
                t_i2 = len(self.nodes) - 1            # 'len(self.nodes) - 1' is the index of dummy edge
            else:
                t2 = np.array(adjTriangles[1])
                t_i2 = self.triangle_indices[tuple(t2)]

            capacity = mesh.length(edge)
            self.edges[index] = np.array([t_i1, t_i2])
            index += 1
            self.graph.add_edge(self.nodes[t_i1], self.nodes[t_i2], capacity, capacity)
            # two capacities - one for forward edge and another for backward


    def addSourcesAndTargets(self, triangles_in_mj, mesh, birth):
        even_set = []
        odd_set = []

        e = []
        o = []
        for triangle in triangles_in_mj:
            # the following index means this..
            # if index >= 0, it means triangle contains a vertex with critical value alpha i
            # if index < 0, the triangle does not contain any vertex with critical value
            index_critical_vertex = mesh.has_critical_vertex_i(triangle)
            if index_critical_vertex >= 0:
                assert (index_critical_vertex - birth) >= 0
                if (index_critical_vertex - birth) % 2 == 0:
                    even_set.append(self.triangle_indices[tuple(triangle)])
                    e.append(triangle)
                else:
                    odd_set.append(self.triangle_indices[tuple(triangle)])
                    o.append(triangle)
        
        if False:
            self.writetotree(e, mesh)
            self.writeoddtotree(o, mesh)

        # Set of sources = all triangles Ui with (i minus b) as even + Dummy vertex psi _ j
        for i in even_set:
            self.addSource(i)

        self.addSource(len(self.nodes) - 1) # add dummy vertex to the list of sources

        # Set of targets = all triangles Ui with (i minus b) as odd
        for i in odd_set:
            self.addTarget(i)

    def addSource(self, index):
        self.graph.add_tedge(self.nodes[index], np.inf, 0)


    def addTarget(self, index):
        self.graph.add_tedge(self.nodes[index], 0, np.inf)

    def output_cross_edges(self):
        flow = self.graph.maxflow()

        print("\n#######################################\n")
        print(f"Maximum flow in step 5 is : {flow}")

        # After the maxflow is computed, fetch the min cut
        self.cut = self.graph.get_grid_segments(self.nodes)
        E_j = self.get_cross_edges_from_dual_graph()    # These are the edges crossing the cut
        Z_j = self.get_edges_in_mj(E_j)
        return Z_j

    def get_edges_in_mj(self, E_j):
        Z_j = []
        indices_and_triangles = {value: key for key, value in self.triangle_indices.items()}

        for edge in E_j:
            dual_vertex_1 = edge[0]
            dual_vertex_2 = edge[1]

            triangle1 = indices_and_triangles[dual_vertex_1]
            triangle2 = indices_and_triangles[dual_vertex_2]
            common_edge = self.find_common_edge(triangle1, triangle2)

            Z_j.append(common_edge)
        return Z_j
    
    def find_common_edge(self, triangle1, triangle2):
        # Convert tuples to sets to find the intersection
        set1 = set(triangle1)
        set2 = set(triangle2)

        # Find the common vertices
        common_vertices = list(set1 & set2)
        common_vertices = sorted(common_vertices)

        # Check if there are exactly two common vertices (which form the common edge)
        if len(common_vertices) == 2:
            return tuple(common_vertices)
        else:
            return None  # No common edge or something went wrong
    
    def get_cross_edges_from_dual_graph(self):
        cross_edges = []
        for edge in self.edges:
            node1 = edge[0]
            node2 = edge[1]
            if self.cut[node1] ^ self.cut[node2]:
                # if 1 and 2 are in different sets of the cut => 1 ^ 2 = True
                cross_edges.append((node1, node2))

        return cross_edges

    def computeMaxFlow(self):
        flow = self.graph.maxflow()

        print("\n#######################################\n")
        print(f"Maximum flow is : {flow}")

    def outputCrossEdges(self):
        # After the maxflow is computed, fetch the min cut
        self.cut = self.graph.get_grid_segments(self.nodes)
        edges = self.getCrossEdges()
        return edges
    
    def writetotree(self, tri, mesh):
        new_simplextree = SimplexTree()
        for item in tri:
            new_simplextree.insert(np.array(item), 1.0)

        # Write the edges in Zj to ply
        fileWriter.write_simplex_tree_to_ply(f"evenset.ply", new_simplextree, mesh.vertices)

    def writeoddtotree(self, tri, mesh):
        new_simplextree = SimplexTree()
        for item in tri:
            new_simplextree.insert(np.array(item), 1.0)

        # Write the edges in Zj to ply
        fileWriter.write_simplex_tree_to_ply(f"oddset.ply", new_simplextree, mesh.vertices)