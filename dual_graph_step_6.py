import numpy as np
import maxflow
import meshHelpers
from mesh import Mesh
from gudhi.simplex_tree import SimplexTree
#import networkx as nx
#import matplotlib.pyplot as plt
from collections import deque


class dual_step_6():

    def __init__(self, k_2, z_j, array_of_connected_components, mesh) -> None:

        # gather all the triangles and edges in k_2
        allSimplices = meshHelpers.removeFiltration(k_2.get_simplices())
        (edges_in_k2, triangles_in_k2) = meshHelpers.getEdgesAndTriangles(allSimplices)
        numTriangles = len(triangles_in_k2)


        # Initialize maxflow graph with number of vertices/nodes in dual graph
        # This should be sum of the following three
        #       1) numTriangles in k beta = 'numTriangles'
        #       2) + 1 dummy vertex for φ_bar (the vertex that does not correspond to any of the connected components)
        #       3) + 1 dummy vertex for each of connected component = len(array_of_connected_components)
        #
        # Therefore total number of nodes = (1) + (2) + (3)
        # And total number of non terminal edges ~ 2 times number of triangles in k2

        # We use 'num_vertices' and 'number_non_terminal_edges' to initialize maxflow.Graph object.
        # THESE NUMBERS NEED NOT BE ACCURATE. These numbers are used in object initialization to allow
        # efficient memory management (according to MaxFlow documentation)
        # Therefore, this is simply an approximate estimate. 
        num_vertices = numTriangles + len(array_of_connected_components) + 1
        number_non_terminal_edges = 2 * numTriangles
        self.graph = maxflow.Graph[float](num_vertices, number_non_terminal_edges)


        # Vertices indices are considered in the order (1), (2) and (3)
        # (1) (2) and (3) are described above.
        self.nodes = self.graph.add_nodes(num_vertices)


        # Following is a dict that specifies vertex index against each triangle
        # For each of the triangle, 'vertex_indices[key]' returns value that gives vertex index of 'key'
        self.vertex_indices = {}
        index = 0
        for triangle in triangles_in_k2:
            self.vertex_indices[tuple(triangle)] = index
            index += 1

        # Dummy vertex phi_bar corresponds to (0, 0, 0)
        # Dummy vertex phi_0 corresponds to (0, 0, 1)
        # Dummy vertex phi_1 corresponds to (0, 0, 2) ... and so on for each of the k connected components from step 4
        self.vertex_indices[(0, 0, 0)] = index
        index += 1

        for compindex in range(len(array_of_connected_components)):
            self.vertex_indices[(0, 0, compindex + 1)] = index
            index += 1

        # Define the sources and targets in dual graph
        self.define_sources_and_targets()
        
        # Add edges in the dual graph based on its conditions on its cofaces in k2
        self.add_edges_in_dual_graph(edges_in_k2, k_2, array_of_connected_components, z_j, mesh)

    # End of constructor (init function)

    ############################################################################################        

    def add_edges_in_dual_graph(self, edges_in_k2, k_2, array_of_c_j, zj, mesh):

        #self.edges = np.empty_like(edges_in_k2)
        self.edges = []

        self.non_augmenting_edges = {}
        edges = {}
        for edge in edges_in_k2:

            # get cofaces of codimension 1 (i.e. triangles adjacent to edge)
            adjTriangles = mesh.getCofaces(edge, 1)

            [t_i1, t_i2] = self.get_indices_of_vertices(k_2, adjTriangles, array_of_c_j)
            capacity = mesh.length(edge)

            key = (t_i1, t_i2)
            if key in self.non_augmenting_edges:
                value = self.non_augmenting_edges[key]
                value.append(edge)
            else:
                value = [edge]
            self.non_augmenting_edges[key] = value

            if not (t_i1, t_i2) in edges:
                #self.non_augmenting_edges[(t_i1, t_i2)] = edge
                edges[(t_i1, t_i2)] = capacity
            else:
                edges[(t_i1, t_i2)] += capacity

        #############################################################################################
        # In addition to the above edges, for each φj , add to G an augmenting edge 
        # connecting φj to φ and let its weight be sum of the weights of edges in Zj
        vertex_index_phi_bar = self.vertex_indices[(0, 0, 0)]

        # For each j, there must be a connected component, and corresponding edges in zj
        assert( len(zj) == len(array_of_c_j) )

        for index in range(len(array_of_c_j)):
            vertex_index_component = self.vertex_indices[(0, 0, index + 1)]
            edges[(vertex_index_phi_bar, vertex_index_component)] += self.sum_of_weights(zj[index], mesh)

        #############################################################################################
        # now add edges
        index = 0
        for edge, weight in edges.items():
            self.edges.append(edge)
            index += 1

            ref1 = self.nodes[edge[0]]
            ref2 = self.nodes[edge[1]]
            
            #self.graph.add_edge(self.nodes[t_i1], self.nodes[t_i2], weight, weight)
            self.graph.add_edge(ref1, ref2, weight, weight)

        alledge = edges.keys()
        vals = self.bfs_connected_components(self.nodes, alledge)
        assert(len(vals) == 1)

    ############################################################################################
            
    def sum_of_weights(self, edges_in_Z_j_current_component, mesh):
        total_weight = 0
        for edge in edges_in_Z_j_current_component:
            total_weight += mesh.length(edge)
        return total_weight

    ############################################################################################
    def define_sources_and_targets(self):
        source_index = self.vertex_indices[(0, 0, 1)]   # Get node index of φ 0
        target_index = self.vertex_indices[(0, 0, 0)]   # Get node index of φ bar

        self.addSource(source_index)
        self.addTarget(target_index)

    ############################################################################################

    def get_indices_of_vertices(self, k2, adjTriangles, array_of_c_j):
        # Three cases...

        assert (len(adjTriangles) == 2) # There must be 2 cofaces to each edge
        #if len(adjTriangles) == 2:
        triangle1 = adjTriangles[0]
        triangle2 = adjTriangles[1]
        # The case where BOTH the triangles are in k beta
        if k2.find(triangle1) and k2.find(triangle2):
            i = self.vertex_indices[tuple(triangle1)]
            j = self.vertex_indices[tuple(triangle2)]
            return sorted((i, j))
            

        # The case where one of them is in k beta
        if k2.find(triangle1):
            i = self.vertex_indices[tuple(triangle1)]
            dummy_index = self.find_triangle_in_connected_components(tuple(triangle2), array_of_c_j)
            j = self.vertex_indices[(0, 0, dummy_index)] #connect to appropriate dummy vertex
            return sorted((i, j))

        elif k2.find(triangle2):
            dummy_index = self.find_triangle_in_connected_components(tuple(triangle1), array_of_c_j)
            i = self.vertex_indices[(0, 0, dummy_index)] #connect to appropriate dummy vertex
            j = self.vertex_indices[tuple(triangle2)]
            return sorted((i, j))
        

        # If you are here, neither triangle is in k beta
        # Find if the triangle belongs to one of the components in c_j,
        # if so, connect the dual vertices dual to the triangle1
        # else connect them to phi bar

        dummy_index = self.find_triangle_in_connected_components(tuple(triangle1), array_of_c_j)
        i = self.vertex_indices[(0, 0, dummy_index)]


        dummy_index = self.find_triangle_in_connected_components(tuple(triangle2), array_of_c_j)
        j = self.vertex_indices[(0, 0, dummy_index)]

        return sorted((i, j))

    ############################################################################################
    '''
    Returns value of 'index' + 1 if the 'triangle' is found in 'connected_components'
    where 'index' is the index within the array of components.
    (This corresponds to j in C_j in the document)

    if not found, it returns 0.
    '''
    def find_triangle_in_connected_components(self, triangle, connected_components):
        index = 0
        for component in connected_components:
            if triangle in component:
                return index + 1
            index += 1
        
        return 0    # return 0 if not found

    def output_cross_edges(self):
        self.computeMaxFlow()

        # enable this to test networkx di graph
        # self.draw()

        # After the maxflow is computed, fetch the min cut
        self.cut = self.graph.get_grid_segments(self.nodes)
        edges = self.get_cross_edges()

        # All the cross edges are divided into 2 sets
        # Set 1 contains all the edges Zμ0 , . . . , Zμl ==> These are the edges with sources in φμ0 , . . . , φμl
        # Set 2 contains all the non augmenting edges (edges from cases A to C as described in the document)
        non_augmenting = self.filter_augmenting_edges(edges)

        # Get corresponding triangles in the mesh
        return self.get_edges_to_output(non_augmenting)
    
    def filter_augmenting_edges(self, edges):
        ret = []

        phi_bar = self.vertex_indices[(0, 0, 0)]
        phi_0 = self.vertex_indices[(0, 0, 1)]

        # last vertex corresponds to 
        index_of_last_node = len(self.vertex_indices) - 1
        phi_k = index_of_last_node

        set1 = []   # all edges with dummy vertices except phi bar
        set2 = []   # all non augmenting edges
        for edge in edges:

            if edge in self.non_augmenting_edges:
            #if in_range(source, phi_bar, phi_k):
                #pass # edge contains dummy vertex => augmenting edge
            #else:
                ret.append(edge)   # edge is non augmenting
            else:
                pass
            
        return ret

    def get_cross_edges(self):
        cross_edges = []
        for index in range(len(self.edges)):
            node1 = self.edges[index][0]
            node2 = self.edges[index][1]
            if self.cut[node1] ^ self.cut[node2]:
                # if 1 and 2 are in different sets of the cut => 1 ^ 2 = True
                cross_edges.append((node1, node2))

        return cross_edges
    
    def get_edges_to_output(self, edges):
        ret = []

        for edge in edges:
            if edge in self.non_augmenting_edges:
                original_edges = self.non_augmenting_edges[edge]
                for each_edge in original_edges:
                    ret.append(each_edge)

        return ret

    def computeMaxFlow(self):
        flow = self.graph.maxflow()

        print("\n#######################################\n")
        print(f"Maximum flow in step 6 is : {flow}")

    ############################################################################################
            
    def addSource(self, index):
        self.graph.add_tedge(self.nodes[index], np.inf, 0)

    def addTarget(self, index):
        self.graph.add_tedge(self.nodes[index], 0, np.inf)
        
    ############################################################################################
    '''    
    def draw(self):
        G = self.graph.get_nx_graph()

        pos = nx.spring_layout(G)

        # Draw the directed graph (nodes, edges, and labels)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)

        # Display the graph
        plt.show()
    '''

    # Function to perform BFS and find connected components
    def bfs_connected_components(self, V, E):
        visited = set()  # To track visited nodes
        components = []  # List to store all connected components

        # Create an adjacency list from the set of edges
        adjacency_list = {v: set() for v in V}
        for u, v in E:
            adjacency_list[u].add(v)
            adjacency_list[v].add(u)

        # Iterate through all vertices in the graph
        for node in V:
            if node not in visited:
                # Start a new component and perform BFS
                component = []
                queue = deque([node])  # Queue for BFS

                while queue:
                    current_node = queue.popleft()

                    if current_node not in visited:
                        visited.add(current_node)
                        component.append(current_node)

                        # Add all neighbors of the current node to the queue
                        for neighbor in adjacency_list[current_node]:
                            if neighbor not in visited:
                                queue.append(neighbor)

                components.append(component)  # Store the found component

        return components