import numpy as np
import maxflow

class dual():
    def __init__(self, mesh) -> None:
        numTriangles = len(mesh.triangles)
        self.graph = maxflow.Graph[float](numTriangles, 2 * numTriangles)
        self.nodes = self.graph.add_nodes(numTriangles)
        self.addEdges(mesh)
        self.addSourcesAndSinks(mesh)

    def computeMaxFlow(self):
        flow = self.graph.maxflow()

        print("\n#######################################\n")
        print(f"Maximum flow is : {flow}")

    def outputCrossEdges(self):
        # After the maxflow is computed, fetch the min cut
        self.cut = self.graph.get_grid_segments(self.nodes)
        edges = self.getCrossEdges()
        return edges

    def addEdges(self, mesh):

        self.edges = np.empty_like(mesh.edges)
        index = 0

        for edge in mesh.edges:
            adjTriangles = mesh.getTrianglesAdjacentToEdge(edge)

            t1 = np.array(adjTriangles[0])
            t2 = np.array(adjTriangles[1])
            t_i1 = mesh.getTriangleIndex(t1)
            t_i2 = mesh.getTriangleIndex(t2)

            capacity = mesh.length(edge)
            self.edges[index] = np.array([t_i1, t_i2])
            index += 1
            self.graph.add_edge(self.nodes[t_i1], self.nodes[t_i2], capacity, capacity)
            # two capacities - one for forward edge and another for backward


    def addSourcesAndSinks(self, mesh):
        for i in mesh.even_box:
            self.addSource(i)

        for i in mesh.odd_box:
            self.addSink(i)

    def addSource(self, index):
        self.graph.add_tedge(self.nodes[index], np.inf, 0)


    def addSink(self, index):
        self.graph.add_tedge(self.nodes[index], 0, np.inf)

    def getCrossEdges(self):
        cross_edges = []
        for edge in self.edges:
            node1 = edge[0]
            node2 = edge[1]
            if self.cut[node1] ^ self.cut[node2]:
                # if 1 and 2 are in different sets of the cut => 1 ^ 2 = True
                cross_edges.append((node1, node2))

        return cross_edges
    
    def get_cut(self):
        return self.cut