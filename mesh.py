""" 
Mesh.py
Author: Anirudh Pulavarthy

"""

from gudhi.simplex_tree import SimplexTree
import fileReader
import meshHelpers
import numpy as np
import cProfile #check this #Profilers for python

def isEdge(simplex):
    return len(simplex) == 2
    
def isTriangle(simplex): 
    return len(simplex) == 3

def edgesWithV(triangle, v):
    v0 = triangle[0]
    v1 = triangle[1]
    v2 = triangle[2]

    if v == v0:
        return [[v0, v1], [v0, v2]]
    elif v == v1:
        return [[v0, v1], [v1, v2]]
    else:
        return [[v0, v2], [v1, v2]]
    
############################################################################################

def dotProduct(vec1, vec2):
    assert len(vec1) == 3 and len(vec2) == 3
    val = (vec1[0] * vec2[0]) + (vec1[1] * vec2[1]) + (vec1[2] * vec2[2])
    return val

############################################################################################

def difference(vec_a, vec_b):
    assert len(vec_a) == len(vec_b)
    dif = []
    for i in range(len(vec_a)):
        dif.append(vec_a[i] - vec_b[i])

    return dif

def squared(list):
    return [i ** 2 for i in list]

############################################################################################
# This class defines a triangular mesh using gudhi.simplex_tree module
class Mesh(SimplexTree):

    ########################################################################################

    def getCofaces(self, sim, codim):
        """
        Fetch cofaces of a given simplex.
 
        :param sim: the simplex
        :param codim: codimension
        :return: vertices/edges/triangles as lists of tuples
        """
        cofaces = super().get_cofaces(sim, codim)
        return meshHelpers.removeFiltration(cofaces)
    
    ########################################################################################

    def findAdjacentTriangles(self, sim):
        b_IsEdge = isEdge(sim)
        b_IsTriangle = isTriangle(sim)

        # assert b_IsEdge or b_IsTriangle

        adjList = []
        if b_IsEdge:
            self.findAdjacentTrianglesUtil(sim, adjList)

        elif b_IsTriangle:
            edges = super().get_boundaries(sim)
            for edgeWithFiltration in edges:
                edge = edgeWithFiltration[0]
                self.findAdjacentTrianglesUtil(edge, adjList)

        return adjList
    
    def findAdjacentTrianglesUtil(self, sim, theList):
        newList = self.getCofaces(sim, 1)
        for element in newList:
            # if element not in theList:
            theList.append(element)

    ########################################################################################

    def load(self, fileName, non_critical_vertices):
        self.file_name = fileName
        vertsAndFaces = fileReader.read_off(fileName)
        self.vertices = vertsAndFaces[0]
        self.triangles = vertsAndFaces[1]
        self.non_critical = non_critical_vertices

        #Populate all the faces of simplex
        for face in self.triangles:
            super().insert(face)

        assert len(self.vertices) == super().num_vertices()
        self.populateTrianglesIndices()
        self.populateEdges()

    ########################################################################################

    def getSimpStr(self, s):
        # assert isTriangle(s)
        i = str(s)
        return i

    ########################################################################################

    def BFS(self, s):
        # String representation of triangle s is used as a hash value
        # for the BFS function. For a triangle [0, 1, 2] => '[0, 1, 2]' is the hash value

        # create a queue to BFS
        queue = []
        visited = {}

        queue.append(s)
        visited[tuple(s)] = True

        while queue:

            ip = len(visited)
            if ip % 1000 == 0:
                print(f"Processed {ip} triangles")

            s = queue.pop()

            adjacentTriangles = self.findAdjacentTriangles(s)
            for i in adjacentTriangles:
                if tuple(i) not in visited:
                    queue.append(i)
                    visited[tuple(s)] = True

        return visited

    ########################################################################################

    def isTwoConnected(self):

        print("Mesh contains...")
        print(f"\t {len(self.vertices)} vertices")
        print(f"\t{len(self.triangles)} faces")

        # fetch a random triangle
        randomTriangle = self.getRandomTriangle(0)
        visited = self.BFS(randomTriangle)

        numVisited = len(visited)
        print(f"BFS visited {numVisited} triangles")

        # if all triangles in mesh are visited, then mesh must be 2-cc
        return numVisited == len(self.triangles)

    ########################################################################################

    def getRandomTriangle(self, index = 0):
        """
        This function returns the first triangle that contains vertex given by index
        :return: A random triangle
        """

        # This function assumes vertex #0 is contained in at least one triangle
        cofaces = self.getCofaces([index], 2)

        for coface in cofaces:
            if isTriangle(coface):
                return coface   # return the first triangle that contains vert #0
            
        # If there is no triangle that contains vert #0, this function fails
        assert False
        
    ########################################################################################

    def getAllTriangles(self):
        allSimplices = super().get_simplices()
        triWithoutFiltration = meshHelpers.removeFiltration(allSimplices)
        return meshHelpers.getTriangles(triWithoutFiltration)

    ########################################################################################

    def starOfVertex(self, vert):
        starWithFiltration = super().get_star([vert])

        # remove filtration from the list and return
        return meshHelpers.removeFiltration(starWithFiltration)
    
    ############################################################################################

    def lowerStar(self, v):
        """
        Returns lower star of a vertex v
        :param: v: the vertex
        :return: lower star as a list of lists
        """

        star = self.starOfVertex(v)
        (edges_star, triangles_star) = meshHelpers.getEdgesAndTriangles(star)

        edges_lowerstar = []
        triangles_lowerstar = []

        index_v = self.indexOf(v)

        for edge in edges_star:
            index_a = self.indexOf(edge[0])
            index_b = self.indexOf(edge[1])

            if max(index_a, index_b) == index_v: edges_lowerstar.append(edge)

        for triangle in triangles_star:
            index_a = self.indexOf(triangle[0])
            index_b = self.indexOf(triangle[1])
            index_c = self.indexOf(triangle[2])

            if max(index_a, index_b, index_c) == index_v: triangles_lowerstar.append(triangle)

        return (edges_lowerstar, triangles_lowerstar)
    
    ############################################################################################

    def upperStar(self, v):
        """
        Returns upper star of a vertex v
        :param: v: the vertex
        :return: upper star as a list of lists
        """

        star = self.starOfVertex(v)
        (edges_star, triangles_star) = meshHelpers.getEdgesAndTriangles(star)

        edges_upperstar = []
        triangles_upperstar = []

        index_v = self.indexOf(v)

        for edge in edges_star:
            index_a = self.indexOf(edge[0])
            index_b = self.indexOf(edge[1])

            if min(index_a, index_b) == index_v: edges_upperstar.append(edge)

        for triangle in triangles_star:
            index_a = self.indexOf(triangle[0])
            index_b = self.indexOf(triangle[1])
            index_c = self.indexOf(triangle[2])

            if min(index_a, index_b, index_c) == index_v: triangles_upperstar.append(triangle)

        return (edges_upperstar, triangles_upperstar)
    
    ############################################################################################

    def indexOf(self, vertex):
        return self.ordered_verts[vertex]

    ############################################################################################

    def edgeExists(self, vert_a, vert_b):
        return super().find([vert_a, vert_b])

    ############################################################################################

    def isStar_ADisc(self, vertex):
        starOfV = self.starOfVertex(vertex)

        # filter all triangles in the star      
        triangles = []
        for item in starOfV:
            if isTriangle(item):
                triangles.append(item)
        
        num_Triangles = len(triangles)
        num_Visited = 0

        current = triangles[0]
        start = triangles[0]
        next = None

        self.e1 = None
        while next != start:
            num_Visited += 1
            next = self.findAdjacentUtil(vertex, current)   
            current = next

            if next == start:
                return num_Visited == num_Triangles
        
        assert False
        return next == start
                    
    ############################################################################################

    def findAdjacentUtil(self, vert, t1):
        # This function aims to find a triangle t2 that is adjacent to t1
        # Both t1 and t2 contain 'vert'
        
        # Find edges in t1 that contain vert
        edges = edgesWithV(t1, vert)
        e2 = edges[0] if self.e1 == edges[1] else edges[1]
        self.e1 = e2

        # select the other edge that hasn't been visited yet
        adjacent_triangles = self.findTrianglesAdjacentToEdge(e2)

        # one of the adjacent triangles is t1 itself
        # ignore t1 and return t2 which is the adjacent triangle..
        if adjacent_triangles[0] == t1:
            return adjacent_triangles[1]
        elif adjacent_triangles[1] == t1:
            return adjacent_triangles[0]
        else:
            assert False

    ############################################################################################

    def findTrianglesAdjacentToEdge(self, edge):
        return self.getCofaces(edge, 1)
    
    ############################################################################################

    def checkVertexConsistency(self, vertices):
        # ensure that star of each vertex is a disc

        for vertex in vertices:
            if not self.isStar_ADisc(vertex):
                return False

        return True
    
    ############################################################################################

    def checkEdgeConsistency(self):
        """
        Checks if all the edges are adjacent to exactly two edges.
        :return: True/False
        """
        for edge in self.edges:
            adjTriangles = super().get_cofaces(edge, 1)
            if len(adjTriangles) != 2:
                return False

        return True

    ############################################################################################

    def isManifold(self):

        print("\nChecking if the mesh is a manifold")
        # check if this is right
        allVertices = range(super().num_vertices())

        isEdgeConsistent = self.checkEdgeConsistency()
        if isEdgeConsistent:
            print("Mesh is edge consistent")
        else:
            print("Mesh is NOT edge consistent")

        isVertexConsistent = self.checkVertexConsistency(allVertices)
        if isVertexConsistent:
            print("Mesh is vertex consistent")
        else:
            print("Mesh is NOT vertex consistent")

        return isEdgeConsistent and isVertexConsistent

    ############################################################################################

    def computeDirection(self, verts):

        coord_a = self.vertices[verts[0] - 1]
        coord_b = self.vertices[verts[1] - 1]

        diffVec = difference(coord_a, coord_b)
        magnitude = (sum(squared(diffVec))) ** 0.5
        ret = [ i/magnitude for i in diffVec ]
        return ret

    ############################################################################################

    def computeDirection2(self, verts):

        coord_a = np.array(self.vertices[verts[0] - 1])
        coord_b = np.array(self.vertices[verts[1] - 1])

        diff = coord_a - coord_b
        magnitude = np.sqrt(np.sum(np.square(diff)))
        ret = diff / magnitude
        return ret
    
    ############################################################################################

    def compute_function_values(self, verts):
        index = 0
        self.function_values = {}
        
        print(f"Function values are being computed in the direction of vertices ({verts[0]}) to ({verts[1]})..")
        direction = self.computeDirection2(verts)
        for vertex in self.vertices:
            self.function_values[index] = np.dot(vertex, direction)
            index += 1
    
    ############################################################################################

    def order_vertices_by_function_values(self):
        self.sorted_function_values = sorted(self.function_values.items(), key=lambda x: x[1])

        index = 0
        self.ordered_verts = {}
        for vertAndPair in self.sorted_function_values:
            vert = vertAndPair[0]
            self.ordered_verts[vert] = index
            index += 1

    ############################################################################################

    def isCriticalVertex(self, v):
        """
        Determine if a vertex v is critical or not
        :param: v = vertex 
        :return: True if critical, False otherwise
        """

        if v in self.non_critical:  return False

        starOfV = self.starOfVertex(v)
        triangles_star = meshHelpers.getTriangles(starOfV)

        [edges_lowerstar, triangles_lowerstar] = self.lowerStar(v)

        if len(triangles_lowerstar) == 0 and len(edges_lowerstar) == 0:
            return True

        if len(triangles_star) == len(triangles_lowerstar):
            return True # considering all triangles in lower star should exist in star too!
        
        c1 = self.getC1(edges_lowerstar, triangles_lowerstar)

        if c1 > 1:
            return True
        
        if len(triangles_lowerstar) == 0:
            c2 = 0
        else:
            c2 = self.getC2(triangles_lowerstar)

        return (c1 + c2) != 1
    
    ############################################################################################

    def getC1(self, edges, triangles):
        """
        Count the number of edges in the lower star of v
        which are not adjacent to any triangles in the lower star of v

        :param: edges - edges in lower star
        :param: triangles - triangles in lower star
        :return: the value of C1
        """

        # this dictionary contains all triangles as its keys
        # and is used to find in constant time if a certain triangle
        # exists in the list 'triangles'
        c1 = 0
        triangles_dict = {}
        for item in triangles:
            triangles_dict[tuple(item)] = True

        for edge in edges:
            adjTriangles = self.getCofaces(edge, 1)
            t1 = tuple(adjTriangles[0])
            t2 = tuple(adjTriangles[1])
            if (t1 in triangles_dict) or (t2 in triangles_dict):
                pass
            else:
                c1 += 1 # both triangles are not in lower star

        return c1

    ############################################################################################

    def getC2(self, triangles):
        """
        Count the number of ‘2-connected’ components for the triangles in the lower star of v,
        where two triangles in the lower star of v are connected if and only if they share 
        a common edge which is also in the lower star of v

        :param: triangles - triangles in lower star
        :return: the value of C2
        """

        # this dictionary contains all triangles as its keys
        # and is used to find in constant time if a certain triangle
        # exists in the list 'triangles'
        triangles_dict = {}
        for item in triangles:
            triangles_dict[tuple(item)] = False

        # Do a BFS to find 2-cc starting at the first triangle
        t0 = triangles[0]

        queue = []
        queue.append(t0)
        triangles_dict[tuple(t0)] = True

        while queue:
            t1 = queue.pop()

            adj_triangles = self.findAdj_inLowerStar(t1, triangles_dict)

            for t2 in adj_triangles:
                if triangles_dict[tuple(t2)] == False:
                    queue.append(t2)
                    triangles_dict[tuple(t2)] = True

        # after BFS
        if False in triangles_dict.values():
            return 2    # more than one 2-connected components exist..
        else:
            return 1
        

    ############################################################################################

    def findAdj_inLowerStar(self, sim, dict_t):
        """ 
        Find the triangles adjacent to t that are also in the lower star of V
        :param: t = triangle t
        :param: dict_t = dict containing triangles that are in the lower star of V """
        b_IsEdge = isEdge(sim)
        b_IsTriangle = isTriangle(sim)

        adjList = []
        if b_IsEdge:
            self.findAdj_inLowerStarUtil(sim, dict_t, adjList)

        elif b_IsTriangle:
            edges = super().get_boundaries(sim)
            for edgeWithFiltration in edges:
                edge = edgeWithFiltration[0]
                self.findAdj_inLowerStarUtil(edge, dict_t, adjList)

        return adjList
    
    def findAdj_inLowerStarUtil(self, sim, dict_t, theList):
        """ 
        Find the triangles adjacent to t that are also in the lower star of V
        :param: sim = triangle t
        :param: dict_t = dict containing triangles that are in the lower star of V """
        newList = self.getCofaces(sim, 1)
        for element in newList:
            if tuple(element) in dict_t:
                if element not in theList:
                    theList.append(element)

    ############################################################################################

    def findCriticalVertices(self):
        critVertices = []
        numVert = super().num_vertices()
        for i in range(numVert):
            if self.isCriticalVertex(i):
                critVertices.append(i)
                
        return critVertices
    
    ############################################################################################

    def computeCriticalValues(self):
        self.critical_vertices = self.findCriticalVertices()
        
        critical_values = []

        for vertex in self.critical_vertices:
            critical_values.append(self.function_values[vertex])

        # indices of critical vertices in increasing order of function values
        self.ordered_critical_vertices = [x for _, x in sorted(zip(critical_values, self.critical_vertices))]

        critical_values.sort()
        self.critical_values = critical_values

    ############################################################################################

    def checkCompatibility(self):
        self.computeCriticalValues()

        even_CriticalValues = self.critical_values[0::2]
        odd_CriticalValues = self.critical_values[1::2]

        # even box contains indices of vertices in dual mesh (indices of triangles in current mesh)
        # that have critical values with even indices
        self.even_box = []
        self.odd_box = []

        for triangle in self.triangles:
            countEvenCriticalValues = 0
            for even_value in even_CriticalValues:
                if self.containsCriticalValue(triangle, even_value):
                    if countEvenCriticalValues > 1:
                        return False    # if the triangle contains more than one even critical values, mesh is incompatible
                    else:
                        countEvenCriticalValues += 1

                    for odd_value in odd_CriticalValues:
                        if self.containsCriticalValue(triangle, odd_value):
                            return False    # the triangle contains more than one critical value (1 odd + 1 even)
                        
                    dual_vertex_index = self.indices_triangles[str(triangle)]
                    self.even_box.append(dual_vertex_index)

            for odd_value in odd_CriticalValues:
                if self.containsCriticalValue(triangle, odd_value):
                    dual_vertex_index = self.indices_triangles[str(triangle)]
                    self.odd_box.append(dual_vertex_index)
                        

        return True


        allTriangles = self.triangles
        for triangle in allTriangles:

            count = 0
            for critical_value in self.critical_values:
                if self.containsCriticalValue(triangle, critical_value):
                    count += 1

                if count > 1:
                    print("Mesh has at least one triangle with more than one critical values..")
                    return False
        
        
        print("Mesh is compatible..\n")
        return True
    
    ############################################################################################

    # Before calling this function, make sure that the critical values are populated
    # If not, make a call to computeCriticalValues()

    def hasAnyCriticalValue(self, triangle):
        for value in self.critical_values:
            if self.containsCriticalValue(triangle, value):
                return True
            
        return False
    
    ############################################################################################

    def countCriticalValues(self, triangle):
        count = 0
        for value in self.critical_values:
            if self.containsCriticalValue(triangle, value):
                count += 1
            
        return count
        
    ############################################################################################

    def containsCriticalValue(self, triangle, value):
        v0 = triangle[0]
        v1 = triangle[1]
        v2 = triangle[2]

        val0 = self.function_values[v0]
        val1 = self.function_values[v1]
        val2 = self.function_values[v2]

        maxVal = max(val0, val1, val2)
        minVal = min(val0, val1, val2)

        return (value >= minVal) and (value <= maxVal)
    
    ############################################################################################
    
    def length(self, edge):
        assert self.find(edge)
        coord_A = self.vertices[edge[0]]
        coord_B = self.vertices[edge[1]]
        return np.linalg.norm(coord_A - coord_B)

    ############################################################################################

    def updateK(self, val):
        self.k = val
        # k is the number of triangles that have a critical value

    ############################################################################################
    
    def edgesContaining(self, vertex):
        cofaces = self.getCofaces(vertex, 1)
        edges = np.array(cofaces)
        return edges
    
    ############################################################################################

    def populateTrianglesIndices(self):
        for triangle in self.triangles:
            triangle.sort()

        self.indices_triangles = {}
        index = 0
        for triangle in self.triangles:
            self.indices_triangles[str(triangle)] = index
            index += 1

    ############################################################################################
        
    def getTriangleIndex(self, triangle):
        triangle.sort()
        return self.indices_triangles[str(triangle)]
    
    ############################################################################################

    def populateEdges(self):
        allSimplices = super().get_simplices()
        edgesWithoutFiltration = meshHelpers.removeFiltration(allSimplices)
        allEdges = meshHelpers.getEdges(edgesWithoutFiltration)
        self.edges = np.array(allEdges)
    
    ############################################################################################

    def getEdgeLength(self, index):
        edge = self.edges[index]
        return self.length(edge)

    ############################################################################################

    def getTrianglesAdjacentToEdge(self, edge):
        assert len(edge) == 2
        return self.getCofaces(edge, 1)
        
    ############################################################################################
    
    def outputCrossEdges(self, dual_edges):
        edges = []
        for dual_edge in dual_edges:
            index_triangle_1 = dual_edge[0]
            index_triangle_2 = dual_edge[1]

            t1 = self.triangles[index_triangle_1]
            t2 = self.triangles[index_triangle_2]

            common_edge = np.intersect1d(t1, t2)
            edges.append(common_edge)

        return edges
    
    def verify(self, cut):
        for evenVal in self.even_box:
            for oddVal in self.odd_box:
                if cut[evenVal] ^ cut[oddVal]:
                    pass #print("its okay")
                else:
                    print(f"Even is {evenVal} and odd is {oddVal}")

        for evenVal1 in self.even_box:
            for evenVal2 in self.even_box:
                if evenVal1 == evenVal2:
                    pass
                elif cut[evenVal1] ^ cut[evenVal2]:
                    print(f"Even1 is {evenVal1} and even2 is {evenVal2}")
                else:
                    pass

        for evenVal1 in self.odd_box:
            for evenVal2 in self.odd_box:
                if evenVal1 == evenVal2:
                    pass
                elif cut[evenVal1] ^ cut[evenVal2]:
                    print(f"Odd1 is {evenVal1} and Odd2 is {evenVal2}")
                else:
                    pass

    ############################################################################################

    ##
    ## This method returns a list of simplices in K_ij such that all vertices of simplex have
    ## their function values between the critical values i_alpha1 and i_alpha2 (like below)
    ## i_alpha1 < f(v) < i_alpha2
    ##
                
    def get_K_ij(self, i_alpha1, i_alpha2):
        ret = []

        in_range = lambda x, a, b: a < x < b

        alpha1 = self.getAlpha(i_alpha1)
        alpha2 = self.getAlpha(i_alpha2)

        verts = self.get_vertices_with_function_values_between(i_alpha1, i_alpha2, True)
        ret.extend([vert] for vert in verts)

        for edge in self.edges:
            function_value_0 = self.function_values[edge[0]]
            function_value_1 = self.function_values[edge[1]]

            if in_range(function_value_0, alpha1, alpha2) \
                and in_range(function_value_1, alpha1, alpha2):

                ret.append(edge.tolist())

        
        for triangle in self.triangles:
            function_value_0 = self.function_values[triangle[0]]
            function_value_1 = self.function_values[triangle[1]]
            function_value_2 = self.function_values[triangle[2]]

            if in_range(function_value_0, alpha1, alpha2) \
                and in_range(function_value_1, alpha1, alpha2) \
                and in_range(function_value_2, alpha1, alpha2):

                ret.append(triangle.tolist())

        return ret
        ret = []
        alpha1 = self.getAlpha(i_alpha1)
        alpha2 = self.getAlpha(i_alpha2)

        capture = False

        for vertex_index, function_val in self.sorted_function_values:
            if function_val == alpha1:
                capture = True
                ret.append(vertex_index)
            elif function_val == alpha2:
                return ret
            elif capture:
                ret.append(vertex_index)

        print("Reached critical value = +inf")
        return ret
    
    ############################################################################################
    ##
    ## This method returns a list of vertices such that alpha1 <= f(v) < alpha2
    ## If less_than_flag is set to True, condition will be changed to alpha1 < f(v) < alpha2

    def get_vertices_with_function_values_between(self, i_alpha1, i_alpha2, less_than_flag = False):
        ret = []
        alpha1 = self.getAlpha(i_alpha1)
        alpha2 = self.getAlpha(i_alpha2)

        capture = False
        if np.isneginf(alpha1):
            capture = True

        for vertex_index, function_val in self.sorted_function_values:
            if function_val == alpha1:
                capture = True

                if less_than_flag == False:
                    ret.append(vertex_index)

            elif function_val == alpha2:
                break
            elif capture:
                ret.append(vertex_index)

        return ret
    
    ############################################################################################
    ##
    ## This method returns a list of vertices such that alpha1 < f(v) <= alpha2

    def get_vertices_with_function_values_between_2(self, i_alpha1, i_alpha2):
        ret = []
        alpha1 = self.getAlpha(i_alpha1)
        alpha2 = self.getAlpha(i_alpha2)

        capture = False
        if np.isneginf(alpha1):
            capture = True

        for vertex_index, function_val in self.sorted_function_values:
            if function_val == alpha1:
                capture = True
            elif function_val == alpha2:
                ret.append(vertex_index)
                break
            elif capture:
                ret.append(vertex_index)

        return ret
    
    ############################################################################################
    
    # get critical value of critical vertex at the given 'index'
    def getAlpha(self, index):
        assert index >= 0 and index < len(self.critical_values)
        return self.critical_values[index]
    
    ############################################################################################

    # This function returns the index of V containing the speficied 'function_value'
    # Index refers to zero-based index in list self.sorted_function_values
    def indexOfV(self, function_value):
        index = 0

        # Binary search the list..
        for vertex_index, function_val in self.sorted_function_values:
            if function_val == function_value:
                return index
            index += 1

    ############################################################################################
    
    def populate_K_ij(self, simplex_tree, b, d):

        list_K = self.get_K_ij(b, d)

        for item in list_K:
            simplex_tree.insert(np.array(item), 1.0)

        return
    
    ############################################################################################

    def has_critical_vertex_i(self, triangle):
        
        for value in self.critical_values:
            if self.containsCriticalValue(triangle, value):
                return self.critical_values.index(value)
            
        return -1

        if self.containsCriticalValue(triangle, f0):
            return self.critical_values.index(f0)
        elif f1 in self.critical_values:
            return self.critical_values.index(f1)
        elif f2 in self.critical_values:
            return self.critical_values.index(f2)

        '''
        if vertex0 in self.ordered_critical_vertices:
            return self.ordered_critical_vertices.index(vertex0) + 1
        
        if vertex1 in self.ordered_critical_vertices:
            return self.ordered_critical_vertices.index(vertex1) + 1
        
        if vertex2 in self.ordered_critical_vertices:
            return self.ordered_critical_vertices.index(vertex2) + 1
        '''
        
        return -1