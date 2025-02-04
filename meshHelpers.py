
def getVertices(sim):
    """
    Fetch all vertices (simplices of 1-dimesion) an array of simplices.

    :param sim: the simplex
    :return: vertices as lists of lists
    """
    verts = []
    for s in sim:
        if len(s[0]) == 1:
            verts.append(s[0])
    
    return verts

############################################################################################

def getEdges(sim):
    """
    Fetch all edges (simplices of 2-dimesion) in an array of simplices.

    :param sim: the simplex
    :return: edges as lists of lists
    """
    edges = []
    for s in sim:
        if len(s) == 2:
            edges.append(s)
    
    return edges

############################################################################################

def getTriangles(sim):
    """
    Fetch all triangles (simplices of 3-dimesion) in an array of simplices.

    :param sim: the simplex
    :return: list of triangles from the list of simplices
    """

    # triangles = dict()
    triangles = []
    for s in sim:
        if len(s) == 3:
            triangles.append(s)
    
    return triangles

############################################################################################

def removeFiltration(simplicialcomplex):
    """
    Remove filtration from a list of simplices

    :param sim: the simplex
    :return: A list containing only simplices
    """

    return [simplex[0] for simplex in simplicialcomplex]

############################################################################################

def removeFiltration2(simplicialcomplex):
    """
    Remove filtration from a list of simplices
    Filtration 0.5 indicates a deleted simplex. Therefore remove it too.

    :param sim: the simplex
    :return: A list containing only simplices
    """

    return [simplex[0] for simplex in simplicialcomplex if simplex[1] != 0.5]

############################################################################################

def getEdgesAndTriangles(simplex_array):
    """
    Get edges and triangles from an array of simplices

    :param simplex_array: array of simplices
    :return: list containing [edges, triangles]
    """

    triangles = []
    edges = []

    for simplex in simplex_array:
        if len(simplex) == 2: edges.append(simplex)
        elif len(simplex) == 3: triangles.append(simplex)

    return (edges, triangles)