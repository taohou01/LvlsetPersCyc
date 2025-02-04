""" 
Filtration.py
Author: Anirudh Pulavarthy

"""
import itertools
import fileWriter
import meshHelpers
import os
import maxflow
import numpy as np
from dual_graph_step_5 import dual_step_5
from dual_graph_step_6 import dual_step_6
from pyfzz import pyfzz
from gudhi.simplex_tree import SimplexTree

################################################################################################################################
# These flags must be set/unset to be able to write to ply files
ply_flags = {f'flag{i+1}': False for i in range(6)}
'''
Using the information below, enable the flags as necessary
To enable, set them to True in ACTIVATE.py
(WTP = write to ply)

flag1 : This flag lets you WTP the current state of the complex after each forward/backward arrow in an L filtration.
flag2 : This flag lets you WTP the complexes k1 and k2 (k tilde and k beta bar as described in the document).
flag3 : In the list 'connected_components', enable this flag to WTP any one of the components by specifying the 'index'.
flag4 : For the above connected_component, enable this flag to WTP its boundary.
flag5 : This flag WTP only the connected_components that have the edge sigmabetaminus1.
flag6 : WTP the cross edge Z_j in Step 5 of Closed Open Step 2.
'''
################################################################################################################################

enable_checks = True
counter = 0
LF_map = {}

class interval:
  def __init__(self, max):
      self.x = 0
      self.y = 1
      self.max = max - 1 # zero based index

  def __iter__(self):
    yield (self.x, self.y)

    while (self.x != self.max):
        if (self.y - self.x) == 1:
            self.y += 1
            yield (self.x, self.y)
        else:
            self.x += 1
            yield (self.x, self.y)


# Insert simplices between 'start' and 'end' into 'filtration' and use the 'mesh' as needed
def ExecuteInsert(filtration, tree_kij, interval, mesh):

    start = interval[0]
    end = interval[1]

    '''
    K_ij = mesh.get_K_ij(start, start + 1)

    (edges, triangles) = meshHelpers.getEdgesAndTriangles(K_ij)
    for edge in edges:
        tree_kij.insert(np.array(edge))

    for triangle in triangles:
        tree_kij.insert(np.array(triangle))

    vertices_in_K_ij = meshHelpers.getVertices(K_ij)
    '''

    vertices_to_expand = mesh.get_vertices_with_function_values_between(start + 1, end)

    ##################################################################################################################
    # Checks are disabled later...
    if enable_checks: 
        # For a forward arrow K(i,i+1) ,→ K(i,i+2) in L(f),
        # let u1, u2, . . . , uk be all the vertices with function values in [αi+1, αi+2) 
        # such that f(u1) < f(u2) < · · · < f(uk)
        # We are checking for u1 = vi+1

        # Instead of comparing the vertex indices, I am comparing the function value of the vertex
        # By doing this, I am avoiding writing another function in mesh class that fetches the
        # vertex index of critical vertex by function value

        func_value1 = mesh.function_values[vertices_to_expand[0]]       # function value of u1
        func_value2 = mesh.critical_values[start + 1]                   # function value of (v1 + 1)
        assert func_value1 == func_value2

        # here comparison is done with start + 1, because 'interval' has (alpha, alpha + 2)
        # therefore we can use (start + 1) or (end - 1). I chose the former.
    ##################################################################################################################
    
    # all vertices in K(i,j) are ordered by the function values
    for vertex in vertices_to_expand:
        InsertSimplex([vertex], filtration, tree_kij, 1)

        [edges_lowerstar, triangles_lowerstar] = mesh.lowerStar(vertex)

        for edge in edges_lowerstar:
            InsertSimplex(edge, filtration, tree_kij, 2)
    
        for triangle in triangles_lowerstar:
            InsertSimplex(triangle, filtration, tree_kij, 3)
    
    # Update the map
    LF_map[(start, end - 1)] = counter    

    # After the expanding the arrow K(i,i+1) ==> K(i,i+2)
    # assert that expansion gives K(i, i+2)
    if enable_checks:
        assert (start + 2 == end)

        simplices_in_tree_after_expansion = meshHelpers.removeFiltration2(tree_kij.get_simplices())
        k_i_iplus2 = mesh.get_K_ij(start, end)

        missing_simplices = []
        for simplex in simplices_in_tree_after_expansion:
            # assert simplex in k_i_iplus2
            if not simplex in k_i_iplus2:
                missing_simplices.append(simplex)


        # Here the complexes "simplices_in_tree_after_expansion" and "k_i_iplus2" must have the same simplices
        # Instead of comparing simplex by simplex, I decided to check if the lengths of both the arrays are equal.
        # Just for simplicity.
        assert (len(simplices_in_tree_after_expansion) == len(k_i_iplus2))

    return


# Insert simplices between 'start' and 'end' into 'filtration' and use the 'mesh' as needed
def ExecuteInsertOld(filtration, tree_kij, interval, mesh):

    start = interval[0]
    end = interval[1]

    K_ij = mesh.get_K_ij(end - 1, end)

    (edges, triangles) = meshHelpers.getEdgesAndTriangles(K_ij)
    for edge in edges:
        tree_kij.insert(np.array(edge))

    for triangle in triangles:
        tree_kij.insert(np.array(triangle))

    vertices_in_K_ij = meshHelpers.getVertices(K_ij)

    ##################################################################################################################
    # Checks are disabled later...
    if enable_checks: 
        # For a forward arrow K(i,i+1) ,→ K(i,i+2) in L(f),
        # let u1, u2, . . . , uk be all the vertices with function values in [αi+1, αi+2) 
        # such that f(u1) < f(u2) < · · · < f(uk)
        # We are checking for u1 = vi+1

        # Instead of comparing the vertex indices, I am comparing the function value of the vertex
        # By doing this, I am avoiding writing another function in mesh class that fetches the
        # vertex index of critical vertex by function value

        func_value1 = mesh.function_values[vertices_in_K_ij[0]]     # function value of u1
        func_value2 = mesh.critical_values[start + 1]   # function value of (v1 + 1)
        assert func_value1 == func_value2

        # here comparison is done with start + 1, because 'interval' has (alpha, alpha + 2)
        # therefore we can use (start + 1) or (end - 1). I chose the former.
    ##################################################################################################################
    
    # all vertices in K(i,j) are ordered by the function values
    for vertex in vertices_in_K_ij:
        np_vert = np.array([vertex])

        # Add Lower star of vertex to filtration
        # Insert the vertex first
        filtration.append(('i', [vertex]))

        ## check if the vertex is already there.. not needed
        tree_kij.insert(np_vert)

        [edges_lowerstar, triangles_lowerstar] = mesh.lowerStar(vertex)

        # Insert the edges later
        for edge in edges_lowerstar:
            if enable_checks:
                assert True
                #assert tree_kij.find([edge[0]])
                #assert tree_kij.find([edge[1]])

            filtration.append(('i', edge))
            tree_kij.insert(np.array(edge))
    
        # Insert the triangles last
        for triangle in triangles_lowerstar:
            if enable_checks:
                ##### BEFORE ADDING THE TRIANGLE TO SIMPLEX, VERIFY THAT THE EDGES ARE ADDED
                face0 = [triangle[0], triangle[1]]
                face1 = [triangle[0], triangle[2]]
                face2 = [triangle[1], triangle[2]]

                # Check that all the three faces are added to k_ij
                assert tree_kij.find(face0)
                assert tree_kij.find(face1)
                assert tree_kij.find(face2)

                # THIS SHOULD ASSERT FALSE, SINCE THE TRIANGLE IS NOT ADDED YET. THIS CODE WORKS..
                # assert !tree_kij.find(triangle)

            filtration.append(('i', triangle))
            tree_kij.insert(np.array(triangle))
    
    return

def DeleteSimplex(simplex, filtration, tree, type):

    if enable_checks:
        assert(isAlreadyInserted(simplex, tree) == True)

    if type == 3:
        # Delete the triangle
        filtration.append(('d', simplex))
        tree.assign_filtration(simplex, 0.5)

    elif type == 2:
        # Delete the edge
        filtration.append(('d', simplex))
        tree.assign_filtration(simplex, 0.5)
    
    elif type == 1:
        # Delete the vertex
        filtration.append(('d', simplex))
        tree.assign_filtration(simplex, 0.5)

    global counter
    counter += 1

def ExecuteDelete(filtration, tree_kij, interval, mesh):
    start = interval[0]
    end = interval[1]
    
    vertices_to_expand = mesh.get_vertices_with_function_values_between_2(start - 1, start)

    # all vertices in K(i,j) are ordered by the function values
    for vertex in vertices_to_expand:
        [edges_upperstar, triangles_upperstar] = mesh.upperStar(vertex)

        # Remove in the order - triangles, edges, vertex

        # Delete the triangles first
        for triangle in triangles_upperstar:
            DeleteSimplex(triangle, filtration, tree_kij, 3)

        # Delete the edges later
        for edge in edges_upperstar:
            DeleteSimplex(edge, filtration, tree_kij, 2)

        # Delete the vertex last
        DeleteSimplex([vertex], filtration, tree_kij, 1)

    # Update map
    LF_map[(start, end - 1)] = counter

    if enable_checks:
        assert (start + 1 == end)

        simplices_in_tree_after_expansion = meshHelpers.removeFiltration2(tree_kij.get_simplices())
        k_i_iplus1 = mesh.get_K_ij(start, end)

        missing_simplices = []
        for simplex in simplices_in_tree_after_expansion:
            # assert simplex in k_i_iplus2
            if not simplex in k_i_iplus1:
                missing_simplices.append(simplex)

        # Here the complexes "simplices_in_tree_after_expansion" and "k_i_iplus2" must have the same simplices
        # Instead of comparing simplex by simplex, I decided to check if the lengths of both the arrays are equal.
        # Just for simplicity.
        assert (len(simplices_in_tree_after_expansion) == len(k_i_iplus1))

    return


def isAlreadyInserted(simplex, tree):
    """
        Checks if the simplex is already inserted in the tree
        :param: simplex: the vertex / edge / triangle
        :return: True if the simplex is present
    """
    found = tree.find(simplex)
    if found:
        f = tree.filtration(simplex)
        return (f == 1.0)    # filtration = 0.5 => deleted, otherwise 1.0
    
    else:   return False

    
def InsertSimplex(simplex, filtration, tree, type):

    if enable_checks:
        assert(isAlreadyInserted(simplex, tree) == False)

    if type == 1:
        # Insert a vertex here
        filtration.append(('i', simplex))
        tree.insert(simplex, 1.0)

    elif type == 2:
        # Insert an edge here
        filtration.append(('i', simplex))
        tree.insert(simplex, 1.0)
        
    elif type == 3:
        # Insert a triangle here
        if enable_checks:
                ##### BEFORE ADDING THE TRIANGLE TO SIMPLEX, VERIFY THAT THE EDGES ARE ADDED
                face0 = [simplex[0], simplex[1]]
                face1 = [simplex[0], simplex[2]]
                face2 = [simplex[1], simplex[2]]

                # Check that all the three faces are added to k_ij
                assert tree.find(face0)
                assert tree.find(face1)
                assert tree.find(face2)

                # THIS SHOULD ASSERT FALSE, SINCE THE TRIANGLE IS NOT ADDED YET.
                # assert !tree.find(simplex)

        filtration.append(('i', simplex))
        tree.insert(simplex, 1.0)

    global counter
    counter += 1    # after inserting the simplex, add 1 to counter

def build_L_Filtration(file, mesh):
    critical_values = mesh.critical_values
    critical_values.insert(0, - np.inf)
    critical_values.append(np.inf)
    max = len(critical_values) - 1

    operations = itertools.cycle(['insert', 'delete'])
    pairwise_intervals = itertools.pairwise(interval(max))

    filtration = []
    tree_kij = SimplexTree()

    LF_map = {}
    LL = []
    sequence = {}
    F_indices = []
    F_indices.append(0)

    fileName, file_extension = os.path.splitext(file)
    for start, end in pairwise_intervals:
        op = next(operations)
        print(f"Operation {op} from {start} to {end}")
        sequence[str(start) + str(end)] = op

        size_before = len(filtration)
        if op == 'insert':
            ExecuteInsert(filtration, tree_kij, end, mesh)
        else:
            ExecuteDelete(filtration, tree_kij, end, mesh)

        size_after = len(filtration)
        print(f"{op}ed {size_after - size_before} simplices...\n")

        LF_map[size_after] = end
        LL.append( (size_after, end) )
        F_indices.append(size_after)

        if ply_flags['flag1']:
            newfileName = fileName + "_" + str(end)
            fileWriter.write_simplex_tree_to_ply(newfileName, tree_kij, mesh.vertices)  #anirudh
    
    return filtration, F_indices, LF_map, sequence

def Barcode(F_filtr):
    fzz = pyfzz()
    bars = fzz.compute_zigzag(F_filtr)
    barsWithoutDimension = [ (k[0], k[1]) for k in bars if k[2] == 1]
        
    return barsWithoutDimension

def induceBarcodeold(bars, indices):
    induced_barcode = {}
    for birth, death in bars:
        left = right = None
        #print(f"birth = {birth}, death = {death}")

        for val in range(birth, death + 1):
            #print(f"Checking for value = {val}")
            if val in indices:
                left = val
                print(f"K(x, x') = {left}")
                break
        
        for val in range(death, birth - 1, -1):
            #print(f"Checking for value = {val}")
            if val in indices:
                right = val
                induced_barcode[(birth, death)] = (left, right)
                print(f"K(y, y') = {right}")
                print(f"K(x, x') = {left} and K(y, y') = {right}\n")
                break

    return induced_barcode

def induceBarcode2(bars, indices):
    induced_barcode = {}
    for birth, death in bars:
        left = right = None
        #print(f"birth = {birth}, death = {death}")

        for val in range(birth, death + 1):
            #print(f"Checking for value = {val}")
            if val in indices:
                left = val
                #print(f"K(x, x') = {left}")
                break
        
        for val in range(death, birth - 1, -1):
            #print(f"Checking for value = {val}")
            if val in indices:
                right = val
                induced_barcode[(birth, death)] = (left, right)
                #print(f"K(y, y') = {right}")
                #print(f"K(x, x') = {left} and K(y, y') = {right}\n")
                break
    return induced_barcode

def floorOfK(indices, K):
    '''
    Return the first integer less than K in the list 'indices'
    This function also assumes that 'indices' is a sorted list.

    Binary search is used to find the integer.
    '''

    low = 0
    high = len(indices) - 1

    while (low <= high):
        mid = (int)((low + high) / 2)
        if indices[mid] <= K:
            low = mid + 1
            floor = indices[high]
            next_to_floor = indices[high + 1] if high < len(indices) - 1 else -1    #remove this after testing
        else:
            high = mid - 1
            floor = indices[high]
            next_to_floor = indices[high + 1] if high < len(indices) - 1 else -1    #remove this after testing
        # loop ends here
    
    assert (floor <= K and next_to_floor > K)
            
    return floor

def ceilingOfK(indices, K):
    '''
    Return the first integer greater than K in the list 'indices'
    This function also assumes that 'indices' is a sorted list.

    Binary search is used to find the integer.
    '''

    low = 0
    high = len(indices) - 1

    while (low <= high):
        mid = (int)((low + high) / 2)
        if indices[mid] >= K:
            high = mid - 1

            before_ceiling = indices[low-1] if low > 0 else -1      # remove after testing
            ceiling = indices[low]
        else:
            low = mid + 1

            before_ceiling = indices[low-1] if low > 0 else -1      # remove after testing
            ceiling = indices[low]

        # loop ends here
    
    assert (ceiling >= K and before_ceiling < K)          # remove after testing
    return ceiling
            

# This version uses binary search
def induceBarcode(bars, indices):
    induced_barcode = {}

    for birth, death in bars:
        left = right = None

        left = ceilingOfK(indices, birth)
        right = floorOfK(indices, death)
        
        if left <= right:
            induced_barcode[(birth, death)] = (left, right)

    return induced_barcode

def run_closed_open_step2(output_from_step1, number_of_intervals):
    mesh, closed_open_dict, F_filtration, LF_map, sequence = output_from_step1

    keys_list = list(closed_open_dict.keys())
    values_list = list(closed_open_dict.values())

    msg = "Number of intervals entered must be less than the total"\
        " number of closed open intervals"
    assert number_of_intervals <= len(keys_list), msg

    index = 0
    while index < number_of_intervals:
        # Each value of index corresponds to each closed open interval
        # in the decreasing order of lengths

        key = keys_list[index]
        value = values_list[index]

        beta, delta = key       # corresponds to a closed open interval
        begin, end = value      # corresponds to respective induced interval

        b = begin[0] + 1 
        d = end[1]

        # k1 = mesh.get_K_ij(b - 1, d)
        # k1 = [[item] for item in k1]

        # k1      # This complex corresponds to K~ from the document
        # k2      # This complex corresponds to K-beta-bar from document
        k1 = SimplexTree()
        k2 = SimplexTree()

        mesh.populate_K_ij(k1, b - 1, d)
        k_deltaplus1 = buildComplex(delta + 1, F_filtration)

        for item in k_deltaplus1: # K1 is K(b - 1, d) UNION K(delta + 1)
            k1.insert(np.array(item), 1.0)

        k2_init = buildComplex(beta, F_filtration)
        for item in k2_init:
            k2.insert(np.array(item), 1.0)

        # add triangles of mesh K whose all edges are in K beta
        for triangle in triangles_with_edges_in_k2(mesh, k2):
            k2.insert(triangle, 1.0)

        if ply_flags['flag2']:
            fileWriter.write_simplex_tree_to_ply("k1.ply", k1, mesh.vertices)
            fileWriter.write_simplex_tree_to_ply("k2.ply", k2, mesh.vertices)

        allSimplices = k1.get_simplices()
        triWithoutFiltration = meshHelpers.removeFiltration(allSimplices)
        triangles_in_k1 = meshHelpers.getTriangles(triWithoutFiltration)

        count1 = len(triangles_in_k1)

        allSimplices2 = k2.get_simplices()
        triWithoutFiltration2 = meshHelpers.removeFiltration(allSimplices2)
        triangles_in_k2 = meshHelpers.getTriangles(triWithoutFiltration2)

        count2 = len(triangles_in_k2)

        count1 = 0
        connected_components = []
        for triangle in triangles_in_k1:
            if not k2.find(triangle):
                count1 += 1
                filt = k1.filtration(triangle)
                if k1.filtration(triangle) == 1.0:  # indicates an unparsed triangle
                    # compute 2 connected component of triangle
                    comp = compute_connected_component(triangle, k1, k2)
                    connected_components.append(comp)

        print("========================================")
        print(f"Number of connected components: {len(connected_components)}")

        # USE THIS FOLLOWING STATEMENTS TO PRINT ANY CONNECTED COMPONENT AND ITS BOUNDARY TO A PLY
        if ply_flags['flag3']:
            write_connected_component_to_ply(connected_components, index = 0, mesh = mesh)
        
        if ply_flags['flag4']:
            write_boundary_to_ply(connected_components, index = 0, mesh = mesh)

        new_connected_components = connected_components_with_boundary_in_k2(connected_components, k2)
        print("========================================")
        print(f"\nNumber of connected components whose boundaries are subset of k2: {len(new_connected_components)}")

        if enable_checks:
            assert(len(F_filtration[beta - 1][1]) == 2)     # assert that sigma(beta - 1) is an edge
            assert((F_filtration[beta - 1][0]) == 'i')      # assert that operation in filtration is an insertion

            edge = F_filtration[beta - 1][1]    # the edge sigma(beta - 1)
            result = find_components_with_sigmabetaminus1_in_boundary(new_connected_components, edge)
            components_withedge = sum(x for x in result)

            # only one component in the connected components must have the edge sigma beta minus1
            assert (components_withedge == 1)       

            if ply_flags['flag5']:
                # for each of the connected component, output the component to ply and its boundary to ply
                for index, comp in enumerate(new_connected_components):
                    write_connected_component_to_ply(new_connected_components, index, mesh)
                    write_boundary_to_ply(new_connected_components, index, mesh)

        edges_in_Z_j = []
        edges_in_Z_0 = []
        for j, c_j in enumerate(new_connected_components):
            g_j = dual_step_5(c_j, mesh, birth=b, death=d)
            edges_in_Z_j_current_component = g_j.output_cross_edges()

            if j == 0:
                edges_in_Z_0 = edges_in_Z_j_current_component

            edges_in_Z_j.append(edges_in_Z_j_current_component)

            if ply_flags['flag6']:
                write_edges_to_ply(f"edges_in_Z_{j}", edges_in_Z_j_current_component, mesh)

        ####################################################################################
        # step 6 of closed-open step 2
        
        g_step6 = dual_step_6(k2, edges_in_Z_j, new_connected_components, mesh)
        edges_step_6 = g_step6.output_cross_edges()

        final_edges = edges_in_Z_0
        final_edges.extend(edges_step_6)

        file_name = get_final_edges_file_name(mesh.file_name, i = index)
        print("================================================")
        print(np.version)
        print(maxflow.version)
        print("================================================")
        write_edges_to_ply(file_name, final_edges, mesh)

        index += 1  # Move to the next closed open interval

def get_final_edges_file_name(file, i):
        split_name = os.path.splitext(file)
        
        #split_name[0] is the file title
        new_name = split_name[0].replace("_", "")
        ret = f"{new_name}_cocyc_intv[{i}]"
        return ret

def write_connected_component_to_ply(connected_components, index, mesh):
    tree_comp = SimplexTree()
    for item in connected_components[index]:
        tree_comp.insert(np.array(item), 1.0)
    
    # Write component to ply
    fileWriter.write_simplex_tree_to_ply(f"comp_{index}.ply", tree_comp, mesh.vertices)

def write_edges_to_ply(filename, edges, mesh):
    new_simplextree = SimplexTree()
    for item in edges:
        new_simplextree.insert(np.array(item), 1.0)

    # Write the edges in Zj to ply
    fileWriter.write_simplex_tree_to_ply(filename, new_simplextree, mesh.vertices)

def write_boundary_to_ply(connected_components, index, mesh):
    boundary = computeBoundary(connected_components[index])

    tree_boundary = SimplexTree()
    for item in boundary:
        tree_boundary.insert(np.array(item), 1.0)

    # Write boundary of the component to ply
    fileWriter.write_simplex_tree_to_ply(f"boundary_comp_{index}.ply", tree_boundary, mesh.vertices)

def find_components_with_sigmabetaminus1_in_boundary(connected_components, the_edge):
    ret = []
    for component in connected_components:
        boundary = computeBoundary(component)
        if tuple(the_edge) in boundary:
            ret.append(1)
        else:
            ret.append(0)

    return ret

def connected_components_with_boundary_in_k2(connected_components, k2):
    ret = []
    
    for component in connected_components:
        boundary = computeBoundary(component)

        if boundarysubsetofk2(boundary, k2):
            ret.append(component)

    return ret

def boundarysubsetofk2(boundary, k2):
    for item in boundary:
        if not k2.find(item):
            return False

    return True

def compute_connected_component(triangle, k1, k2):
    ret = []
    queue = []
    queue.append(triangle)              # add the triangle to the queue

    ret.append(tuple(triangle))         # add the triangle to the 2 connected component that is returned by the function
    k1.assign_filtration(triangle, 0.5) # marking triangle as visited
    
    while queue:
        s = queue.pop()

        adjacentTriangles = find_adjacent_triangles_in_k1_not_in_k2(s, k1, k2)

        for i in adjacentTriangles:
            if k1.filtration(i) != 0.5:         # if triangle i is not visited
                queue.append(i)                 # add it to queue

                ret.append(tuple(i))            # add the triangle to the 2 connected component
                k1.assign_filtration(i, 0.5)    # marking triangle as visited
    
    return ret

def find_adjacent_triangles_in_k1_not_in_k2(triangle, k1, k2):

    adjacent_triangles_not_in_k1 = []
    e0 = [triangle[0], triangle[1]]
    e1 = [triangle[0], triangle[2]]
    e2 = [triangle[1], triangle[2]]

    edges = [e0, e1, e2]

    for edge in edges:
        if k1.find(edge) and (not k2.find(edge)):

            cofaces = k1.get_cofaces(edge, 1)
            all_adjacent_triangles = meshHelpers.removeFiltration(cofaces)

            for t in all_adjacent_triangles:
                if k1.find(t) and (not k2.find(t)):
                    adjacent_triangles_not_in_k1.append(t)

    return adjacent_triangles_not_in_k1

    
def triangles_with_edges_in_k2(mesh, k2):
    triangles = []

    #all_edges_in_k2 = [simplex for simplex in k2 if len(simplex) == 2]          # all simplexes (that are lists) with length 2 are edges
    #edges_dict = { tuple(edge): True for edge in all_edges_in_k2 }              # This dict is a hash map that contains all edges in k2

    for tri in mesh.triangles:
        e0 = [tri[0], tri[1]]
        e1 = [tri[0], tri[2]]
        e2 = [tri[1], tri[2]]

        e0_OK = k2.find(e0)
        e1_OK = k2.find(e1)
        e2_OK = k2.find(e2)

        if e0_OK and e1_OK and e2_OK:
            triangles.append(tri)

    return triangles


def buildComplex(index, F_filtration):
    complex = {}
    cur_index = 0

    while cur_index < index:
        op, simplex = F_filtration[cur_index]
        if op == 'i':
            complex[tuple(simplex)] = True
        elif op == 'd':
            complex[tuple(simplex)] = False

        cur_index += 1

    ret = [key for key, value in complex.items() if value is True]
    return ret

def computeBoundary(A):
    #allSimplices = A.get_simplices()
    #edgesWithoutFiltration = meshHelpers.removeFiltration(allSimplices)
    #allEdges = meshHelpers.getEdges(edgesWithoutFiltration)

    #edges_in_A = [simplex for simplex in A if len(simplex) == 2]        # filter edges
    #triangles_in_A = [simplex for simplex in A if len(simplex) == 3]    # filter triangles

    #edges_dict = { tuple(edge): False for edge in edges_in_A }

    edges_set = set()
    for triangle in A:

        assert (len(triangle) == 3)       # ensures A is strictly a triangle set

        e0 = (triangle[0], triangle[1])
        e1 = (triangle[0], triangle[2])
        e2 = (triangle[1], triangle[2])

        if e0 in edges_set:
            edges_set.remove(e0)
        else:
            edges_set.add(e0)

        if e1 in edges_set:
            edges_set.remove(e1)
        else:
            edges_set.add(e1)

        if e2 in edges_set:
            edges_set.remove(e2)
        else:
            edges_set.add(e2)

    #Bret = [item for item in edges_set if edges_set[item] == True]
    return edges_set
    

def is_interval_closed_open(k_beta, k_delta, sequence):
    #return True
    (begin, end) = k_beta
    print(k_beta, k_delta)

    if end - begin != 2:
        # This would simply mean k_beta does not have a forward arrow
        return False
    
    key = str((begin, begin + 1)) + str((begin, end))

    if not sequence[key] == 'insert':
        return False
    
    (begin, end) = k_delta

    if end - begin != 1:
        # This would simply mean k_delta does not have a forward arrow
        return False
    
    key = str((begin, end)) + str((begin + 1, end))

    if not sequence[key] == 'insert':
        return False

    return True

'''
For a given x, this function returns determines indices i and j within sorted_list
such that  sorted_list[i] < x < sorted_list[j]

If x is found in the list, i and j both indicate position of x in the list
'''
def find_indices(sorted_list, x):
    n = len(sorted_list)
    
    # Edge cases: x is out of the bounds of the list
    if x <= sorted_list[0]:
        return None, 0  # No valid i, j = 0
    if x >= sorted_list[-1]:
        return n-1, None  # i = n-1, No valid j
    
    # Binary search to find the correct positions
    low, high = 0, n - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] == x:
            # If x is exactly in the list, i and j should be around it
            if mid > 0 and mid < n - 1:
                return mid - 1, mid + 1
            elif mid == 0:
                return None, 1  # No valid i, j = 1
            elif mid == n - 1:
                return n - 2, None  # i = n-2, No valid j
        elif sorted_list[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    # After the loop, low is the first position where sorted_list[low] > x
    j = low
    i = low - 1

    return i, j

def is_interval_closed_open2(birth, death, indices):
    i1, j1 = find_indices(indices, birth)
    i2, j2 = find_indices(indices, death)

    # check if K(beta - 1) ----> K(beta) is a forward arrow or not
    # if this is a forward arrow, i1 should be even
    if i1 % 2 == 1: return False

    # check if K(delta) ----> K(delta + 1) is a forward arrow or not
    # if this is a forward arrow, i2 should be even
    if i2 % 2 == 1: return False

    return True


def run_closed_open_step1(file, mesh):
    F_filtr, indices, LF_map, sequence = build_L_Filtration(file, mesh)

    barcode = Barcode(F_filtr)
    induced_barcode = induceBarcode(barcode, indices)

    closed_open = {}
    print("Barcode on L(F)....")

    for (birth, death), (induced_birth, induced_death) in induced_barcode.items():
        interval_begin = LF_map[induced_birth]
        interval_end = LF_map[induced_death]

        print(f"\t({birth}, {death}) => ({interval_begin}, {interval_end})")

        if (enable_checks == False) or is_interval_closed_open2(birth, death, indices):
            # assert that begin is of the form (b - 1, b + 1)
            (b, c) = interval_begin

            if enable_checks:
                assert c == b + 2

            # assert that end is of the form (d - 1, d)
            (b, c) = interval_end
            if enable_checks:
                assert c == b + 1
            
            closed_open[(birth, death)] = (interval_begin, interval_end)

    print("End of barcode")
    print("========================================")
    print("\nThese are closed open intervals before ranking by decreasing order of lengths:")
    print(closed_open)
    print("========================================")

    vals = mesh.critical_values
    lengths = []
    for interval in closed_open.values():
        first = interval[0]     #$ This should be of the form (b - 1, b + 1)
        second = interval[1]    #$ This should be of the form (d - 1, d)

        low = first[0] + 1 
        high = second[1]

        alpha_b = mesh.critical_values[low]
        alpha_d = mesh.critical_values[high]

        length = abs(alpha_d - alpha_b)
        lengths.append(length)

    closed_open = rank_intervals(closed_open, lengths)

    print("\nThese are closed open intervals after ranking by interval lengths:")
    print(closed_open)
    print("========================================")

    return mesh, closed_open, F_filtr, LF_map, sequence

def rank_intervals(closed_open_intervals, lengths):
    dict_lengths = {}
    new_dict = {}
 
    inverted_dict = {value: key for key, value in closed_open_intervals.items()}

    only_intervals = list(closed_open_intervals.values())
    dict_lengths = {only_intervals[i]: lengths[i] for i in range(len(lengths))}

    # sorting of dictionary based on value
    sorted_dict = {k: v for k, v in sorted(dict_lengths.items(), key=lambda item: item[1], reverse=True)}
    for i in sorted_dict.keys():
        new_dict_key = inverted_dict[i]
        new_dict[new_dict_key] = i

    return new_dict