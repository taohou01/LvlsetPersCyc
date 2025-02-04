import os
import numpy as np

################################################################################################

def getNewFilename1(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]

    newFile = f"{fileTitle}_vertices.off"

    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_vertices_{count}.off"
        count += 1

    return newFile

def doesLineHaveCritVertex(line, vertex):
    findStr1 = " " + str(vertex) + " "
    findStr2 = " " + str(vertex) + "\n"

    return (line.find(findStr1) != -1) or (line.find(findStr2) != -1)

def color_mesh_1(file, critical_vertices):
    """
    Color Mesh function 1
    This function generates a colored mesh.
    All the triangles containing critical vertices are colored in red
 
    :param file: path to the original mesh file
    :return: a file named x_vertices.off where x is the name of original file
    """

    assert os.path.exists(file)
 
    with open(file, 'r') as oldFile:

        file2 = getNewFilename1(file)
        with open(file2, 'w') as newFile:

            print(f"{file2} contains colored triangles with critical vertices\n")

            lines = oldFile.readlines()
            lines = [line.strip() for line in lines]
    
            assert lines[0] == 'OFF'
    
            parts = lines[1].split(' ')
            assert len(parts) == 3
    
            num_vertices = int(parts[0])
            assert num_vertices > 0
    
            num_faces = int(parts[1])
            assert num_faces > 0

            for i in range(num_vertices + 2):
                newFile.write(lines[i] + "\n")

            gray_string = " 0.5 0.5 0.5 0.5\n"
            red_string = " 1.0 0.0 0.0 0.5\n"

            for i in range(num_faces):
                line = lines[2 + num_vertices + i] + "\n"
                line_contains_critical_vertex = False

                for vertex in critical_vertices:
                    if doesLineHaveCritVertex(line, vertex):
                        line_contains_critical_vertex = True
                        break

                line = line.rstrip()
                if line_contains_critical_vertex:
                    line = line + red_string
                else:
                    line = line + gray_string

                newFile.write(line)

################################################################################################

def getNewFilename2(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]

    newFile = f"{fileTitle}_values.off"

    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_values_{count}.off"
        count += 1

    return newFile

def color_mesh_2(file, mesh):
    """
    Color Mesh function 2
    This function generates a colored mesh.
    All the triangles containing critical values are colored in red
 
    :param file: path to the original mesh file
    :return: a file named x_values.off where x is the name of original file
    """
 
    if not os.path.exists(file):
        print(f"Specfied file - {file} - does not exist")

    with open(file, 'r') as oldFile:

        file2 = getNewFilename2(file)
        with open(file2, 'w') as newFile:

            print(f"{file2} contains colored triangles with critical values")

            lines = oldFile.readlines()
            lines = [line.strip() for line in lines]
    
            assert lines[0] == 'OFF'
    
            parts = lines[1].split(' ')
            assert len(parts) == 3
    
            num_vertices = int(parts[0])
            assert num_vertices > 0
    
            num_faces = int(parts[1])
            assert num_faces > 0

            for i in range(num_vertices + 2):
                newFile.write(lines[i] + "\n")

            gray_string = " 0.5 0.5 0.5 0.5\n"
            red_string = " 1.0 0.0 0.0 0.5\n"

            count = 0
            for i in range(num_faces):
                line = lines[2 + num_vertices + i]
                triangle = [int(item) for item in line.split()]
                triangle = triangle[1:] # exclude "3"

                if mesh.hasAnyCriticalValue(triangle):
                    count += 1
                    line = line + red_string
                else:
                    line = line + gray_string

                newFile.write(line)

            mesh.updateK(count)
            return
        
################################################################################################

def hasCriticalVertex(triangle, critical_vertices):
    v1 = triangle[0]
    if v1 in critical_vertices:
        return True
    
    v2 = triangle[1]
    if v2 in critical_vertices:
        return True
    
    v3 = triangle[2]
    if v3 in critical_vertices:
        return True
    
    return False

def getNewFilename3(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]

    newFile = f"{fileTitle}_compatible.off"

    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_compatible_{count}.off"
        count += 1

    return newFile

def color_mesh_3(file, mesh):
    """
    Color Mesh function 3
    This function generates a colored mesh.
    Triangles adjacent to critical vertices are colored red
    Triangles which contain more than one critical values are colored blue (if any)

    :param file: path to the original mesh file
    :return: a file named x_compatibile.off where x is the name of original file
    """
 
    if not os.path.exists(file):
        print(f"Specfied file - {file} - does not exist")

    with open(file, 'r') as oldFile:

        file2 = getNewFilename3(file)
        with open(file2, 'w') as newFile:

            print(f"{file2} contains colored triangles with critical values")

            lines = oldFile.readlines()
            lines = [line.strip() for line in lines]
    
            assert lines[0] == 'OFF'
    
            parts = lines[1].split(' ')
            assert len(parts) == 3
    
            num_vertices = int(parts[0])
            assert num_vertices > 0
    
            num_faces = int(parts[1])
            assert num_faces > 0

            for i in range(num_vertices + 2):
                newFile.write(lines[i] + "\n")

            gray_string = " 0.5 0.5 0.5 0.5\n"
            blue_string = " 0.0 0.0 1.0 0.5\n"
            red_string = " 1.0 0.0 0.0 0.5\n"

            count = 0
            critical_vertices = mesh.findCriticalVertices()
            for i in range(num_faces):
                line = lines[2 + num_vertices + i]
                triangle = [int(item) for item in line.split()]
                triangle = triangle[1:] # exclude 3

                if mesh.countCriticalValues(triangle) > 1:
                    line = line + blue_string
                elif hasCriticalVertex(triangle, critical_vertices):
                    line = line + red_string
                else:
                    line = line + gray_string

                newFile.write(line)

            print(f"Finished generating colored mesh {file2}..\n")
            return
        
################################################################################################
        
def writePlyHeader(file, n_verts, n_faces, n_edges):

    file.write("ply\n")
    file.write("format ascii 1.0\n")

    # define vertex element with properties - x, y and z
    file.write(f"element vertex {2 * n_edges}\n")
    file.write("property float x\nproperty float y\nproperty float z\n")

    # define edge element with properties v1, v2, r, g and b (v1 and v2 are vertex indices, RGB is color)
    file.write(f"element edge {n_edges}\n")
    file.write("property int vertex1\nproperty int vertex2\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")

    # file.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
    # file.write(f"element face {n_faces + n_edges}\n")
    # file.write("property int vertex1\nproperty int vertex2\nproperty int vertex3\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
    # file.write("property list uchar int vertex_index\n")
    # file.write(f"element edge {n_edges}\n")
    # file.write("property int vertex1\nproperty int vertex2\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
    # file.write("property list uchar int vertex_index\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")

    file.write("end_header\n")

def getPlyFilename(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]

    newFile = f"{fileTitle}_oocyc.ply"
    count = 1

    while os.path.exists(newFile):
        newFile = f"{fileTitle}_oocyc_{count}.ply"
        count += 1

    return newFile

def color_mesh_4(file, mesh, edges):
    """
    Color Mesh function 4
    This function generates a PLY file with only outputted cross edges.
    All the edges are colored in blue
 
    :param file: path to the original mesh file
    :return: a file named x_oocyc.ply where x is the name of original file
    """

    if not os.path.exists(file):
        print(f"{file} - file does not exist")

    with open(file, 'r') as oldFile:

        file2 = getPlyFilename(file)
        with open(file2, 'w') as newFile:

            print(f"{file2} contains the .ply file with colored edges")

            lines = oldFile.readlines()
            lines = [line.strip() for line in lines]
    
            parts = lines[1].split(' ')

            num_vertices = int(parts[0])
            num_faces = int(parts[1])
            num_colored_edges = len(edges)

            writePlyHeader(newFile, num_vertices, num_faces, num_colored_edges)

            red = " 255 0 0\n"
            gray = " 128 128 128\n"
            # for i in range(num_vertices):
            #    newFile.write(lines[2 + i] + "\n")

            # for i in range(num_faces):
            #     line = lines[2 + num_vertices + i]
            #     line = line + red

            #     newFile.write(line)

            blue_string = "0 0 255\n"
            for edge in edges:
                addVertexToFile(newFile, mesh.vertices[edge[0]])
                addVertexToFile(newFile, mesh.vertices[edge[1]])

            index = 0
            for edge in edges:
                newFile.write(f"{index} {index + 1} {blue_string}")
                index += 2
                # newFile.write(f"3 {edges[i][0]} {edges[i][1]} {edges[i][1]} {blue_string}")
                # newFile.write(f"{edges[i][0]} {edges[i][1]} {blue_string}")

################################################################################################

def writeOFFHeader(file, numVerts, numFaces):
    file.write("OFF\n")
    file.write(f"{numVerts} {numFaces} 0\n")

def addVertexToFile(file, vert):
    file.write(f"{vert[0]} {vert[1]} {vert[2]}\n")

def addTriangleToList(npList, index, face):
    npList[index] = face

def getNewFilename4(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]

    newFile = f"{fileTitle}_edgesOnly.off"

    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_edgesOnly_{count}.off"
        count += 1

    return newFile

def edges_only(file, mesh, edges):
    """
    edges_only - This function generates an off file with colored edges.
 
    :param file: path to the original mesh file
    :return: a file named x_edgesOnly.off where x is the name of original file
    """
    if not os.path.exists(file):
        print(f"{file} - file does not exist")
        return
    else:
        file2 = getNewFilename4(file)
        with open(file2, 'w') as newFile:
            print(f"{file2} contains only colored edges")

            numVertices = 3 * len(edges)
            numFaces = len(edges)
            writeOFFHeader(newFile, numVertices, numFaces)

            faces_list = np.empty(shape=[len(edges), 3], dtype=np.int16)
            vert_index = 0
            face_index = 0
            # populate edges as triangles
            for edge in edges:
                vert1 = edge[0]
                vert2 = edge[1]
                addVertexToFile(newFile, mesh.vertices[vert1])
                addVertexToFile(newFile, mesh.vertices[vert2])
                addVertexToFile(newFile, mesh.vertices[vert2])

                # triangle = np.array(edge[0], edge[1], edge[1])
                # addTriangleToList(faces_list, index, triangle)
                triangle = np.array((vert_index, vert_index + 1, vert_index + 2))
                vert_index += 3

                faces_list[face_index] = triangle
                face_index += 1

            for face in faces_list:
                #addFaceToFile(file, face)
                newFile.write(f"3 {face[0]} {face[1]} {face[2]} 0.0 0.0 1.0 0.5\n")

################################################################################################

def getNewFilename5(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]

    newFile = f"{fileTitle}_oocyc.txt"

    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_oocyc_{count}.txt"
        count += 1

    return newFile

def writeEdgesToTextFile(file, edges):
    """
    writeEdgesToTextFile - This function writes outputted edges to a text file
 
    :param file: path to the original mesh file
    :return: a file named x_oocyc.txt where x is the name of original file
    """
    if not os.path.exists(file):
        print(f"{file} - file does not exist")

    file2 = getNewFilename5(file)
    with open(file2, 'w') as newFile:
        print(f"{file2} contains cross edges in a text file")
        for edge in edges:
            newFile.write(f"({edge[0]}, {edge[1]})\n")

########################################################################################

def getNewFileName6(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]
    newFile = f"{fileTitle}_even_odd.ply"
    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_even_odd_{count}.ply"
        count += 1
    return newFile
    
def even_odd_boxes(file, mesh):
    # This function has a few bugs. Instead use the even_odd_v2() function
    # I retained this because parts of this function can be reused if needed.
    if not os.path.exists(file):
        print(f"{file} - file does not exist")

    file2 = getNewFileName6(file)
    with open(file2, 'w') as newFile:
        print(f"{file2} contains even and odd boxes")

        # write header for even box
        newFile.write("ply\n")
        newFile.write("format ascii 1.0\n")

        # define vertex element with properties - x, y and z
        vert_count = len(mesh.even_box) + len(mesh.odd_box)
        newFile.write(f"element vertex {3 * vert_count}\n")
        newFile.write("property float x\nproperty float y\nproperty float z\n")

        even_count = len(mesh.even_box)
        odd_count = len(mesh.odd_box)
        newFile.write(f"element face {even_count + odd_count}\n")
        newFile.write("property list uchar int vertex_index\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        newFile.write("end_header\n")

        for index in mesh.even_box:
            t = mesh.triangles[index]
            addVertexToFile(newFile, mesh.vertices[t[0]])
            addVertexToFile(newFile, mesh.vertices[t[1]])
            addVertexToFile(newFile, mesh.vertices[t[2]])

        for index in mesh.odd_box:
            t = mesh.triangles[index]
            addVertexToFile(newFile, mesh.vertices[t[0]])
            addVertexToFile(newFile, mesh.vertices[t[1]])
            addVertexToFile(newFile, mesh.vertices[t[2]])

        red = "255 0 0"
        for i in range(even_count):
            # i = index
            newFile.write(f"3 {3 * i} {3 * i + 1} {3 * i + 2} {red}\n")
            # i += 3

        blue = "0 255 0"
        offset = 3 * even_count
        for i in range(odd_count):
            index = i + offset
            newFile.write(f"3 {index} {index + 1} {index + 2} {blue}\n")
            # newFile.write(f"3 {index} {index + 1} {index + 2} {blue}\n")
            index += 3

########################################################################################

def getNewFileName7(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]
    newFile = f"{fileTitle}_sources_sinks.ply"
    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_sources_sinks_{count}.ply"
        count += 1
    return newFile

def sources_sinks(file, mesh, cut):
    if not os.path.exists(file):
        print(f"{file} - file does not exist")

    file2 = getNewFileName7(file)
    with open(file2, 'w') as newFile:    
        print(f"{file2} contains colored sources and sinks")

        # write header for even box
        newFile.write("ply\n")
        newFile.write("format ascii 1.0\n")

        # define vertex element with properties - x, y and z
        vert_count = len(mesh.vertices)
        newFile.write(f"element vertex {vert_count}\n")
        newFile.write("property float x\nproperty float y\nproperty float z\n")

        numTriangles = len(mesh.triangles)
        newFile.write(f"element face {numTriangles}\n")
        newFile.write("property list uchar int vertex_index\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        newFile.write("end_header\n")

        for vertex in mesh.vertices:
            newFile.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        blue = "0 0 255\n"
        green = "0 255 0\n"
        for index in range(numTriangles):
            triangle = mesh.triangles[index]
            v0 = triangle[0]
            v1 = triangle[1]
            v2 = triangle[2]
            if cut[index]:
                newFile.write(f"3 {v0} {v1} {v2} {blue}")
            else:
                newFile.write(f"3 {v0} {v1} {v2} {green}")

########################################################################################

def getNewFileName_even_odd_v2(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]
    newFile = f"{fileTitle}_even_odd.ply"
    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_even_odd_{count}.ply"
        count += 1
    return newFile

def even_odd_v2(file, mesh):
    if not os.path.exists(file):
        print(f"{file} - file does not exist")

    file2 = getNewFileName_even_odd_v2(file)
    with open(file2, 'w') as newFile:
        print(f"{file2} contains even_odd_mesh_v2")

        # write header for even box
        newFile.write("ply\n")
        newFile.write("format ascii 1.0\n")

        vert_count = len(mesh.vertices)
        newFile.write(f"element vertex {vert_count}\n")
        newFile.write("property float x\nproperty float y\nproperty float z\n")
        
        face_count = len(mesh.even_box) + len(mesh.odd_box)
        newFile.write(f"element face {face_count}\n")
        newFile.write("property list uchar int vertex_index\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        newFile.write("end_header\n")

        for vertex in mesh.vertices:
            addVertexToFile(newFile, vertex)

        red = "255 0 0"
        for even_face_index in mesh.even_box:
            t = mesh.triangles[even_face_index]
            newFile.write(f"3 {t[0]} {t[1]} {t[2]} {red}\n")

        blue = "0 255 0"   
        for odd_face_index in mesh.odd_box:
            t = mesh.triangles[odd_face_index]
            newFile.write(f"3 {t[0]} {t[1]} {t[2]} {blue}\n")

########################################################################################
    
def getNewFileName_v6(file):
    split_name = os.path.splitext(file)
    fileTitle = split_name[0]
    newFile = f"{fileTitle}.ply"
    count = 1
    while os.path.exists(newFile):
        newFile = f"{fileTitle}_{count}.ply"
        count += 1
    return newFile
    
def readSimplices(tree):
    vert = []
    edges = []
    faces = []
    simplices = tree.get_simplices()
    for simp in simplices:
        (simplex, filtration) = simp
        if filtration == 1.0:
            if len(simplex) == 1:
                vert.append(simplex)
            elif len(simplex) == 2:
                edges.append(simplex)
            elif len(simplex) == 3:
                faces.append(simplex)
    
    return (vert, edges, faces)

'''
This function writes a simplex tree at to a ply file
file    : The file name
tree    : The simplex tree
vertices: Array of vertices = (x, y, z) coordinates
'''
def write_simplex_tree_to_ply(file, tree, vertices, edge_color = None, face_color = None):

    (verts, edges, faces) = readSimplices(tree)
    counter = 0
    vertMap = {}
    for vert in verts:
        vertMap[vert[0]] = counter
        counter += 1

    file2 = getNewFileName_v6(file)

    # if not os.path.exists(file):
    #     print(f"{file} - file does not exist")

    with open(file2, 'w') as newFile:
        newFile.write("ply\n")
        newFile.write("format ascii 1.0\n")

        newFile.write(f"element vertex {len(verts)}\n")
        newFile.write("property float x\nproperty float y\nproperty float z\n")

        newFile.write(f"element edge {len(edges)}\n")
        newFile.write("property int vertex1\nproperty int vertex2\n")

        newFile.write(f"element face {len(faces)}\n")
        newFile.write("property list uchar int vertex_index\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        newFile.write("end_header\n")

        for vert in verts:
            addVertexToFile(newFile, vertices[vert[0]])

        for edge in edges:
            if edge_color is None:
                newFile.write(f"{vertMap[edge[0]]} {vertMap[edge[1]]}\n")
            else:
                newFile.write(f"3 {vertMap[edge[0]]} {vertMap[edge[1]]} {vertMap[edge[1]]} {edge_color}\n")

        if face_color is None:
            gray = " 128 128 128\n"
            face_color = gray
        for face in faces:
            newFile.write(f"3 {vertMap[face[0]]} {vertMap[face[1]]} {vertMap[face[2]]} {face_color}")

    return