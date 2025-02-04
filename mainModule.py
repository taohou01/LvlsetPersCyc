from mesh import Mesh
from dual_mesh import dual
import fileWriter

# These flags must be set/unset to be able to write to ply files
ply_flags = {f'flag{i+1}': True for i in range(7)}
ply_flags['flag2'] = False
ply_flags['flag4'] = False
ply_flags['flag7'] = False

########################################################################
'''
flag1 : Prints all triangles containing critical vertices in red.
flag2 : Prints all triangles containing any critical values in red.

''' 
########################################################################

def runMain(fileName, direction, non_critical = [], twoConnected = False, Manifold = False):

    myMesh = Mesh()
    myMesh.load(fileName, non_critical)
    print("Mesh file loaded.")

    if not twoConnected:
        if not myMesh.isTwoConnected():
            print("Mesh is not two-connected.")
            return

    if not Manifold:
        if not myMesh.isManifold():
            print("Mesh is not a manifold.")
            return
    
    myMesh.compute_function_values(direction)
    myMesh.order_vertices_by_function_values()

    crit_verts = myMesh.findCriticalVertices()
    print(f"Critical vertices are {crit_verts}")

    if ply_flags['flag1']:
        fileWriter.color_mesh_1(fileName, crit_verts)

    if myMesh.checkCompatibility():
        if ply_flags['flag2']:
            fileWriter.color_mesh_2(fileName, myMesh)
        print("Mesh is compatible")
    else:
        if ply_flags['flag3']:
            fileWriter.color_mesh_3(fileName, myMesh)   # output incompatible triangles with blue color
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Mesh is not compatible!!!!!!!!!")
        return
        # refine mesh here when required

    dual_graph = dual(myMesh)
    dual_graph.computeMaxFlow()
    dual_edges = dual_graph.outputCrossEdges()
    mesh_edges = myMesh.outputCrossEdges(dual_edges)

    # This method prints any 'bad' vertices indices to the console..
    # myMesh.verify(dual_graph.cut)

    if ply_flags['flag4']:
        fileWriter.writeEdgesToTextFile(fileName, mesh_edges)

    if ply_flags['flag5']:
        fileWriter.color_mesh_4(fileName, myMesh, mesh_edges)
    
    # fileWriter.even_odd_boxes(fileName, myMesh)
    if ply_flags['flag6']:
        fileWriter.even_odd_v2(fileName, myMesh)
        
    if ply_flags['flag7']:
        fileWriter.sources_sinks(fileName, myMesh, dual_graph.get_cut())

    return myMesh