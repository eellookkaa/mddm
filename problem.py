import pulp
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
import numpy as np
import shapely.affinity as affine
from minkowskisum import minkowskisum
from IFP import getIFP
import pulp as pl
from my_utils import find_reference_point, translate_polygon_to_origin, move_polygon_to_target_coordinate, getPoints, plot_polygons
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from shapely.affinity import translate
from shapely.ops import unary_union
import math
import mosek
import pandas as pd
def translate_polygons_left_bottom_to_origin(polygons):
  translated_polygons = []
  for polygon in polygons:
    left_bottom_point = polygon.bounds[0:2]
    translated_polygon = affine.translate(polygon, xoff=-left_bottom_point[0], yoff=-left_bottom_point[1])
    translated_polygons.append(translated_polygon)
  return translated_polygons


def problem(polygons, W, L, minimize_total, warmStart):

    problem = pulp.LpProblem("PolygonPlacement", pulp.LpMinimize)
    C = L + 1
    R = W + 1
    D = C * R
    N = len(polygons)
    if minimize_total == False:
        z = pulp.LpVariable('z', lowBound=10, upBound=20)
    else:
        z_length = pulp.LpVariable("TotalLength", lowBound=3, upBound=L)
        z_width = pulp.LpVariable("TotalWidth", lowBound=3, upBound=W)
        z = z_length + z_width

    problem.setObjective(z)

    IFP = {}
    for i, poly in enumerate(polygons):
        exterior_coords = list(poly.exterior.coords)
        int_exterior_coords = [(int(coord[0]), int(coord[1])) for coord in exterior_coords]
        ifp = getIFP(W, L, int_exterior_coords)
        IFP[i] = []
        for coord in ifp:
                d = coord[0] * R + coord[1] + 1
                IFP[i].append(d)
    
    delta = {}

    for i, poly in enumerate(polygons):
        for d in range(1, D+1):
        # for d in IFP[i]:
            delta[(i, d)] = pulp.LpVariable(f'delta_{i}_{d}', cat=pulp.LpBinary)
    
    if minimize_total == False: 
        for i, poly in enumerate(polygons):
            for x in range(C):
                for y in range(R):
                        d = x * R + y + 1
                        if d in IFP[i]:
                            pol = move_polygon_to_target_coordinate(poly, (x,y))
                            problem += delta[(i, d)]*(pol.bounds[2]) + delta[(i, d)]*(pol.bounds[3])<= z

    else: 
        for i, poly in enumerate(polygons):
            for x in range(C):
                for y in range(R):
                        d = x * R + y + 1
                        if d in IFP[i]:
                            pol = move_polygon_to_target_coordinate(poly, (x,y))
                            problem += delta[(i, d)]*(pol.bounds[2]) <= z_length
        
        for i, poly in enumerate(polygons):
            for x in range(C):
                for y in range(R):
                        d = x * R + y + 1
                        if d in IFP[i]:
                            pol = move_polygon_to_target_coordinate(poly, (x,y))
                            problem += delta[(i, d)]*(pol.bounds[3]) <= z_width
                        

   

    for i, poly in enumerate(polygons):
        problem += pulp.lpSum(delta[(i, d)] for d in IFP[i]) == 1

    NFP = {}
    for i in range(N):
        for j in range(N):
            if i!=j:
                for x in range(C):
                    for y in range(R):
                        d = x * R + y + 1
                        if d in IFP[i]:
                            polygon1 = move_polygon_to_target_coordinate(polygons[i], (x,y))
                            polygon2 = translate(polygons[j], xoff=0, yoff=0)
                            nfp = minkowskisum(np.array(polygon1.exterior.coords), np.array(polygon2.exterior.coords), var=0)
                            
                            polygon2 = move_polygon_to_target_coordinate(polygons[j], (0,0))
                            nfp1 = minkowskisum(np.array(polygon1.exterior.coords), np.array(polygon2.exterior.coords), var=1)                       
                            polygon = unary_union([Polygon(nfp), (Polygon(nfp1)), polygon1])

                            interior_coords = getPoints(polygon)
                            NFP[(i,j,d)]= []
                            for coorpoint in interior_coords:
                                    if coorpoint[0]>=0 and coorpoint[1]>=0 and coorpoint[0]<C and coorpoint[1]<R:
                                        dot = coorpoint[0] * R + coorpoint[1] + 1
                                        NFP[(i,j,d)].append(dot) 

    for i in range(N):
        for j in range(N):
            if i!=j:
                for d in IFP[i]:
                        for e in NFP[(i,j,d)]:
                                problem += delta[(i, d)] + delta[(j, e)] <= 1

    if warmStart == True:
        #initial = [(0,154), (1,108), (2,12), (3,134), (4,23), (5,63), (6,15), (7,127)]
        initial = [(0,154), (1,108), (2,12), (3,134), (4,23)]

        for i in initial:
            delta[(i[0], i[1])].setInitialValue(1)
    
    solverCBC = pl.getSolver('PULP_CBC_CMD',
        # keepFiles = 1,
        # mip = 1,
        msg = True,
        # cuts = 'on',
        # presolve = True,
        # strong = 0,
        # options=['perturbation off', 'cuts root', 'presolve more', 'greedyHeuristic on'],
        # gapRel = 0,
        # maxSeconds = None,
        # heur = 'on',
        # timeLimit = 1000,
        warmStart = True, 
        # heuristics = 'on',
     
        )
  
    
    solverCOIN = pl.getSolver('COIN_CMD', 
        keepFiles = 1,
        mip = 1,
        msg = 1,
        # cuts = 'forceOn',
        # presolve = 'on',
        # strong = 0,
        # options=['perturbation off', 'cuts ifmove', 'presolve off'],
        # gapRel = 0.9,
        # gapAbs = 0.08,
        # maxSeconds = None,
        # timeLimit = 100,
        warmStart = True
         )
    
    solverGLPK= pl.getSolver('GLPK_CMD', 
        mip = 1,
        msg = 1,
       
        # timeLimit = 1000,
        options = ["--mipgap", "0.01"]
        )
    

   
    solverMOSEK =pl.getSolver('MOSEK', 
                        mip = True,
                        msg = True,
                        # keepFiles = 1,
                        # optimizer = 'free',
                        # presolve_use ='on',
                        # mio_heuristic_level = 3
                        # options = {
                                 
                        #         #    mosek.iparam.mio_cut_selection_level: 0,
                        #         # mosek.iparam.mio_data_permutation_method: mosek.miodatapermmethod.cyclic_shift,
                        #         #  mosek.iparam.presolve_use: mosek.presolvemode.on,
                        #              mosek.iparam.mio_heuristic_level: 3,
                        #             # mosek.iparam.sim_hotstart: mosek.simhotstart.status_keys,

                        #            }
)
    
    solverGUROBI= pl.getSolver('GUROBI', 
        mip = 1,
        msg = 1,
        keepFiles = 1,
        #NormAdjust=2
        warmStart = False,
        LogFile='log.txt'
        # Cuts=2,
        # PerturbValue = 5,
        # Presolve = 0,
        # options=[('Presolve', 0)]
        )
   
    solver = problem.solve(solverGUROBI)
  
    print(f'time = {problem.solutionTime}')
    print(f'status = {pulp.LpStatus[problem.status]}')
    print(f'objective = {pulp.value(problem.objective)}')
   
    if pulp.LpStatus[problem.status] == "Optimal":

        variable_values = {var.name: var.varValue for var in problem.variables()}
        df = pd.DataFrame(list(variable_values.items()), columns=["Variable", "Value"])
    
        df.to_csv("variable_values.csv", index=False)
    else:
        print("The model is not solved to optimality.")
  
  

    placement = []
    for i in range(N):
        for x in range(C):
            for y in range(R):
                d = x * R + y + 1
                if d in IFP[i]:
                    if pulp.value(delta[(i, d)]) == 1:
                        print('valid: ', delta[(i, d)])
                        placement.append((i, (x,y)))
    print('placement' , placement)
    return  placement, pulp.value(z)



if __name__ == "__main__":
    polygons = [
        Polygon([(0, 0), (4, 1), (3, 4)]),
        Polygon([(0, 0), (4, 0), (2, 3)]),
        Polygon([(0, 0), (0, 5), (4, 5), (4, 0)]),
        # Polygon([(0, 1), (0, 4), (2, 5), (4, 4), (4,1), (2,0)]),
        # Polygon([(0, 2), (0, 4), (1, 5), (4, 5), (4, 0), (2, 0)]),
        # Polygon([(0, 0), (0, 3), (3, 3), (3, 0)]),
        # Polygon([(0, 0), (0, 3), (3, 3), (3, 0)]),
        # Polygon([(0, 2), (0, 4), (1, 5), (4, 5), (4, 0), (2, 0)]),
    ]

    W = 15
    L = 15
    minimize_total = True
    warmStart = False
    
    
    placement, roll_length = problem(polygons, W, L,  minimize_total, warmStart)

    print(f"Minimized roll length: {roll_length}")

    for i in range(len(polygons)):
         a, b = find_reference_point(polygons[i].exterior.coords)
         print('poly:',i, ' ', 'point: ', a)

    fig, ax = plt.subplots()
    new_polys=[]
   
    for i, (x, y) in placement:
        poly = polygons[i]
        new_poly = move_polygon_to_target_coordinate(poly, (x,y))
        new_polys.append(new_poly)
        x, y = new_poly.exterior.xy
        ax.plot(x, y, label=f'Polygon {i}')

    maxx = 0
    maxy = 0
    for poly in new_polys:
        if poly.bounds[2]>maxx:
            maxx = poly.bounds[2]
        if poly.bounds[3]>maxy:
            maxy = poly.bounds[3]
    
    print('area: ', maxx*maxy)
    print('maxx: ', maxx)
    print('maxy: ', maxy)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Polygons Placed on Roll of Material")
    # plt.legend()
    plt.show()




