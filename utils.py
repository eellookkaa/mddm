import numpy as np
from scipy.spatial import ConvexHull

def compute_coefficients(polygon):
    '''
    Function that takes a polygon (2d numpy array)
    and return the coefficients of its edges as list
    NB: the points of the polygon must be in order!
    :param polygons: list of polygons, each polygon is np.array, shape = (n_vertices, 2)
    :param H: float, value of the roll
    :return: a, b, c, dictionaries (n, v): coefficient of the straight line connecting vertex v with the next
    '''
    a_list = []
    b_list = []
    c_list = []

    n_edges = polygon.shape[0]
    for v in range(n_edges):
        next_v = 0 if v == n_edges - 1 else v + 1
        x1, y1 = polygon[v]
        x2, y2 = polygon[next_v]
        # straight line formula on the plane
        a = y2 - y1
        b = -x2 + x1
        c = -x1 * (y2 - y1) + y1 * (x2 - x1)

        # check directional vector
        # special case, triangles
        if n_edges == 3:
            if v == 0:
                x_pol, y_pol = polygon[2]
            elif v == 1:
                x_pol, y_pol = polygon[0]
            else:
                x_pol, y_pol = polygon[1]
        else:
            if v == 0 or v == n_edges - 1: 
                # I can consider vertex number 2
                x_pol, y_pol = polygon[2]
            else:   
                # I can consider vertex 0
                x_pol, y_pol = polygon[0]
            
        if a * x_pol + b * y_pol + c < 0:
            a = -a
            b = -b
            c = -c
        
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    return a_list, b_list, c_list

# Minkowski_sum
# The given code defines a function called Minkowski_sum that takes two numpy 2D arrays, A and B, as input. The arrays represent the vertices of two polygons in a 2D space. The function calculates the Minkowski sum of these polygons and returns the resulting polygon as an array of vertices.

# Here's a breakdown of the code:

# The function Minkowski_sum takes two parameters: A and B, which represent the vertices of the polygons.

# The variable new_vertices is initialized as an empty list. This list will store the vertices of the resulting polygon.

# The code then enters a nested loop. The outer loop iterates over each vertex v1 in array A, and the inner loop iterates over each vertex v2 in array B.

# Inside the nested loop, the sum of v1 and v2 is calculated using the + operator, and the result is appended to the new_vertices list.

# After the nested loop completes, the new_vertices list contains all the vertices of the Minkowski sum polygon.

# The new_vertices list is converted into a numpy array using np.array().

# The ConvexHull function is applied to the new_vertices array to calculate the convex hull of the points. This function returns a ConvexHull object.

# Finally, the function returns the vertices of the convex hull polygon using the points attribute of the ConvexHull object, accessed using polygon.points, and indexing it with polygon.vertices to get the vertices in the correct order.
# 

def Minkowski_sum(A, B):
    # A, B numpy 2d arrays (n_verteces, 2)
    new_vertices = []
    for v1 in A:
        for v2 in B:
            new_vertices.append(v1 + v2)
    
    polygon = ConvexHull(np.array(new_vertices))
    return polygon.points[polygon.vertices]


def compute_no_fit_polygons(polygons):
    
    N = len(polygons)
    no_fit_polygons = {}

    for i in range(N):
        for j in range(i+1, N):
            A = polygons[i]
            B = polygons[j]
            no_fit_polygons[(i, j)] = Minkowski_sum(A, -B)
    
    return no_fit_polygons



    
# polygons = [np.array([[0, 0], [0, 1], [1, 1], [1, 0]]), np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])]
# result = compute_no_fit_polygons(polygons)
# result = np.array(list(result.values()))


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

# def plot_polygons(polygons):
#     fig, ax = plt.subplots()
#     handles = []
#     labels = []
    
#     # Generate a color map for the polygons
#     colors = plt.cm.get_cmap('tab10', len(polygons))
    
#     for i, polygon in enumerate(polygons):
#         # Create a Polygon patch
#         patch = Polygon(polygon, closed=True, facecolor=colors(i))
        
#         # Add the patch to the plot
#         ax.add_patch(patch)
        
#         # Create a legend handle and label for the polygon
#         handle = plt.Line2D([], [], color=colors(i), marker='s', markersize=10, label=f'Polygon {i+1}')
#         handles.append(handle)
#         labels.append(f'Polygon {i+1}')
    
#     # Add the legend to the plot
#     ax.legend(handles, labels)
    
#     # Set the plot limits
#     ax.set_xlim([-2, 2])
#     ax.set_ylim([-2, 2])
    
#     # Set aspect ratio to equal
#     ax.set_aspect('equal')
    
#     # Show the plot
#     plt.show()

# # Example usage
# polygons = [
#     np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
#     np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]), result[0]
# ]

# plot_polygons(polygons)


