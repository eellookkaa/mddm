from shapely.affinity import translate
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
def find_reference_point(vertices):
    if not vertices:
        return None, None  # No vertices provided

    reference_point = vertices[0]  # Initialize with the first vertex
    reference_point_index = 0


    for index, vertex in enumerate(vertices):
        if vertex[1] > reference_point[1] or (vertex[1] == reference_point[1] and vertex[0] < reference_point[0]):
            reference_point = vertex
            reference_point_index = index

    return reference_point, reference_point_index

def translate_polygon_to_origin(polygon_vertices):
    # Find the reference point (top-left vertex) of the polygon
    reference_point, position = find_reference_point(polygon_vertices)

    # Translate the polygon to the origin (0,0) by subtracting the reference point's coordinates
    translated_polygon = [(vertex[0] - reference_point[0], vertex[1] - reference_point[1]) for vertex in polygon_vertices]

    return translated_polygon



def move_polygon_to_target_coordinate(polygon, target_coordinate):
    reference_point,reference_point_index = find_reference_point(polygon.exterior.coords)
    # Get the current reference point
    current_reference_point = polygon.exterior.coords[reference_point_index]

    # Calculate the translation vector based on the difference between the current and target coordinates
    translation_vector = (target_coordinate[0] - current_reference_point[0], target_coordinate[1] - current_reference_point[1])

    # Check if the target and real coordinates are the same
    if translation_vector == (0, 0):
        # No need to move the polygon
        return polygon

    # Translate the polygon using the calculated translation vector
    moved_polygon = translate(polygon, xoff=translation_vector[0], yoff=translation_vector[1])

    return moved_polygon


def getPoints(polygon):
    # Find the bounds of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Create a grid within the bounds of the polygon
    grid_spacing = 1  # Adjust this value as needed
    x_grid = np.arange(min_x, max_x, grid_spacing)
    y_grid = np.arange(min_y, max_y, grid_spacing)

    # Create an empty list to store points inside the polygon
    points_inside = []

    # Iterate through the grid points and check if they are inside the polygon
    for x in x_grid:
        for y in y_grid:
            point = (x, y)
            if polygon.contains(Point(point)):
                points_inside.append(point)
    return points_inside 



def plot_polygons(polygons, placement):
    fig, ax = plt.subplots()

    for i, (x, y) in placement:
        poly = polygons[i]
        x_coords, y_coords = poly.exterior.xy
        ax.fill(x_coords, y_coords, edgecolor='black', facecolor='none')
        ax.text(x, y, f'Polygon {i}', ha='center', va='center')

    ax.set_aspect('equal', 'box')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.gca().invert_yaxis()  # invert y-axis to match the coordinate system

    plt.show()