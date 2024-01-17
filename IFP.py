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


# Example usage with a list of polygon vertices (x, y)


def translate_polygon_to_origin(polygon_vertices):
    # Find the reference point (top-left vertex) of the polygon
    reference_point, position = find_reference_point(polygon_vertices)

    # Translate the polygon to the origin (0,0) by subtracting the reference point's coordinates
    translated_polygon = [(vertex[0] - reference_point[0], vertex[1] - reference_point[1]) for vertex in polygon_vertices]

    return translated_polygon, position

def translate_polygon_to_positions(dot_board_height,dot_board_width,  polygon_vertices, position):
    
    possible_positions = []

    for x in range(dot_board_width - polygon_vertices[0][0] + 1):

        for y in range(dot_board_height - polygon_vertices[0][1] + 1):
        # for y in range(dot_board_height + 1):
            current_position = [(vertex[0] + x, vertex[1] + y) for vertex in polygon_vertices]
            
            # Check if the translated polygon is fully within the dot board
            if all(0 <= vertex[0] <= dot_board_width and 0 <= vertex[1] <= dot_board_height for vertex in current_position):
                possible_positions.append(current_position[position])

    return possible_positions

# Example usage with a dot board of 7x9 and a polygon's vertices
# dot_board_width = 7
# dot_board_height = 9
# polygon_vertices = [(2, 0), (0, 2), (4, 2), (2,4)]  # Example polygon vertices

# Translate the polygon to (0,0) and calculate the possible positions
def getIFP(dot_board_width, dot_board_height, polygon_vertices):
    translated_polygon, position = translate_polygon_to_origin(polygon_vertices)
    return translate_polygon_to_positions(dot_board_width, dot_board_height, translated_polygon, position)


# for position in possible_positions:
#     for vertex in position:
#         print(vertex)




