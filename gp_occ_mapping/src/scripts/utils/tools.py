import numpy as np

def is_in_poly(x, y, polygon):
    """
    whether one point is inside a 2-D poly
    :param x: point_x
    :param y: point_y
    :param polygon: array of polygon vertex：k * 2
    :return: true or false
    """
    poly_xs = polygon[:, 0]
    poly_ys = polygon[:, 1]
    odd_cross = False
    for i in range(len(poly_xs)):
        j = i - 1 if i > 0 else len(poly_xs) - 1
        if poly_ys[i] < y <= poly_ys[j] or poly_ys[j] < y <= poly_ys[i]:
            if poly_xs[i] + (y - poly_ys[i]) * (poly_xs[j] - poly_xs[i]) / (poly_ys[j] - poly_ys[i]) < x:  # wont divide 0
                odd_cross = False if odd_cross else True
    return odd_cross


def graph_in_poly(graph, polygon):
    """
    setting pixel to one if the pixel in polygon
    :param graph: a 2-D graph: m * m arraty
    :param polygon: array of polygon vertex：k * 2
    :return: new 2-D array
    """
    
    new_graph = graph
    
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if is_in_poly(i,j, polygon):
                new_graph[i, j] = 1
    
    return new_graph

