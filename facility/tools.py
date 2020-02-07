import math
import numpy as np

def distance(p1, p2):
    return math.sqrt( (p1.x - p2.x)**2 + (p1.y - p2.y)**2 )


def calculate_distance_matrix(customers, facilities):

    dmatrix = np.zeros((len(customers), len(facilities)))
    for i, c in enumerate(customers):
        for j, f in enumerate(facilities):
            dmatrix[i, j] = distance(c.location, f.location)

    return dmatrix


def validate(solution, customers, facilities):

    for f in facilities:

        demand = sum([c.demand for c in customers if solution[c.index] == f.index])
        print(f.index, demand, f.capacity)
