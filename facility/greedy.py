import numpy as np
import math
from tools import *


def trivial_solve(customers, facilities):
    solution = [-1]*len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    return solution


def greedy_solve(customers, facilities):
    '''
    Assign customer to  closest facility with available capacity
    '''
    dmatrix = calculate_distance_matrix(customers, facilities)

    capacities = [f.capacity for f in facilities]
    assignments = [0] * len(customers)
    for i, c in enumerate(customers):

        options = np.argsort(dmatrix[i, :])
        for o in options:
            if capacities[o] >= c.demand:
                assignments[c.index] = o
                capacities[o] -= c.demand
                break

    return assignments


def greedy_solve2(customers, facilities):

    dmatrix = calculate_distance_matrix(customers, facilities)



    capacities = [f.capacity for f in facilities]
    assignments = []
    for c in customers:

        options = np.argsort(dmatrix[c.index, :])
        for o in options:
            if capacities[o] >= c.demand:
                assignments.append(o)
                capacities[o] -= c.demand
                break

    return assignments
