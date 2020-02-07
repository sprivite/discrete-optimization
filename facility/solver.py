#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from greedy import *
from ortools_solver import *

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def parse_data(lines):

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    return facilities, customers


def grade(solution, customers, facilities):
    '''
    Compute the objective function for a proposed allocation of customers to
    facilities.
    '''
    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += distance(customer.location, facilities[solution[customer.index]].location)

    return obj


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    facilities, customers = parse_data(lines)
    solutions = []

    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = trivial_solve(customers, facilities)
    obj = grade(solution, customers, facilities)
    solutions.append((solution, obj))

    if 0:#len(customers) > 1000:
        for _ in range(100):
            solution = greedy_solve(customers, facilities)
            obj = grade(solution, customers, facilities)
            solutions.append((solution, obj))
            np.random.shuffle(customers)

    validate(solution, customers, facilities)

    if 1:#len(customers) <= 1000:
        solution = ortools_solve(customers, facilities, -100)
        obj = grade(solution, customers, facilities)
        solutions.append((solution, obj))

    validate(solution, customers, facilities)

    # display results
    print([s[1] for s in solutions])

    # get winner
    solution, obj = sorted(solutions, key=lambda x: x[1])[0]

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
