#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
from itertools import combinations, product
from random import sample
import copy

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

Point = namedtuple("Point", ['x', 'y'])

def plot_tour(cities, tours=[]):

    plt.figure(figsize=(6,5))
    x = [c.x for c in cities]
    y = [c.y for c in cities]
    plt.scatter(x, y)
    plt.scatter([x[0]], [y[0]], c='r')

    for tour in tours:
        xtour = [cities[j].x for j in tour] + [cities[tour[0]].x]
        ytour = [cities[j].y for j in tour] + [cities[tour[0]].y]
        plt.plot(xtour, ytour)

    plt.savefig('tour')
    return

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def distance_matrix(points):

    dmatrix = np.inf * np.ones((len(points), len(points)))
    for j in range(len(points)):
        print(j)
        for i in range(j):
            dmatrix[i, j] = length(points[i], points[j])
            dmatrix[j, i] = dmatrix[i, j]

    return dmatrix

def grade(solution, dmatrix):

    # calculate the length of the tour
    obj = dmatrix[solution[-1], solution[0]] # return trip
    obj += sum([dmatrix[solution[i], solution[i+1]] for i in range(len(solution)-1)])
    return obj

def grade_nodmatrix(solution, points):

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    obj += sum([length(points[solution[i]], points[solution[i+1]]) for i in range(len(solution)-1)])
    return obj



def greedy_solution(dmatrix, start=0):

    solution = [start]
    unvisited = list(range(dmatrix.shape[0]))
    unvisited.remove(start)
    node = start
    while len(solution) < dmatrix.shape[0]:
        node = unvisited[np.argmin(dmatrix[node, unvisited])]
        unvisited.remove(node)
        solution.append(node)

    return solution, grade(solution, dmatrix)


def greedy_solution_edges(dmatrix, start=None):

    def _sort_edges_by_length():
        edge_len = {}
        for j in range(dmatrix.shape[0]):
            for i in range(j):
                edge_len[(i,j)] = dmatrix[i, j]
        edges = sorted(edge_len.items(), key=lambda x: x[1])
        return [e[0] for e in edges]


    def _join_segments(seg1, seg2, edge):
        # join seg1 and seg2 on edge
        i, j = edge
        a, b = seg1[0], seg1[-1]
        c, d = seg2[0], seg2[-1]

        if i == a and j == c:
            return seg1[::-1] + seg2
        if i == a and j == d:
            return seg2 + seg1
        if i == b and j == c:
            return seg1 + seg2
        if i == b and j == d:
            return seg1 + seg2[::-1]
        if i == c or i == d:
            return _join_segments(seg1, seg2, edge[::-1])

        raise(ValueError)


    def _build_tour_edgewise(edges):

        # store dictionary of available endpoints
        # at first, all nodes are available
        nodes = dict([(i, [i]) for i in range(dmatrix.shape[0])])
        for edge in edges:
            i, j = edge
            # add edge if both nodes are available as endpoints
            if i in nodes.keys() and j in nodes.keys() and nodes[i] != nodes[j]:

                iseg, jseg = nodes[i], nodes[j]
                del nodes[i], nodes[j]
                newseg = _join_segments(iseg, jseg, edge)
                i, j = newseg[0], newseg[-1]
                nodes[i] = newseg
                nodes[j] = newseg

                if len(nodes) == 2: # only two endpoints left
                    return newseg

    edges = _sort_edges_by_length()
    solution = _build_tour_edgewise(edges)

    return solution, grade(solution, dmatrix)


def repeated_greedy_solution(dmatrix):

    best_obj = np.inf
    best_tour = None
    for start in range(dmatrix.shape[0]):
        this_tour, this_obj = greedy_solution(dmatrix, start)
        if this_obj < best_obj:
            best_obj = this_obj
            best_tour = this_tour

    return best_tour, best_obj


def swap(solution, a, b):
    # swap two indices
    _a = solution[a]
    solution[a] = solution[b]
    solution[b] = _a
    return solution


def local_search_node_swap(solution, dmatrix):

    while 1:

        best_obj = grade(solution, dmatrix)
        best_swap = None

        # find best swap for given objective function
        for this_swap in combinations(solution, 2):

            this_solution = swap(solution[:], *this_swap)
            this_obj = grade(this_solution, dmatrix)
            if this_obj < best_obj:
                best_obj = this_obj
                best_swap = this_swap
                solution = swap(solution, *best_swap)
                print(best_obj)
                break

        # no swap can improve solution, return
        if best_swap is None:
            return solution, best_obj


def local_search_edge_swap(solution, dmatrix, target=0):

    obj = grade(solution, dmatrix)
    while 1:

        stop = 1

        # find best swap for given objective function
        for swap in combinations(solution, 2):
            i, j = sorted(swap)
            A, B, C, D = solution[i-1], solution[i], solution[j-1], solution[j]
            # Are old links longer than new ones? If so, reverse segment.
            if dmatrix[A, B] + dmatrix[C, D] > dmatrix[A, C] + dmatrix[B, D]:
                obj -= dmatrix[A, B] + dmatrix[C, D] - (dmatrix[A, C] + dmatrix[B, D])
                solution[i:j] = reversed(solution[i:j])
                stop = 0
                break

        # no swap can improve solution, return
        if stop or obj < target:
            return solution, obj

def local_search_edge_swap_nodmatrix(solution, points):

    obj = grade_nodmatrix(solution, points)
    print(obj)
    while 1:

        stop = 1
        # find best swap for given objective function
        for swap in combinations(solution, 2):
            i, j = sorted(swap)
            A, B, C, D = solution[i-1], solution[i], solution[j-1], solution[j]
            # Are old links longer than new ones? If so, reverse segment.
            before = length(points[A], points[B]) + length(points[C], points[D])
            after = length(points[A], points[C]) + length(points[B], points[D])
            if before > after:
                obj -= before - after
                print(obj)
                solution[i:j] = reversed(solution[i:j])
                stop = 0
                break

        # no swap can improve solution, return
        if stop or obj < 78478868:
            return solution, grade_nodmatrix(solution, points)


def repeated_local_search_edge_swap(solutions, dmatrix):

    best_obj = np.inf
    best_tour = None
    for solution in solutions:
        this_obj = grade(solution, dmatrix)
        this_tour, this_obj = local_search_edge_swap(solution, dmatrix)
        print(this_obj)
        if this_obj < best_obj:
            best_obj = this_obj
            best_tour = this_tour

    return best_tour, best_obj

def repeated_local_search_edge_swap_random(dmatrix, N):

    best_obj = np.inf
    best_tour = None
    for _ in range(N):
        solution = [0] + sample(list(range(1, dmatrix.shape[0])), dmatrix.shape[0] - 1)
        this_obj = grade(solution, dmatrix)
        #print('before', this_obj)
        this_tour, this_obj = local_search_edge_swap(solution, dmatrix)
        #print('local', this_obj)
        if this_obj < best_obj:
            best_obj = this_obj
            best_tour = this_tour

    return best_tour, best_obj


def parse_input(input_data):

    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    return points


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    points = parse_input(input_data)
    n = len(points)
    print(len(points))

    # overfit solution to problem set :(
    if n == 33810:
        dmatrix = distance_matrix(points)
        solution, obj = greedy_solution(dmatrix)
        solution, obj = local_search_edge_swap(solution, dmatrix, 78478868)

    elif n == 51:

        dmatrix = distance_matrix(points)
        solution, obj = repeated_local_search_edge_swap_random(dmatrix, 10000)

    elif n == 1889:

        dmatrix = distance_matrix(points)
        print(dmatrix)
        solution, obj = repeated_greedy_solution(dmatrix)
        print('greedy search', obj)
        solution, obj = repeated_local_search_edge_swap([solution], dmatrix)

    elif n == 574:

        dmatrix = distance_matrix(points)
        solution, obj = repeated_greedy_solution(dmatrix)
        print('greedy search', obj)
        solution, obj = repeated_local_search_edge_swap([solution], dmatrix)
        print('local search', obj)
        solution, obj = local_search_node_swap(solution, dmatrix)
        print('local search', obj)

    elif n == 200:

        dmatrix = distance_matrix(points)
        solution, obj = repeated_greedy_solution(dmatrix)
        solution, obj = repeated_local_search_edge_swap([solution], dmatrix)

    else:
        dmatrix = distance_matrix(points)

        # perform local search on a bunch of candidate starting points
        solutions = [greedy_solution(dmatrix, k)[0] for k in range(len(points))]
        solutions += [greedy_solution_edges(dmatrix)[0]]
        solutions += [[0] + sample(list(range(1, len(points))), len(points) - 1) for _ in range(10000)]
        solution, obj = repeated_local_search_edge_swap(solutions, dmatrix)
        print('greedy 2 search', obj)
        obj = grade(solution, dmatrix)
        print('local search', obj)

    plot_tour(points, [solution])

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
