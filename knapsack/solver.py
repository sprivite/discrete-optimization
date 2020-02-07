#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])



def _solve_it_value_density(items, capacity):
    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order of value density until knapsack
    # is filled
    value = 0
    weight = 0
    taken = [0]*len(items)

    items.sort(key=lambda x: x.value / x.weight)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return taken, value


def _solve_it_inorder(items, capacity):

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return taken, value


def _solve_it_dp(items, capacity):

    # build a lookup table of previous solutions
    table = np.zeros((len(items) + 1, capacity + 1))

    # iteratively solve subproblems
    for k in range(1, len(items) + 1):

        weight, value = items[k-1].weight, items[k-1].value

        # if we do not take this item, we can at least do
        # what we had before; this is a lower bound
        table[k, :] = table[k-1, :]

        if weight > capacity:
            continue

        # once there is enough space for the current item,
        # we get to choose.
        table[k, weight:] = np.maximum(
            table[k-1, weight:],
            value + table[k-1, :(capacity + 1 - weight)]
            )

    # extract optimal value
    value = table[len(items), capacity]

    # construct taken list by parsing table backwards
    taken = [0] * len(items)
    for k in range(len(items)):
        _taken = int( table[len(items) - k, capacity] != table[len(items)-k-1, capacity] )
        capacity -= _taken * items[len(items) - k - 1].weight
        taken[items[len(items) - k - 1].index] = _taken

    return taken, int(value)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # taken1, value1 = _solve_it_inorder(items, capacity)
    # taken2, value2 = _solve_it_value_density(items, capacity)
    taken, value = _solve_it_dp(items, capacity)

    # print(value1, taken1)
    # print(value2, taken2)
    # print(value3, taken3)

    # if value1 > value2:
    #     taken = taken1
    #     value = value1
    # else:
    #     taken = taken2
    #     value = value2

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
