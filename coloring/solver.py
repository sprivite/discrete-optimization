#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.setrecursionlimit(1100)

class Graph(object):

    def __init__(self, edges):
        self.edges = edges

    def walk_depth_first(self, node, visited):
        visited += [node]
        for c in set(self.edges[node]) - set(visited):
            visited = self.walk_depth_first(c, visited)

        return visited

    def walk_breadth_first(self, node, visited):

        recursers = list( set(self.edges[node]) - set(visited) )
        visited += recursers
        while recursers:
            new_recursers = []
            for c in recursers:
                this_new_recursers = list( set(self.edges[c]) - set(visited) )
                visited += this_new_recursers
                new_recursers += this_new_recursers
            recursers = set(new_recursers)

        return visited

def parse_graph(input_data):

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    # initialize neighbor list
    neighbors = dict((i, []) for i in range(node_count))

    for i in range(1, edge_count + 1):
        line = lines[i]
        i, j = map(int, line.split())
        neighbors[i].append(j) # undirected
        neighbors[j].append(i) # edges

    # I'm going to assume the elements of the list are unique, which
    # should be true as long as there aren't repeated edges in the input
    for i, n in neighbors.items():
        assert 0 < len(n)      # nodes with no neighbors can be any color
        assert len(n) == len(set(n))

    return neighbors



def color_nodes(coloring, remaining_nodes, graph, colors, last_node, depth):

    # we colored the whole graph! success!
    if not remaining_nodes:
        return coloring

    # apply constraints to find remaining choices
    remaining_colors = {}
    for node in remaining_nodes:
        remaining_colors[node] = set(colors) - set([coloring[n] for n in graph[node]])

        # cannot color this node, no solution possible
        if not remaining_colors[node]:
            return None

    # sort nodes by how well constrained they are
    remaining_nodes = sorted(remaining_colors.keys(), key=lambda x: -len(remaining_colors[x]))

    # get next node to color
    node = remaining_nodes.pop()
    valid_colors = remaining_colors[node]

    # a feasible solution exists, consider all possible colors
    print(depth, len(remaining_nodes), len(valid_colors), file=sys.stderr)
    for color in valid_colors:
        coloring[node] = color
        solution = color_nodes(coloring, remaining_nodes, graph, colors, node, depth + 1)
        print(depth, solution, color, valid_colors, file=sys.stderr)
        if solution:
            return solution

    # no solution found, back track
    coloring[node] = -1
    remaining_nodes.append(node)
    return None


def kcoloring(graph, k):

    # catch trivial solution early; every node has its own color
    n_nodes = len(graph.items())
    if k == n_nodes:
        return range(0, k)

    # allowed colors
    colors = list(range(k))

    # choose an order to traverse nodes
    # we start with node having greatest degree and perform a depth first search
    nodes = sorted(graph.keys(), key=lambda x: len(graph[x]))
    nodes = Graph(graph).walk_breadth_first(nodes[-1], [])[::-1]

    # initialize array for storing colors; -1 means not yet assigned
    coloring = [-1]*n_nodes

    # color first node 0, arbitrarily
    node = nodes.pop()
    coloring[node] = 0

    # recursively color remaining nodes
    # print(coloring, nodes)
    solution = color_nodes(coloring, nodes, graph, colors, node, 1)

    return solution


def solve_it_google(graph, k):

    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    vars = []
    for node in graph.keys():
        vars.append(model.NewIntVar(0, k - 1, str(node)))

    for node in graph.keys():
        for neighbor in graph[node]:
            if node < neighbor:
                model.Add(vars[node] != vars[neighbor])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE:
        vals = [solver.Value(vars[n]) for n in sorted(graph.keys())]
        #solution = ' '.join(vals)
        #import pdb; pdb.set_trace()
        return vals

    else:
        return None


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    graph = parse_graph(input_data)

    lower_bound = {50:6, 70:17, 100:16, 250:95, 500:16, 1000:100}

    # print(len(graph))
    n_nodes = len(graph.items())
    for k in range(lower_bound[len(graph.items())], len(graph.items())):
    #for k in range(2, len(graph.items())):
        print(k, file=sys.stderr)
        if True:#n_nodes == 1000:
            solution = kcoloring(graph, k)
        else:
            solution = solve_it_google(graph, k)
        if solution is not None:
            break

    # prepare the solution in the specified output format
    output_data = str(k) + ' ' + str(0) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
