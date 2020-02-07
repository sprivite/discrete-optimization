from ortools.linear_solver import pywraplp
from tools import *

def set_default(constraint, variables):
    for var in variables:
        constraint.SetCoefficient(var, 0)
    return

def format_result(assignments, customers):

    solution = [0] * len(customers)
    for key, var in assignments.items():
        c, f = key
        if var.solution_value() == 1:
            solution[c] = f

    return solution


def ortools_solve(customers, facilities, thresh=0):

    solver = pywraplp.Solver('SolveIntegerProblem',
        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    #
    # decision variables
    #
    assignments = {} # 0/1 assignment of customer c to facility f
    for c in customers:
        for f in facilities:
            assignments[(c.index, f.index)] = solver.IntVar(0, 1, f'assign{c.index},{f.index}')

    openings = {} # 0/1 open facility f
    for f in facilities:
        openings[f.index] = solver.IntVar(0, 1, f'open{f.index}')

    #
    # constraints
    #
    print('Setting constraints ...')
    variables = list(assignments.values()) + list(openings.values())

    # unique assignment constraint
    for c in customers:
        constraint = solver.Constraint(1, 1)
        set_default(constraint, variables)
        for f in facilities:
            constraint.SetCoefficient(assignments[c.index, f.index], 1)

    # capacity constraints
    for f in facilities:

        constraint = solver.Constraint(-solver.infinity(), 0)
        set_default(constraint, variables)
        constraint.SetCoefficient(openings[f.index], -f.capacity)

        for key, var in assignments.items():
            if key[1] == f.index:
                constraint.SetCoefficient(var, customers[key[0]].demand)

    #
    # objective function
    #
    print('Defining objective ...')
    objective = solver.Objective()
    for f in facilities:
        objective.SetCoefficient(openings[f.index], -f.setup_cost)

    dmatrix = calculate_distance_matrix(customers, facilities)
    for key, var in assignments.items():
        objective.SetCoefficient(var, -dmatrix[key[0], key[1]])

    # GO!
    parameters = pywraplp.MPSolverParameters()
    parameters.kDefaultRelativeMipGap = 100
    print('Solving ...')
    objective.SetMaximization()

    result_status = solver.Solve(parameters)
    print('Result:', result_status)
    print('Optimal objective value = %d' % solver.Objective().Value())

    return format_result(assignments, customers)
