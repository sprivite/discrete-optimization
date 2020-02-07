#
# supplementary tools to HW assignment
#
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from collections import namedtuple
import random

def plot_tour(cities, tour):

    x = [c.x for x in cities]
    y = [c.y for y in cities]

    xtour = [cities[j].x for j in tour] + [cities[tour[0]].x]
    ytour = [cities[j].y for j in tour] + [cities[tour[0]].y]

    plt.scatter(x, y, 'bx')
    plt.plot(xtour, ytour, 'k')

    return plt.gcf()


city = namedtuple('City', ['x', 'y'])
def make_cities(n, seed=123):
    random.seed(seed)
    return [city(random.random(), random.random()) for _ in range(n)]
