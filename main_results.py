import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import trad_potential, trad_potential_3r, vel_potential, vel_potential_3r


def main():
    #Change steps to see each simulation run for that many steps.
    steps = 1000
    trad_potential.test_planner_2_robots(steps, rep_weight1=500, rep_weight2=1)
    trad_potential_3r.test_planner_3_robots(steps, rep_weight1=750, rep_weight2=1, rep_weight3=500)
    vel_potential.test_planner_2_robots(steps, rep_weight1=250, rep_weight2=1)
    vel_potential_3r.test_planner_3_robots(steps, rep_weight1=200, rep_weight2=50, rep_weight3=400)

if __name__ == '__main__':
    main()

