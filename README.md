# PFM-Collision-Avoidance
Simulation experiment for preventing robot collision using potential field methods, by adjusting force for each robot

Run the main_results.py to get the plots. They will open in the following order:

    -2 Robots with traditional PFM (trad_potential.py)
    -3 Robots with traditional PFM (trad_potential_3r.py)
    -2 Robots with traditional PFM (vel_potential.py)
    -3 Robots with traditional PFM (vel_potential_3r.py)

As set currently, main_results.py will return the plots used in the paper. To see the plots of the 
collisions, set all rep_weights to 1. 

Feel free to experiment with setting different repulsive weights, the planner will stop if a collision
happens. Try to keep rep_weight2 the same value, since many of these situations depend on robot2 reaching
the center of environment first.

With many steps, the planner can take a while to plot. The terminal will print the progress of the plotting.
Most collision avoidance can be seen at around 1000 steps.
