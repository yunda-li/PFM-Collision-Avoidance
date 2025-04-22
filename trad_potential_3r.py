"""
Traditional PFM formulation for 3 Robots

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from geometry import Sphere, RepulsiveSphere, Attractive

x_start_coord = np.array([[-8, -4, 9], 
                         [6, -4, -5]])

x_goal_coord = np.array([[4, 7, -3],
                         [-6, 7, 7]])

border = Sphere(center = np.zeros((2,1)), radius = -15, distance_influence=1)
robot1 = Sphere(center = x_start_coord[:,0].reshape(-1,1), radius = 2, distance_influence=5)
robot2 = Sphere(center = x_start_coord[:,1].reshape(-1,1), radius = 2, distance_influence=5)
robot3 = Sphere(center = x_start_coord[:,2].reshape(-1,1), radius = 2, distance_influence=5)


world_list = [border]

class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self, x_start_list, x_goal_list):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different goal locations (one
    for each column).
        """

        self.world = world_list
        self.x_goal = x_goal_list
        self.x_start = x_start_list

    def plot(self, axes=None):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a  * marker at the goal
        location.
        """

        if axes is None:
            axes = plt.gca()

        for sphere in self.world:
            sphere.plot('r', axes)


        #Plot stars at goals
        axes.scatter(self.x_goal[0, 0], self.x_goal[1, 0], c='b', marker='*')
        axes.scatter(self.x_goal[0, 1], self.x_goal[1, 1], c='r', marker='*')
        axes.scatter(self.x_goal[0, 2], self.x_goal[1, 2], c='g', marker='*')

        axes.set_xlim([-11, 11])
        axes.set_ylim([-11, 11])
        axes.axis('equal')

class Total: 
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential, other_robot1 = None, other_robot2 = None):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential
        self.other_robot1 = other_robot1    #Sphere object
        self.other_robot2 = other_robot2    #Sphere object

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """
        x_eval = x_eval.reshape(-1,1)

        #Create Attractive instance
        attractive_potential = Attractive(self.potential)
        weight = self.potential['repulsive_weight']

        u_attr = attractive_potential.eval(x_eval)

        u_rep_total = 0

        #world is list of spheres. Create a new sphere instance for each sphere in list
        for sphere in self.world.world:
            rep_sphere = RepulsiveSphere(sphere)

            u_rep = rep_sphere.eval(x_eval)

            u_rep_total += u_rep

        #Add effect of other robot, old RepulsiveSphere
        if self.other_robot1 is not None:
            robot_obstacle1 = RepulsiveSphere(self.other_robot1)
            u_rep_robot = robot_obstacle1.eval(x_eval)
            u_rep_total += u_rep_robot

        if self.other_robot2 is not None:
            robot_obstacle2 = RepulsiveSphere(self.other_robot2)
            u_rep_robot = robot_obstacle2.eval(x_eval)
            u_rep_total += u_rep_robot


        u_eval = u_attr + weight * u_rep_total

        return u_eval

    def grad(self, x_eval):
        """
        Compute the gradient of the total potential,  U=U_ attr+a*U_rep,i, where a is given by
        the variable  potential.repulsiveWeight
        """

        x_eval = x_eval.reshape(-1,1)

        #Create Attractive instance
        attractive_potential = Attractive(self.potential)
        weight = self.potential['repulsive_weight']

        u_attr_grad = attractive_potential.grad(x_eval)

        u_rep_total_grad = np.zeros((2,1), dtype=np.float64)

        #world is list of spheres. Create a new sphere instance for each sphere in list
        for sphere in self.world.world:
            rep_sphere = RepulsiveSphere(sphere)
            u_rep_grad = rep_sphere.grad(x_eval)
            u_rep_total_grad += u_rep_grad.reshape(-1,1)

        #Add effect of other robot
        if self.other_robot1 is not None:
            robot_obstacle1 = RepulsiveSphere(self.other_robot1)

            u_rep_grad_robot = robot_obstacle1.grad(x_eval)

            u_rep_total_grad += u_rep_grad_robot.reshape(-1,1)

        if self.other_robot2 is not None:
            robot_obstacle2 = RepulsiveSphere(self.other_robot2)

            u_rep_grad_robot = robot_obstacle2.grad(x_eval)

            u_rep_total_grad += u_rep_grad_robot.reshape(-1,1)

        grad_u_eval = u_attr_grad + (weight * u_rep_total_grad)

        return grad_u_eval

class Planner:
    """
    A class implementing a generic potential planner and plot the results.
    """
    def __init__(self, function, control, epsilon, nb_steps):
        """
        Save the arguments to internal attributes
        """
        self.function = function  #value of potential fn
        self.control = control    #negative gradient of potential function
        self.epsilon = epsilon
        self.nb_steps = nb_steps

    def run(self, x_start):
        """
        This function uses a given function (given by  control) to implement a
        generic potential-based planner with step size  epsilon, and evaluates
        the cost along the returned path. The planner must stop when either the
        number of steps given by  nb_stepsis reached, or when the norm of the
        vector given by  control is less than 5 10^-3 (equivalently,  5e-3).
        """

        x_start = x_start.reshape(-1,1)

        step_min = 5e-3

        x_path = np.nan * np.ones((2,self.nb_steps))
        u_path = np.nan * np.ones((1,self.nb_steps))

        x_path[:,0] = x_start.flatten()
        u_path[:,0] = self.function(x_start)

        for i in range(1, x_path.shape[1]):
            prev_x_path = x_path[:,i-1]
            prev_x_path = prev_x_path.reshape(-1,1)

            grad = self.control(prev_x_path)

            current_x_path = prev_x_path - self.epsilon * grad
            step_size = np.linalg.norm(grad)

            if step_size > step_min:
                x_path[:,i] = current_x_path.flatten()
                u_path[:,i] = self.function(current_x_path)
            else:
                break

            #For testing purposes, stops path planning if start to ascend gradient
            # if u_path[:,i] > u_path[:,i-1]:
                # break

        return x_path, u_path
    
def run_alternating_steps_3R(plan1, plan2, plan3, x_start1, x_start2, x_start3, nb_steps, axes = None): 
    '''
    This function runs three planners in an alternating fashion. Each planner takes a step, 
    and the robot's new position is considered in the other robot's planning on the next step. 
    This approach helps in avoiding collisions by dynamically adjusting each robot's 
    path based on the other's current position.

    Parameters:
    - plan1 (Planner): The first planner object for robot1.
    - plan2 (Planner): The second planner object for robot2.
    - plan3 (Planner): The third planner object for robot3.
    - x_start1 (numpy.ndarray): The starting position for robot1.
    - x_start2 (numpy.ndarray): The starting position for robot2.
    - x_start3 (numpy.ndarray): The starting position for robot3.
    - nb_steps (int): The number of steps to run for each planner.
    - axes: Matplotlib axes object for plotting. 
      If None, the current axes are used.

    Returns:
    - A tuple containing four numpy arrays (x_path1, x_path2, x_path3), 
    representing the paths for all 3 robots.

    '''
    if axes is None:
        axes = plt.gca()

    x_start1 = x_start1.reshape(-1,1)
    x_start2 = x_start2.reshape(-1,1)
    x_start3 = x_start3.reshape(-1,1)

    step_min = 5e-3

    # Initialize paths for 3 robots
    x_path1 = np.nan * np.ones((2, nb_steps))
    u_path1 = np.nan * np.ones((1, nb_steps))
    x_path2 = np.nan * np.ones((2, nb_steps))
    u_path2 = np.nan * np.ones((1, nb_steps))
    x_path3 = np.nan * np.ones((2, nb_steps))
    u_path3 = np.nan * np.ones((1, nb_steps))


    x_path1[:,0] = x_start1.flatten()
    u_path1[:,0] = plan1.function(x_start1)
    x_path2[:,0] = x_start2.flatten()
    u_path2[:,0] = plan2.function(x_start2)
    x_path3[:,0] = x_start3.flatten()
    u_path3[:,0] = plan3.function(x_start3)

    for i in range(1, nb_steps):
        # Update path 1
        prev_x_path1 = x_path1[:,i-1].reshape(-1,1)
        grad1 = plan1.control(prev_x_path1)
        current_x_path1 = prev_x_path1 - plan1.epsilon * grad1
        step_size1 = np.linalg.norm(grad1)

        if step_size1 > step_min:
            x_path1[:,i] = current_x_path1.flatten()
            u_path1[:,i] = plan1.function(current_x_path1)

            #Update Robot1 RepulsiveSphere Position
            robot1.update_center(current_x_path1)


        else:
            break

        # Update path 2
        prev_x_path2 = x_path2[:,i-1].reshape(-1,1)
        grad2 = plan2.control(prev_x_path2)
        current_x_path2 = prev_x_path2 - plan2.epsilon * grad2
        step_size2 = np.linalg.norm(grad2)

        if step_size2 > step_min:
            x_path2[:,i] = current_x_path2.flatten()
            u_path2[:,i] = plan2.function(current_x_path2)

            #Update Robot2 RepulsiveSphere Position
            robot2.update_center(current_x_path2)


        # Update path 3
        prev_x_path3 = x_path3[:,i-1].reshape(-1,1)
        grad3 = plan3.control(prev_x_path3)
        current_x_path3 = prev_x_path3 - plan3.epsilon * grad3
        step_size3 = np.linalg.norm(grad3)

        if step_size3 > step_min:
            x_path3[:,i] = current_x_path3.flatten()
            u_path3[:,i] = plan3.function(current_x_path3)

            #Update Robot3 RepulsiveSphere Position
            robot3.update_center(current_x_path3)

        else:
            break

    return x_path1, x_path2, x_path3

def test_planner_3_robots(steps, rep_weight1, rep_weight2, rep_weight3):
    '''
        This function sets up a simulation environment with three robots and visualizes their movement as 
        they navigate towards their respective goals while avoiding collisions with each other. 
        The paths of the robots are plotted in the environment.

        Parameters:
        - steps (int): The number of steps to simulate.
    '''
    world = SphereWorld(x_start_coord, x_goal_coord)
    epsilon = 1e-3 #Step size
    nb_steps = steps #2000

    # Start and goal positions
    x_start1 = world.x_start[:, 0].reshape(-1, 1)
    x_start2 = world.x_start[:, 1].reshape(-1, 1)
    x_start3 = world.x_start[:, 2].reshape(-1, 1)
    x_goal1 = world.x_goal[:, 0].reshape(-1, 1)
    x_goal2 = world.x_goal[:, 1].reshape(-1, 1)
    x_goal3 = world.x_goal[:, 2].reshape(-1, 1)

    # Potential order and repulsive weight for the planners
    potential1 = {
        "x_goal": x_goal1,
        "repulsive_weight": rep_weight1, #750 Does the thing
        "shape": "quadratic",
    }
    potential2 = {
        "x_goal": x_goal2,
        "repulsive_weight": rep_weight2, #1
        "shape": "quadratic",
    }
    potential3 = {
        "x_goal": x_goal3,
        "repulsive_weight": rep_weight3, #500
        "shape": "quadratic",
    }

    _, ax = plt.subplots(figsize = (10,8))
    world.plot(axes=ax)  

    total1 = Total(world, potential1, robot2, robot3)
    total2 = Total(world, potential2, robot1, robot3)
    total3 = Total(world, potential3, robot1, robot2)

    plan1 = Planner(total1.eval, total1.grad, epsilon, nb_steps)
    plan2 = Planner(total2.eval, total2.grad, epsilon, nb_steps)
    plan3 = Planner(total3.eval, total3.grad, epsilon, nb_steps)

    # Run the planner
    x_path1, x_path2, x_path3 = run_alternating_steps_3R(plan1, plan2, plan3, x_start1, x_start2, x_start3, nb_steps, ax)

    #Plot Circles along path
    moving_sphere1 = plt.Circle((x_start1[0], x_start1[1]), robot1.radius, color='b', fill=False)
    moving_sphere2 = plt.Circle((x_start2[0], x_start2[1]), robot2.radius, color='r', fill=False)
    moving_sphere3 = plt.Circle((x_start3[0], x_start3[1]), robot3.radius, color='g', fill=False)

    tracking_sphere1 = plt.Circle((x_start1[0], x_start1[1]), robot1.radius, color='c', fill=False)
    tracking_sphere2 = plt.Circle((x_start2[0], x_start2[1]), robot2.radius, color='m', fill=False)
    tracking_sphere3 = plt.Circle((x_start2[0], x_start2[1]), robot3.radius, color='y', fill=False)

    # Add circles to the plot
    ax.add_patch(moving_sphere1)
    ax.add_patch(moving_sphere2)
    ax.add_patch(moving_sphere3)

    ax.add_patch(tracking_sphere1)
    ax.add_patch(tracking_sphere2)
    ax.add_patch(tracking_sphere3)

    # Loop through each step and update the plot
    for i in range(nb_steps):
        world.plot(axes=ax)  

        # Path 1 in blue
        if i < x_path1.shape[1]:
            ax.plot(x_path1[0, :i+1], x_path1[1, :i+1], 'b.-')  # Path 1 in blue
            moving_sphere1.center = (x_path1[0, i], x_path1[1, i])
            ax.add_patch(moving_sphere1)

        #Path 2 in red
        if i < x_path2.shape[1]:
            ax.plot(x_path2[0, :i+1], x_path2[1, :i+1], 'r.-')  # Path 2 in red
            moving_sphere2.center = (x_path2[0, i], x_path2[1, i])
            ax.add_patch(moving_sphere2)

        #Path 3 in green
        if i < x_path3.shape[1]:
            ax.plot(x_path3[0, :i+1], x_path3[1, :i+1], 'g.-')  
            moving_sphere3.center = (x_path3[0, i], x_path3[1, i])
            ax.add_patch(moving_sphere3)

        #Circle trail every 50 steps
        if i % 50 == 0:
            print(f'Progress: {i} out of {nb_steps} steps')
            new_tracking_sphere1 = plt.Circle((x_path1[0, i], x_path1[1, i]), robot1.radius, color='c', fill=False)
            new_tracking_sphere2 = plt.Circle((x_path2[0, i], x_path2[1, i]), robot2.radius, color='m', fill=False)
            new_tracking_sphere3 = plt.Circle((x_path3[0, i], x_path3[1, i]), robot3.radius, color='y', fill=False)

            ax.add_patch(new_tracking_sphere1)
            ax.add_patch(new_tracking_sphere2)
            ax.add_patch(new_tracking_sphere3)


        # Check for collisions
        distance1 = np.linalg.norm(np.array(moving_sphere1.center) - np.array(moving_sphere2.center))
        distance2 = np.linalg.norm(np.array(moving_sphere1.center) - np.array(moving_sphere3.center))
        distance3 = np.linalg.norm(np.array(moving_sphere2.center) - np.array(moving_sphere3.center))


        if distance1 <= (robot1.radius + robot2.radius): 
            print("Collision detected at step:", i)
            break 
            
        elif distance2 <= (robot1.radius + robot3.radius):
            print("Collision detected at step:", i)
            break 
            
        elif distance3 <= (robot2.radius + robot3.radius):
            print("Collision detected at step:", i)
            break 

        #Plot details
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Robot 1', markerfacecolor='blue',
                markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='blue'),
        Line2D([0], [0], marker='o', color='w', label='Robot 2', markerfacecolor='red', 
               markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='red'),
        Line2D([0], [0], marker='o', color='w', label='Robot 3', markerfacecolor='green', 
               markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='green'),
        Line2D([0], [0], marker='o', color='w', label='Previous Robot 1 Positions', 
               markerfacecolor='cyan', markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='cyan'),
        Line2D([0], [0], marker='o', color='w', label='Previous Robot 2 Positions', 
               markerfacecolor='magenta', markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='magenta'),
        Line2D([0], [0], marker='o', color='w', label='Previous Robot 3 Positions', 
               markerfacecolor='yellow', markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='yellow'),
        Line2D([0], [0], marker='*', color='w', label='Robot 1 Goal', markerfacecolor='blue', markersize=15),
        Line2D([0], [0], marker='*', color='w', label='Robot 2 Goal', markerfacecolor='red', markersize=15),
        Line2D([0], [0], marker='*', color='w', label='Robot 3 Goal', markerfacecolor='green', markersize=15)
    ]

    ax.set_xlim([-60, 70])
    ax.legend(handles=legend_elements, loc = 'lower right')
    ax.set_title(f'Traditional PFM Formulation \n Number of Steps: {nb_steps}')


    plt.show()

if __name__ == '__main__':
    test_planner_3_robots(1000, rep_weight1=750, rep_weight2 = 1, rep_weight3=500)
