"""
Traditional PFM Formulation for 2 Robots

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from geometry import Sphere, RepulsiveSphere, Attractive

x_start_coord = np.array([[-7, -5], 
                         [7, -5]])

x_goal_coord = np.array([[5, 5],
                         [-5, 5]])

border = Sphere(center = np.zeros((2,1)), radius = -15, distance_influence=1)
robot1 = Sphere(center = x_start_coord[:,0].reshape(-1,1), radius = 2, distance_influence=5)
robot2 = Sphere(center = x_start_coord[:,1].reshape(-1,1), radius = 2, distance_influence=5)

world_list = [border]

class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
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
        self.x_goal = x_goal_coord
        self.x_start = x_start_coord


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

        axes.set_xlim([-11, 11])
        axes.set_ylim([-11, 11])
        axes.axis('equal')

class Total: 
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential, other_robot = None):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential
        self.other_robot = other_robot 

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
        if self.other_robot is not None:
            robot_obstacle = RepulsiveSphere(self.other_robot)

            u_rep_robot = robot_obstacle.eval(x_eval)

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
        if self.other_robot is not None:
            robot_obstacle = RepulsiveSphere(self.other_robot)

            u_rep_grad_robot = robot_obstacle.grad(x_eval)

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
    
    
def run_alternating_steps(plan1, plan2, x_start1, x_start2, nb_steps, axes = None):
    '''
    This function runs two planners in an alternating fashion. Each planner takes a step, 
    and the robot's new position is considered in the other robot's planning on the next step. 
    This approach helps in avoiding collisions by dynamically adjusting each robot's 
    path based on the other's current position.

    Parameters:
    - plan1 (Planner): The first planner object for robot1.
    - plan2 (Planner): The second planner object for robot2.
    - x_start1 (numpy.ndarray): The starting position for robot1.
    - x_start2 (numpy.ndarray): The starting position for robot2.
    - nb_steps (int): The number of steps to run for each planner.
    - axes: Matplotlib axes object for plotting. 
      If None, the current axes are used.

    Returns:
    - A tuple containing four numpy arrays (x_path1, u_path1, x_path2, u_path2), 
    representing the paths and control inputs for both robots.

    '''
    if axes is None:
        axes = plt.gca()

    x_start1 = x_start1.reshape(-1,1)
    x_start2 = x_start2.reshape(-1,1)

    step_min = 5e-3

    # Initialize paths for both robots
    x_path1 = np.nan * np.ones((2, nb_steps))
    u_path1 = np.nan * np.ones((1, nb_steps))
    x_path2 = np.nan * np.ones((2, nb_steps))
    u_path2 = np.nan * np.ones((1, nb_steps))


    x_path1[:,0] = x_start1.flatten()
    u_path1[:,0] = plan1.function(x_start1)
    x_path2[:,0] = x_start2.flatten()
    u_path2[:,0] = plan2.function(x_start2)

    for i in range(1, nb_steps):
        # Update path 1
        prev_x_path1 = x_path1[:,i-1].reshape(-1,1)
        grad1 = plan1.control(prev_x_path1)
        current_x_path1 = prev_x_path1 - plan1.epsilon * grad1
        step_size1 = np.linalg.norm(grad1)

        if step_size1 > step_min:
            x_path1[:,i] = current_x_path1.flatten()
            u_path1[:,i] = plan1.function(current_x_path1)

            #Update Robot2 RepulsiveSphere Position
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
            # print(f'New RepSphere2 Center: {robot2.center}')


        else:
            break

    return x_path1, u_path1, x_path2, u_path2

def test_planner_2_robots(steps, rep_weight1, rep_weight2):
    '''
    This function sets up a simulation environment with two robots and visualizes their movement as 
    they navigate towards their respective goals while avoiding collisions with each other. 
    The paths of the robots are plotted in the environment.

    Parameters:
    - steps (int): The number of steps to simulate.
    '''

    world = SphereWorld() 
    epsilon = 1e-3 #Step size
    nb_steps = steps #2000

    # Start and goal positions
    x_start1 = world.x_start[:, 0].reshape(-1, 1)
    x_start2 = world.x_start[:, 1].reshape(-1, 1)
    x_goal1 = world.x_goal[:, 0].reshape(-1, 1)
    x_goal2 = world.x_goal[:, 1].reshape(-1, 1)

    # Potential order and repulsive weight for the planners
    potential1 = {
        "x_goal": x_goal1,
        "repulsive_weight": rep_weight1,
        "shape": "quadratic",
    }
    potential2 = {
        "x_goal": x_goal2,
        "repulsive_weight": rep_weight2,
        "shape": "quadratic",
    }

    _, ax = plt.subplots(figsize = (10,8))
    world.plot(axes=ax)  

    total1 = Total(world, potential1, robot2)
    total2 = Total(world, potential2, robot1)
    plan1 = Planner(total1.eval, total1.grad, epsilon, nb_steps)
    plan2 = Planner(total2.eval, total2.grad, epsilon, nb_steps)

    # Run the planner
    x_path1, _, x_path2, _ = run_alternating_steps(plan1, plan2, x_start1, x_start2, nb_steps, ax)

    #Plot Circles along path
    moving_sphere1 = plt.Circle((x_start1[0], x_start1[1]), robot1.radius, color='b', fill=False)
    moving_sphere2 = plt.Circle((x_start2[0], x_start2[1]), robot2.radius, color='r', fill=False)

    tracking_sphere1 = plt.Circle((x_start1[0], x_start1[1]), robot1.radius, color='c', fill=False)
    tracking_sphere2 = plt.Circle((x_start2[0], x_start2[1]), robot2.radius, color='m', fill=False)

    # Add circles to the plot
    ax.add_patch(moving_sphere1)
    ax.add_patch(moving_sphere2)

    ax.add_patch(tracking_sphere1)
    ax.add_patch(tracking_sphere2)

    for i in range(nb_steps):
        world.plot(axes=ax)

        # Path 1 in blue
        if i < x_path1.shape[1]:
            ax.plot(x_path1[0, :i+1], x_path1[1, :i+1], 'b.-') 
            moving_sphere1.center = (x_path1[0, i], x_path1[1, i])
            ax.add_patch(moving_sphere1)

        # Path 2 in red
        if i < x_path2.shape[1]:
            ax.plot(x_path2[0, :i+1], x_path2[1, :i+1], 'r.-')
            moving_sphere2.center = (x_path2[0, i], x_path2[1, i])
            ax.add_patch(moving_sphere2)

        #Circle trail every 50 steps
        if i % 50 == 0:
            print(f'Progress: {i} out of {nb_steps} steps')
            new_tracking_sphere1 = plt.Circle((x_path1[0, i], x_path1[1, i]), robot1.radius, color='c', fill=False)
            new_tracking_sphere2 = plt.Circle((x_path2[0, i], x_path2[1, i]), robot2.radius, color='m', fill=False)

            ax.add_patch(new_tracking_sphere1)
            ax.add_patch(new_tracking_sphere2)


        # Check for collisions     
        distance = np.linalg.norm(np.array(moving_sphere1.center) - np.array(moving_sphere2.center))
        if distance <= (robot1.radius + robot2.radius):
            print("Collision detected at step:", i)
            break 

        #Plot details
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Robot 1', markerfacecolor='blue',
                markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='blue'),
        Line2D([0], [0], marker='o', color='w', label='Robot 2', markerfacecolor='red', 
               markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='red'),
        Line2D([0], [0], marker='o', color='w', label='Previous Robot 1 Positions', 
               markerfacecolor='cyan', markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='cyan'),
        Line2D([0], [0], marker='o', color='w', label='Previous Robot 2 Positions', 
               markerfacecolor='magenta', markersize=10, fillstyle= 'none', markeredgewidth=2, markeredgecolor='magenta'),
        Line2D([0], [0], marker='*', color='w', label='Robot 1 Goal', markerfacecolor='blue', markersize=15),
        Line2D([0], [0], marker='*', color='w', label='Robot 2 Goal', markerfacecolor='red', markersize=15)
    ]

    ax.set_xlim([-60, 70])
    ax.legend(handles=legend_elements, loc = 'lower right')
    ax.set_title(f'Traditional PFM Formulation \n Number of Steps: {nb_steps}')

    plt.show()

if __name__ == '__main__':
    test_planner_2_robots(1500, rep_weight1=500, rep_weight2=1)
