"""
Classes involving Spheres for PFM Simulation
"""

import numpy as np
from matplotlib import pyplot as plt


def gca_3d():
    """
    GetcurrentMatplotlibaxes andiftheydonotsupport3Dplotting
    addnewaxesthatsupportit    """
    fig = plt.gcf()
    if len(fig.axes) == 0 or not hasattr(plt.gca(), 'plot3D'):
        axis = fig.add_subplot(111, projection='3d')
    else:
        axis = plt.gca()
    return axis

class Sphere: 
    """ Class for plotting and computing distances to spheres (circles, in 2-D). """
    def __init__(self, center, radius, distance_influence):
        """
        Save the parameters describing the sphere as internal attributes.
        """
        self.center = center
        self.radius = radius
        self.distance_influence = distance_influence

    def update_center(self, new_center):
        # self.center = np.array(new_center).reshape(2, 1)
        self.center = new_center.reshape(2, 1)

    def plot(self, color, axes = None):
        """
        This function draws the sphere (i.e., a circle) of the given radius,
        and the specified color, and then draws another circle in gray with radius equal
        to the distance of influence.
        """
        # Get current axes
        if axes is None:
            axes = plt.gca()

        # Add circle as a patch
        if self.radius > 0:
            # Circle is filled in
            kwargs = {'facecolor': (0.3, 0.3, 0.3)}
            radius_influence = self.radius + self.distance_influence
        else:
            # Circle is hollow
            kwargs = {'fill': False}
            radius_influence = -self.radius - self.distance_influence

        center = (self.center[0, 0], self.center[1, 0])
        axes.add_patch(
            plt.Circle(center,
                       radius=abs(self.radius),
                       edgecolor=color,
                       **kwargs))

        axes.add_patch(
            plt.Circle(center,
                       radius=radius_influence,
                       edgecolor=(0.7, 0.7, 0.7),
                       fill=False))


    def distance(self, points):
        """
        Computes the signed distance between points and the sphere, while taking
        into account whether the sphere is hollow or filled in.
        """
        tol = 2.22e-16
        if points.ndim == 1:
            points = points.reshape(2, -1)

        d_points_sphere = np.zeros((1, points.shape[1]))
        for i in range(d_points_sphere.shape[1]):
            point = points[:,i]

            dist = np.linalg.norm(point.flatten() - self.center.flatten())

            #Inside of filled circle, aka radius is greater than dist
            if self.radius > 0 - tol and self.radius > dist - tol:
                dist = -abs(abs(self.radius) - abs(dist))
            #Outside of hollow circle, aka absolute value of radius is smaller than dist
            elif self.radius < 0 + tol and abs(self.radius) < dist + tol:
                # dist = -(dist - abs(self.radius))
                dist = -abs(abs(self.radius) - abs(dist))
            else:
                dist = abs(abs(self.radius) - abs(dist))
                # print(f'dist: {dist}')

            d_points_sphere[:, i] = dist.astype(np.float64)

        return d_points_sphere #returns 1xnb_points array

    def distance_grad(self, points):
        """
        Computes the gradient of the signed distance between points and the
        sphere, consistently with the definition of Sphere.distance.
        """
        if points.ndim == 1:
            points = points.reshape(2, -1)

        edge_dist = self.distance(points)

        grad_d_points_sphere = np.zeros_like(points, dtype=np.float64)

        for i in range(edge_dist.shape[1]):
            point = points[:,i]

            #Distance to edge of sphere
            dist = edge_dist[0,i].item()

            # delta currently gives distance to center. Do i need distance to edge?

            delta = point.flatten() - self.center.flatten()
            #as point gets closer to sphere, dist goes down and grad goes up

            if np.array_equal(point, self.center):
                grad = np.array([[0.],[0.]])
            elif self.radius < 0:
                grad = -delta / (abs(self.radius) - dist)
            else:
                grad = delta / (abs(self.radius) + dist)

            grad_d_points_sphere[:, i] = grad.astype(np.float64)

        return grad_d_points_sphere

class RepulsiveSphere: 
    """ Repulsive potential for a sphere """
    def __init__(self, sphere): 
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """s
        Evaluate the repulsive potential from  sphere at the location x= x_eval. The function
        returns the repulsive potential as given by      (  eq:repulsive  ).
        """
        x_eval = x_eval.reshape(-1,1)

        distance = self.sphere.distance(x_eval)

        distance_influence = self.sphere.distance_influence

        if distance > distance_influence:
            u_rep = 0
        elif (distance_influence > distance and distance > 0):
            u_rep = ((distance**-1 - distance_influence**-1)**2) / 2.
            u_rep = u_rep.item()
        else:
            u_rep = np.nan
            # u_rep = 0
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by (eq:repulsive-gradient).

        x_eval = point
        use Equation 4:
        if 0 < self.sphere.distance() < d_influence
        grad = -(1/self.sphere.distance(x_eval) - 1/d_influence ) * (1/self.sphere.distance() ** 2)
        () self.sphere.distance(x_eval)

        """

        x_eval = x_eval.reshape(-1,1)
        di_x = self.sphere.distance(x_eval).item()
        d_influence = self.sphere.distance_influence

        if di_x == 0:
            grad_u_rep = np.nan * np.ones((2,1))
        elif 0 < di_x and di_x < d_influence:
            grad_u_rep = -((di_x**-1 - d_influence**-1)*((di_x ** -2))*
                           self.sphere.distance_grad(x_eval))
        elif di_x> d_influence:
            grad_u_rep = np.array([[0.],[0.]])
        else:
            grad_u_rep = np.nan * np.ones((2,1))


        return grad_u_rep  #returns 2x1 array

class Attractive:
    """ Repulsive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        x_eval = x_eval.reshape(-1,1)

        x_goal = self.potential['x_goal'].reshape(-1,1)
        shape = self.potential['shape']
        if shape == 'conic':
            expo = 1
        else:
            expo = 2
        u_attr = (np.linalg.norm(x_eval.flatten() - x_goal.flatten()))**expo
        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient
        is given by the formula If  potential['shape'] is equal to 'conic', use p=1; if it is
        equal to 'quadratic', use p=2.
        """

        x_eval = x_eval.reshape(-1,1)
        x_eval = x_eval.astype(np.float64)

        x_goal = self.potential['x_goal']
        x_goal = x_goal.astype(np.float64)
        shape = self.potential['shape']
        norm = np.linalg.norm(x_goal - x_eval)

        if norm < 1e-6:
            grad_u_attr = np.zeros_like(x_eval, dtype = float)
        else:
            if shape == 'conic':
                grad_u_attr = (x_eval - x_goal) / norm
            else:
                grad_u_attr = 2 * (x_eval - x_goal)

        return grad_u_attr

class MovingRepulsiveSphere: 
    """ Repulsive potential for a sphere """
    def __init__(self, sphere): 
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere
        self.n_RO_obs1 = 0  #Unit vector pointing from robot to obstacle
        self.v_RO_obs1 = 0  #Velocity of robot WRT obstacle. < 0 if moving away, >0 if moving towards
        self.n_RO_obs2 = 0
        self.v_RO_obs2 = 0
    
    def set_attr(self, n_RO_obs1, v_RO_obs1, n_RO_obs2 = 0, v_RO_obs2 = 0):
        self.n_RO_obs1 = n_RO_obs1
        self.v_RO_obs1 = v_RO_obs1
        self.n_RO_obs2 = n_RO_obs2
        self.v_RO_obs2 = v_RO_obs2      

    def update_center(self, new_center):
        self.sphere.center = new_center.reshape(2, 1)

    def eval(self, x_eval):
        """s
        Evaluate the repulsive potential from  sphere at the location x= x_eval. The function
        returns the repulsive potential as given by      (  eq:repulsive  ).
        """
        x_eval = x_eval.reshape(-1,1)

        distance = self.sphere.distance(x_eval)

        distance_influence = self.sphere.distance_influence

        if distance > distance_influence:
            u_rep = 0
        elif (distance_influence > distance and distance > 0):
            u_rep = distance**-1 - distance_influence**-1
            u_rep = u_rep.item()
        else:
            u_rep = 0
            # u_rep = np.nan
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by (eq:repulsive-gradient).

        x_eval = point
        use Equation 4:
        if 0 < self.sphere.distance() < d_influence
        grad = -(1/self.sphere.distance(x_eval) - 1/d_influence ) * (1/self.sphere.distance() ** 2)
        () self.sphere.distance(x_eval)

        """

        x_eval = x_eval.reshape(-1,1)
        distance = self.sphere.distance(x_eval).item()
        d_influence = self.sphere.distance_influence

        grad_u_rep = np.zeros((2,1))

        if distance == 0:
            # grad_u_rep = np.nan * np.ones((2,1))
            grad_u_rep = np.zeros((2,1))
        elif self.v_RO_obs1 <= 0:
            grad_u_rep += np.zeros((2,1))
        elif self.v_RO_obs2 <= 0:
            grad_u_rep += np.zeros((2,1))
        elif 0 < distance < d_influence:
            if self.v_RO_obs1 > 0:
                grad_u_rep += (-1 / (distance **2)) * self.n_RO_obs1
            if self.v_RO_obs2 > 0:
                grad_u_rep += (-1 / (distance **2)) * self.n_RO_obs2
        else:
            # grad_u_rep = np.nan * np.ones((2,1))
            grad_u_rep = np.zeros((2,1))

        return grad_u_rep  #returns 2x1 array

class MovingRepulsiveSphere2R: 
    """ Repulsive potential for a sphere """
    def __init__(self, sphere): 
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere
        self.n_RO = 0  #Unit vector pointing from robot to obstacle
        self.v_RO = 0  #Velocity of robot WRT obstacle. < 0 if moving away, >0 if moving towards
    
    def set_attr(self, n_RO, v_RO):
        self.n_RO = n_RO
        self.v_RO = v_RO

    def update_center(self, new_center):
        # self.center = np.array(new_center).reshape(2, 1)
        self.sphere.center = new_center.reshape(2, 1)

    def eval(self, x_eval):
        """s
        Evaluate the repulsive potential from  sphere at the location x= x_eval. The function
        returns the repulsive potential as given by      (  eq:repulsive  ).
        """
        x_eval = x_eval.reshape(-1,1)

        distance = self.sphere.distance(x_eval)

        distance_influence = self.sphere.distance_influence

        if distance > distance_influence or self.v_RO <= 0:
            u_rep = 0
        elif (distance_influence > distance and distance > 0) and self.v_RO > 0:
            u_rep = distance**-1 - distance_influence**-1
            u_rep = u_rep.item()
        else:
            u_rep = 0
            # u_rep = np.nan
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by (eq:repulsive-gradient).

        x_eval = point
        use Equation 4:
        if 0 < self.sphere.distance() < d_influence
        grad = -(1/self.sphere.distance(x_eval) - 1/d_influence ) * (1/self.sphere.distance() ** 2)
        () self.sphere.distance(x_eval)

        """

        x_eval = x_eval.reshape(-1,1)
        distance = self.sphere.distance(x_eval).item()
        d_influence = self.sphere.distance_influence

        if distance == 0:
            # grad_u_rep = np.nan * np.ones((2,1))
            grad_u_rep = np.zeros((2,1))
        elif self.v_RO <= 0:
            grad_u_rep = np.zeros((2,1))
        elif 0 < distance < d_influence and self.v_RO > 0:
            grad_u_rep = (-1/ (distance **2)) * self.n_RO
        else:
            # grad_u_rep = np.nan * np.ones((2,1))
            grad_u_rep = np.zeros((2,1))

        return grad_u_rep  #returns 2x1 array

