import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    g = np.zeros(3)
    x,y,th = xvec
    V,w = u	

    d_th = w*dt
    if abs(w) < EPSILON_OMEGA:
	    g[0] = x + V*np.cos(th)*dt
	    g[1] = y + V*np.sin(th)*dt
	    g[2] = th + d_th
    else:
        g[0] = x + V*(np.sin(th + d_th)-np.sin(th))/w
        g[1] = y + V*(-np.cos(th + d_th) + np.cos(th))/w
        g[2] = th + d_th
	
    Gx = np.eye(3)
    Gu = np.zeros((3,2))
    if compute_jacobians:	
	    if abs(w) < EPSILON_OMEGA:
	        Gu[0,0] = np.cos(th)*dt
	        Gu[1,0] = np.sin(th)*dt
	        Gu[2,1] = dt

	        Gx[0,2] = -V*np.sin(th)*dt
	        Gx[1,2] = V*np.cos(th)*dt
	    else:	
	        Gx[0,2] = V*(np.cos(th + d_th) - np.cos(th))/w  
            	Gx[1,2] = V*(np.sin(th + d_th) - np.sin(th))/w 

	        Gu[0,0] = (np.sin(th + d_th) - np.sin(th))/w
	        Gu[1,0] = (np.cos(th) - np.cos(th + d_th))/w
	        Gu[0,1] = -V*(np.sin(th + d_th) - np.sin(th))/w**2 + V*(dt*np.cos(th + d_th))/w
	        Gu[1,1] = -V*(np.cos(th) - np.cos(th + d_th))/w**2 + V*(dt*np.sin(th + d_th))/w
	        Gu[2,1] = dt
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    xb, yb, thb = tf_base_to_camera
    R1 = np.array([[np.cos(x[2]), -np.sin(x[2]), 0], 
		   [np.sin(x[2]),  np.cos(x[2]), 0],
		   [0,           0,          1]])
    
    pose_cam_wf = np.dot(R1, np.asarray(tf_base_to_camera).reshape((3,1)))+x.reshape((3,1))

    alpha_cam_wf = alpha - pose_cam_wf[2]
    
    R2 = np.array([[np.cos(-alpha), -np.sin(-alpha), 0],
		   [np.sin(-alpha),  np.cos(-alpha), 0],
                   [0,              0,             1]])
    
    proj = np.dot(R2, pose_cam_wf)
    r_cam = r -proj[0]
    h = np.squeeze(np.array([alpha_cam_wf, r_cam]))

    ########## Code ends here ##########

    if not compute_jacobian:
        return h
    else:
	Hx = np.zeros((2,3))
	Hx[0,2] = -1
	Hx[1,0] = -np.cos(alpha)
	Hx[1,1] = -np.sin(alpha)
	Hx[1,2] = -(-xb*np.sin(x[2])*np.cos(-alpha) - yb*np.cos(x[2])*np.cos(-alpha) - xb*np.cos(x[2])*np.sin(-alpha) + yb*np.sin(x[2])*np.sin(-alpha))

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
