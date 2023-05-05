import numpy as np
import math

'''
COMPUTE_LINE_EQUATIONS
    points - a list of two points on a line as [[x1, y1], [x2, y2]]
Returns:
    coefficients a, b, c of line ax + by + c = 0
'''

def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    _FLOAT_EPS_4 = np.finfo(float).eps * 4.0
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def compute_line_equation(points):
    pt1 = points[0]
    pt2 = points[1]
    print("pt1: {}".format(pt1))
    print("pt2: {}".format(pt2))
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    # compute intercept by substituting one of the points in
    # equation y = slope*x + intercept
    intercept = pt2[1] - slope * pt2[0]

    # get line equation coefficients
    a = -slope
    b = 1.0
    c = -intercept
    return a, b, c


'''
COMPUTE_POINT_OF_INTERSECTION
    line1 - defined by its coefficients [a1, b1, c1]
    line2 - defined by its coefficients [a2, b2, c2]
Returns:
    point of intersection (x, y)
'''


def compute_point_of_intersection(line1, line2):
    # ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    (a1, b1, c1) = line1
    (a2, b2, c2) = line2
    x = (-c2 + c1) / (-a1 + a2)
    y = (-a1 * x) - c1
    return (x, y)


'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''


def compute_vanishing_point(points):
    # compute line equations for two lines
    a1, b1, c1 = compute_line_equation([points[0], points[1]])
    a2, b2, c2 = compute_line_equation([points[2], points[3]])

    # compute point of intersection
    vanishing_point = compute_point_of_intersection((a1, b1, c1), (a2, b2, c2))
    #assert np.dot([vanishing_point[0], vanishing_point[1], 1.0], np.transpose([a1, b1, c1])) == 0.0, \
    #    'Dot product of point and line must be zero'
    return vanishing_point


'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''


def compute_K_from_vanishing_points(vanishing_points):
    # form equations A.w = 0 with 4 constraints of omega (w)
    # A = np.zeros((vanishing_points.shape[0], 4), dtype=np.float32)
    A = []
    for i, point_i in enumerate(vanishing_points):
        for j, point_j in enumerate(vanishing_points):
            if i != j and j > i:
                point_i_homogeneous = [point_i[0], point_i[1], 1.0]
                point_j_homogeneous = [point_j[0], point_j[1], 1.0]
                A.append(
                    [point_i_homogeneous[0] * point_j_homogeneous[0] + point_i_homogeneous[1] * point_j_homogeneous[1], \
                     point_i_homogeneous[0] * point_j_homogeneous[2] + point_i_homogeneous[2] * point_j_homogeneous[0], \
                     point_i_homogeneous[1] * point_j_homogeneous[2] + point_i_homogeneous[2] * point_j_homogeneous[1], \
                     point_i_homogeneous[2] * point_j_homogeneous[2]])
    A = np.array(A, dtype=np.float32)
    u, s, v_t = np.linalg.svd(A, full_matrices=True)
    # 4 constraints of omega (w) can be obtained as the last column of v or last row of v_transpose
    w1, w4, w5, w6 = v_t.T[:, -1]
    # form omega matrix
    w = np.array([[w1, 0., w4],
                  [0., w1, w5],
                  [w4, w5, w6]])
    # w = (K.K_transpose)^(-1)
    # K can be obtained by Cholesky factorization followed by its inverse
    K_transpose_inv = np.linalg.cholesky(w)
    K = np.linalg.inv(K_transpose_inv.T)
    # divide by the scaling factor
    K = K / K[-1, -1]

    # return intrinsic matrix
    return K


'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''


def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # compute vanishing line from first pair of vanishing points
    vanishing_line1 = np.array(compute_line_equation(vanishing_pair1)).transpose()
    # compute vanishing line from second pair of vanishing points
    vanishing_line2 = np.array(compute_line_equation(vanishing_pair2)).transpose()
    # compute omega inverse
    w_inv = np.dot(K, K.transpose())
    # compute angle between these two planes
    l1T_winv_l2 = np.dot(vanishing_line1.transpose(), np.dot(w_inv, vanishing_line2))
    sqrt_l1T_winv_l1 = np.sqrt(np.dot(vanishing_line1.transpose(), np.dot(w_inv, vanishing_line1)))
    sqrt_l2T_winv_l2 = np.sqrt(np.dot(vanishing_line2.transpose(), np.dot(w_inv, vanishing_line2)))
    theta = np.arccos(l1T_winv_l2 / np.dot(sqrt_l1T_winv_l1, sqrt_l2T_winv_l2))
    # convert the angle between planes to degrees and return
    return np.degrees(theta)


'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''


def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    ## estimate real-world direction vectors given vanishing points
    # first image
    d1i = []
    for v1i in vanishing_points1:
        # vanishing point (v) and 3-dimensional direction vector (d) are related as [d = K.v]
        v1i_homogeneous = np.array([v1i[0], v1i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v1i_homogeneous.T)
        d1i.append(KinvV / np.sqrt(
            KinvV[0] ** 2 + KinvV[1] ** 2 + KinvV[2] ** 2))  # normalize to make sure you obtain a unit vector
    d1i = np.array(d1i)
    # second image
    d2i = []
    for v2i in vanishing_points2:
        # vanishing point (v) and 3-dimensional direction vector (d) are related as [d = K.v]
        v2i_homogeneous = np.array([v2i[0], v2i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v2i_homogeneous.T)
        d2i.append(KinvV / np.sqrt(
            KinvV[0] ** 2 + KinvV[1] ** 2 + KinvV[2] ** 2))  # normalize to make sure you obtain a unit vector
    d2i = np.array(d2i)

    # the directional vectors in image 1 and image 2 are related by a rotation, R i.e. [d2i = R.d1i] => [R = d2i.d1i_inverse]
    R = np.dot(d2i.T, np.linalg.inv(d1i.T))
    return R


def compute_P(pt_2d, pt_3d):
    A = []
    for n in range(pt_2d.shape[0]):
        A.append([[pt_3d[n][0], pt_3d[n][1], pt_3d[n][2], 1, 0, 0, 0, 0, -pt_2d[n][0] * pt_3d[n][0],
                   -pt_2d[n][0] * pt_3d[n][1], -pt_2d[n][0] * pt_3d[n][2], -pt_2d[n][0]],
                  [0, 0, 0, 0, pt_3d[n][0], pt_3d[n][1], pt_3d[n][2], 1, -pt_2d[n][1] * pt_3d[n][0],
                   -pt_2d[n][1] * pt_3d[n][1], -pt_2d[n][1] * pt_3d[n][2], -pt_2d[n][1]]])
    A = np.array(A)
    A = A.reshape(A.shape[0] * A.shape[1], A.shape[2])

    # SVD
    u, s, vh = np.linalg.svd(A)

    # get the smallest singular value of SVD
    P = vh[-1, :].reshape(3, 4)

    return A, P


def compute_KR(P):
    M = P[0:3, 0:3]
    # QR decomposition
    q, r = np.linalg.qr(np.linalg.inv(M))
    R = np.linalg.inv(q)
    K = np.linalg.inv(r)
    # translation vector
    t = np.dot(np.linalg.inv(K), P[:, -1])

    D = np.array([[np.sign(K[0, 0]), 0, 0],
                  [0, np.sign(K[1, 1]), 0],
                  [0, 0, np.sign(K[2, 2])]])

    # K,R,t correction
    K = np.dot(K, D)
    R = np.dot(np.linalg.inv(D), R)
    t = np.dot(np.linalg.inv(D), t)
    t = np.expand_dims(t, axis=1)

    # normalize K
    K = K / K[-1, -1]

    return K, R, t


def reproject(pt_3d, K, R, t):
    pt_3d = np.hstack((pt_3d, np.ones((pt_3d.shape[0], 1))))

    Rt = np.hstack((R, t))
    KRt = np.dot(K, Rt)

    pt_reproject = np.dot(KRt, pt_3d.T)
    pt_reproject = pt_reproject / pt_reproject[-1, :]

    return KRt, pt_reproject  # , extrinsic, projection, intrinsic, P_combined

import plotly
import plotly.graph_objs as go

def visualize_3d(pts_3d):

    # x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.scatter(x, y, z, color='blue')
    # plt.show()
    # plt.close()

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d[:,0].tolist(),
            y=pts_3d[:,2].tolist(),
            z=(-pts_3d[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='firebrick',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=0.8
            )
        )
    )

    plotly.offline.plot({
            "data": total_points_list,
            "layout": go.Layout(title="All Planes")
        }, auto_open=True)

def visualize_3d_color(pts_3d, img, key_pts):
    key_pts = np.rint(key_pts).astype('int')
    kp_x = np.clip(key_pts[:,0], 0, img.shape[0]-1)
    kp_y = np.clip(key_pts[:,1], 0, img.shape[1]-1)
    r = img[:,:,0][kp_y, kp_x]
    g = img[:,:,1][kp_y, kp_x]
    b = img[:,:,2][kp_y, kp_x]

    colors = np.array([r,g,b]).T / 255.0

    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # for p, c in zip(pts_3d, colors):
    #     ax.plot([p[0]], [p[1]], [p[2]], '.', color=(c[0], c[1], c[2]), markersize=8, alpha=0.5)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # plt.close()

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d[:,0].tolist(),
            y=pts_3d[:,1].tolist(),
            z=(pts_3d[:,2]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                line=dict(
                    color=colors,
                    width=0.1
                ),
                opacity=1
            )
        )
    )

    plotly.offline.plot({
            "data": total_points_list,
            "layout": go.Layout(title="All Planes")
        }, auto_open=True)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def get_planer_indexes(points, a, b, c, d):
    '''
        :parm pts    - array of 3D points
        :param plane - coefficient of plane (i.e. A, B, C, D)
        :returns     - indexes of points which are in plance
    '''
    # Compute A*pt[0] + B*pt[1] + C*pt[3] + D for each point pt in pts
    # Checks that abs(...) is below threshold (1e-6) to allow for floating point error
    print(a * points[0] + b * points[1] + c * points[2])
    return np.where(a * points[0] + b * points[1] + c * points[2] == d)
