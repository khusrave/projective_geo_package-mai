from importlib_metadata import NullFinder
import matplotlib.pyplot as plt
import numpy as np

""" stereographic projection from plane to sphere"""
def inv_stereo(x_coord, y_coord ):
    const = x_coord ** 2 + y_coord ** 2 + 1 #denominator
    if const !=0:
        result =  1 / const * np.array([2* x_coord, 2 * y_coord, x_coord ** 2 + y_coord ** 2 - 1 ])
    else:
        result = None
        print("Error denominator is zero")
    return result
    
"""stereographic projection from sphere to plane"""
def stereo(x_sphere, y_sphere, z_sphere):
    if z_sphere != 1:
        result = 1 /(z_sphere - 1) * np.array([x_sphere, y_sphere])
    else:
        result = None
        print("Error denominator is zero")
    return result

"""creating a circle with radius r and pos as position of center"""
def circ(r, pos):
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circl = pos[0] + r * np.cos(theta)
    y_circl = pos[1] + r * np.sin(theta)
    return x_circl, y_circl, [0]*1000

def sphere(r, pos):
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_sph = pos[0] + np.cos(u)*np.sin(v)
    y_sph = pos[1] + np.sin(u)*np.sin(v)
    z_sph = pos[2] + np.cos(v)
    return x_sph, y_sph, z_sph

def eval_stereo_list(x_coords, y_coords):
    n = len(x_coords)
    lstx = []
    lsty = []
    lstz = []
    for i in range(n):
        [xs, ys, zs] =  inv_stereo(x_coords[i], y_coords[i] )
        lstx.append(xs)
        lsty.append(ys)
        lstz.append(zs)
    return lstx, lsty, lstz

def mob_transf(x_coords, y_coords):
    n = len(x_coords)
    lstx = []
    lsty = []
    
    for i in range(n):
        [xs, ys] = invT2( T1([x_coords[i], y_coords[i]], A1, A2, A3 ), B1, B2, B3 )
        lstx.append(xs)
        lsty.append(ys)
        
    lstz = [0]*n
    return lstx, lsty, lstz

def plot_circ(r, pos):
    # ax = plt.figure().add_subplot(projection='3d')
    x_coords, y_coords, z_coords = circ(r, pos)
    ax.plot(x_coords, y_coords, z_coords)

def vecCom(vec):
    return complex(vec[0], vec[1])

def comVec(com):
    return [np.real(com), np.imag(com)]

def T1(Z, A1, A2, A3):
    z = vecCom(Z)
    a1 = vecCom(A1)
    a2 = vecCom(A2)
    a3 = vecCom(A3)
    return comVec((z*(a1-a3)+a2*(a3-a1))/(z*(a1-a2)+a3*(a2-a1)))
 
def invT2(Z, B1, B2, B3):
    z = vecCom(Z)
    b1 = vecCom(B1)
    b2 = vecCom(B2)
    b3 = vecCom(B3)
    return comVec((b3*(b1-b2)*z+b2*(b3-b1))/(z*(b1-b2)+b3*(b1-b2)))

A1 = [1,2]
A2 = [2,3]
A3 = [4,5]
B1 = [6,7]
B2 = [7,8]
B3 = [9,10]


ax = plt.figure().add_subplot(projection='3d')
# Plot of the Stereographic projection of the circle 
x_coords, y_coords, z_coords = eval_stereo_list(circ(1,[1,1])[0], circ(1,[1,1])[1] )
# The unitary sphere
x_sphere, y_sphere, z_sphere = sphere(1,[0,0,0])
# Transform the cirlce in the plane
x_t, y_t, z_t = mob_transf(circ(1,[1,1])[0], circ(1,[1,1])[1] )
# Transformation projected 
x_st, y_st, z_st = eval_stereo_list(x_t, y_t )

ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3)
ax.plot(x_coords, y_coords, z_coords , label='Stereo Proj')
ax.plot(x_st, y_st, z_st , label='Transformation R2')
ax.plot(x_t, y_t, z_t , label='Transformation R3')
# Plot the circle in the plane
plot_circ(1,[1,1])

ax.legend()
ax.set_box_aspect((1, 1, 0.6))

plt.show()