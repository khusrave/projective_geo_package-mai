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

def plot_circ(r, pos):

    # ax = plt.figure().add_subplot(projection='3d')
    x_coords, y_coords, z_coords = circ(r, pos)
    ax.plot(x_coords, y_coords, z_coords)




ax = plt.figure().add_subplot(projection='3d')
x_coords, y_coords, z_coords = eval_stereo_list(circ(1,[1,1])[0], circ(1,[1,1])[1] )
x_sphere, y_sphere, z_sphere = sphere(1,[0,0,0])
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3)
ax.plot(x_coords, y_coords, z_coords , label='Stereo Proj')
plot_circ(1,[1,1])
ax.legend()
ax.set_box_aspect((1, 1, 0.6))

plt.show()