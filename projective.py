from ctypes.wintypes import RGB
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def RGB(r, g ,b):
    """ Function to normalize RGB """
    return (float(r)/255, float(g)/255, float(b)/255)

darkred= RGB(92,26,18)


def inv_stereo(x_coord, y_coord ):
    """ stereographic projection from plane to sphere"""
    const = x_coord ** 2 + y_coord ** 2 + 1 #denominator
    if const !=0:
        result =  1 / const * np.array([2* x_coord, 2 * y_coord, x_coord ** 2 + y_coord ** 2 - 1 ])
    else:
        result = None
        print("Error denominator is zero")
    return result
    

def stereo(x_sphere, y_sphere, z_sphere):
    """stereographic projection from sphere to plane"""
    if z_sphere != 1:
        result = 1 /(z_sphere - 1) * np.array([x_sphere, y_sphere])
    else:
        result = None
        print("Error denominator is zero")
    return result


def circ(r, pos):
    """creating a circle with radius r and pos as position of center"""
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circl = pos[0] + r * np.cos(theta)
    y_circl = pos[1] + r * np.sin(theta)
    return x_circl, y_circl, [0]*1000

def sphere(r, pos):
    """ Function to get the points of a sphere"""
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_sph = pos[0] + np.cos(u)*np.sin(v)
    y_sph = pos[1] + np.sin(u)*np.sin(v)
    z_sph = pos[2] + np.cos(v)
    return x_sph, y_sph, z_sph

def eval_stereo_list(x_coords, y_coords):
    """ Function to apply a inverse stereographic projection to a set of points """
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
    """ Function to plot a circle in 3D """
    x_coords, y_coords, z_coords = circ(r, pos)
    ax.plot(x_coords, y_coords, z_coords)

def vecCom(vec):
    return complex(vec[0], vec[1])

def comVec(com):
    return [np.real(com), np.imag(com)]

def T1(Z, A1, A2, A3):
    """Mobius transformation  send A1->0, A2->1, A3->inf """
    z = vecCom(Z)
    a1 = vecCom(A1)
    a2 = vecCom(A2)
    a3 = vecCom(A3)
    return comVec((z*(a1-a3)+a2*(a3-a1))/(z*(a1-a2)+a3*(a2-a1)))
 
def invT2(Z, B1, B2, B3):
    """Inverse Mobius transformation  send 0->B1, 1->B2, inf->B3"""
    z = vecCom(Z)
    b1 = vecCom(B1)
    b2 = vecCom(B2)
    b3 = vecCom(B3)
    return comVec((b3*(b1-b2)*z+b2*(b3-b1))/(z*(b1-b2)+b3*(b1-b2)))

def mob_transf_T1(x_coords, y_coords):
    """ Function to apply a Mobius transformation define by six points to a set of points """
    n = len(x_coords)
    lstx = []
    lsty = []
    
    for i in range(n):
        [xs, ys] = T1([x_coords[i], y_coords[i]], A1, A2, A3 )
        lstx.append(xs)
        lsty.append(ys)
    lstz = [0]*n
    return lstx, lsty, lstz

def mob_transf_invT2(x_coords, y_coords):
    """ Function to apply a Mobius transformation define by six points to a set of points """
    n = len(x_coords)
    lstx = []
    lsty = []
    
    for i in range(n):
        [xs, ys] = invT2([x_coords[i], y_coords[i]], B1, B2, B3 )
        lstx.append(xs)
        lsty.append(ys)
    lstz = [0]*n
    return lstx, lsty, lstz


def mob_transf_Comp(x_coords, y_coords):
    """ Function to apply a Mobius transformation define by six points to a set of points """
    t1x, t1y, _ = mob_transf_T1(x_coords, y_coords)
    lstx, lsty, lstz = mob_transf_invT2(t1x, t1y)

    return lstx, lsty, lstz

def mob_transf_Comp_ster(x_coords, y_coords):
    """ Function to apply a Mobius transformation define by six points to a set of points """
    t1x, t1y, _ = mob_transf_T1(x_coords, y_coords)
    lstx, lsty, lstz = mob_transf_invT2(t1x, t1y)
    return lstx, lsty, lstz

def line(dir):
    """ Function to define a line """
    l = np.linspace(-50, 50, 1000)
    return dir[0]*l, dir[1]*l, [0]*1000

def plot_line(dir):
    """ Function to plot a line """
    xs, ys, zs = line(dir)
    ax.plot(xs, ys, zs)

def plot_stereo(x_coords, y_coords, plot_label='', cl=RGB(18, 47, 92) ):
    """ Function to plot the inverse stereographic projection of a set of points """
    # Function to plot the inverse stereographic projection of points in the plane
    x_coords, y_coords, z_coords = eval_stereo_list(x_coords, y_coords )
    # Plot the transformed points
    ax.plot(x_coords, y_coords, z_coords, label=plot_label, color=cl)

def plot_stere_circ(r, pos, label):
    """ Function to plot the inverse stereographic projection of a cirlce """
    # Get the points of the circle
    xs, ys, _ = circ(r, pos)
    # Plot the inverse stereographic projection of the circle
    plot_stereo(xs, ys, label)
    
def plot_stere_line(dir, label):
    """ Function to plot the inverse stereographic projection of a line """
    # Get the points of the line
    xs, ys, _ = line(dir)
    # Plot the inverse stereographic projection of the line
    plot_stereo(xs, ys, label)

def plot_sphere(r, pos):
    """ Function to plot a sphere of radius r and position pos """
    # Points of the sphere
    x_sph, y_sph, z_sph = sphere(r, pos)
    # Plot the sphere
    ax.plot_surface(x_sph, y_sph, z_sph, alpha=0.3)

def Lat_circ(n, r_i, r_f):
    """Function to create a list of circles that correspond to latitude constanc circles"""
    # Range of radius
    radius = np.linspace(r_i, r_f, n)
    # List to store the circles
    circles = [] 
    for r in radius:
        # Add circles of the given range of radius
        circles.append(circ(r, [0,0]))
    return circles

def Long_circ(n):
    """Function to define create a list of lines that correspond to constant longitude circles"""
    # Define a range of angles
    angles = np.linspace(0, 2*np.pi, n)
    # List of lines
    lines = [] 
    for theta in angles:
        # Add lines 
        lines.append(line([np.cos(theta), np.sin(theta)]))
    return lines

def plot_3d(xs,ys,zs, lbl='', cl='blue'):
    ax.plot(xs, ys, zs, label= lbl, color= cl)

def plot_2d(xs,ys, lbl='',  cl='blue'):
    ax2d.plot(xs, ys, label= lbl, color= cl)


def eval_plots(ls_geo_objs, cl1='black', cl2='red'):
    """ Function that plot a list of geometric objects """
    for geo_obj in ls_geo_objs:
        xls = geo_obj[0]
        yls = geo_obj[1]
        zls = geo_obj[2]
        x_ster, y_ster, z_ster = eval_stereo_list(xls, yls)
        plot_3d(xs=x_ster, ys=y_ster, zs=z_ster, cl=cl1 )
        plot_2d(xs=xls, ys=yls, cl=cl2 )


def eval_transS(ls_geo_objs, color="red"):
    """ Function that take a list of geometric objects define on R^2 
    apply a mobius transformation and plot the inverse stereographic projection of them"""
    for geo_obj in ls_geo_objs:
        xs = geo_obj[0]
        ys = geo_obj[1]
        xt, yt, zt = mob_transf_Comp(xs, ys)
        plot_stereo(x_coords=xt, y_coords=yt, cl=color)

def eval_transPlane(ls_geo_objs, cl="red"):
    """ Function that take a list of geometric objects define on R^2 
    apply a mobius transformation and plot the inverse stereographic projection of them"""
    for geo_obj in ls_geo_objs:
        xs = geo_obj[0]
        ys = geo_obj[1]
        n = len(xs)
        xt, yt, _ = mob_transf_Comp(xs, ys)
        ax.plot(xt, yt, [0]*n, color=cl)



A1 = [1,2]
A2 = [2,3]
A3 = [4,5]
B1 = [6,7]
B2 = [7,8]
B3 = [9,10]

def vecCom(vec):
    return complex(vec[0], vec[1])

def comVec(com):
    return [np.real(com), np.imag(com)]

fig = plt.figure(figsize=plt.figaspect(2.))
ax2d = fig.add_subplot(2, 1, 1)
ax = fig.add_subplot(2, 1, 2, projection='3d')
# Plots 
# plot_sphere(1,[0,0,0])
# plot_circ(1,[1,1])
# plot_line([1,1])
# plot_stere_circ(1, [1,1], "Stero Circle")
# plot_stere_line([1,1], "Stero Line")
lat_circles = Lat_circ(10, 0.5, 10)
long_circles = Long_circ(20)
t1_lat_circles = list(map(lambda geo: mob_transf_T1(geo[0], geo[1]), lat_circles ))
t2_lat_circles = list(map(lambda geo: mob_transf_invT2(geo[0], geo[1]), t1_lat_circles ))
#eval_plots_stero(lat_circles)
eval_plots(lat_circles)
#eval_plots(lat_circles, cl1='blue', cl2='purple')
eval_plots(t1_lat_circles, cl1='blue', cl2='purple')
eval_plots(t2_lat_circles, cl1='gray', cl2='green')
#eval_plots(lat_circles, 'blue')
#eval_trasn(lat_circles, 'black')
#eval_transPlane(long_circles, darkred)
#eval_transPlane(lat_circles, 'orange')
# Definition of the range of the box
ax2d.axes.set_xlim(left=-2, right=2) 
ax2d.axes.set_ylim(bottom=-2, top=2) 
ax.axes.set_xlim3d(left=-1.5, right=1.5) 
ax.axes.set_ylim3d(bottom=-1.5, top=1.5) 
ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
ax.set_box_aspect((1, 1, 1))

custom_lines1 = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='purple', lw=4),
                Line2D([0], [0], color='green', lw=4)]
            
custom_lines2 = [Line2D([0], [0], color='black', lw=4),
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='gray', lw=4)
                ]

ax2d.legend(custom_lines1,["Latitute","T1", "InvT2"], loc="upper right")
ax.legend(custom_lines2,["Latitute","T1", "InvT2"], loc="upper right")




plt.show()