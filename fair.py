#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector

from disgrid import Data

spanselector = None
spanselector_defined_on = None
mouse_pressed = False

def planet_from_simdata(simid, p1=1, p2=2):
    data = Data(simid)
    
    time_unit = "kyr"
    
    planets = [Planet(), Planet()]
    for pdata,p in zip([data.planets[p1],data.planets[p2]], planets):
        a = pdata.get("semi-major axis")
        time = a.time.to("kyr")
        p.set('a', time, a.data.to("au"))
        # 2D, thus set ascending node to 0
        Omega = np.zeros(len(a.time))
        p.set('Omega', time, Omega, tomap=True)
        omega = pdata.get("pericenter angle")
        p.set('omega', time, omega.data, tomap=True)
        M = pdata.get("mean anomaly")
        p.set('M', time, M.data, tomap=True)
        val = p.omega+p.Omega
        p.set('omega_bar', time, val)
        p.l = p.M + p.omega_bar
        p.unify_length()
    
    return planets

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("simid")
    # parser.add_argument("outdir")
    parser.add_argument("-r", "--resonance", default="2:1", help="Type of the resonance, default is 2:1")
    parser.add_argument("-p", "--planets", type=int, nargs=2, default=[1,2], help="Indices of planets in data.")
    args = parser.parse_args()

    # get the integer values from the resonance string
    parts = args.resonance.split(":")
    if not len(parts) == 2:
        raise ValueError("Resonance argument must be of type 'p+q:p' but is '{}'".format(args.resonance))
    try:
        resonance_p = int(parts[1])
        resonance_q = int(parts[0]) - resonance_p
    except ValueError:
        raise ValueError("Resonance values must be integers, but given are '{}' and '{}'".format(parts[0], parts[1]))

    # get planet data

    planets = planet_from_simdata(args.simid, p1=args.planets[0], p2=args.planets[1])
    p1, p2 = planets

    plt.rcParams['figure.constrained_layout.use'] = True
    fig = plt.figure()
    #gs = fig.add_gridspec(8,4,figure=fig)
    gs = fig.add_gridspec(6,3,figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax1 = fig.add_subplot(gs[1,:])
    #ax2 = fig.add_subplot(gs[2,:])

    plot_semimajor_axis(ax0, p1, p2)
    ax0.grid(alpha=0.3)
    plot_period_ratio(ax1, p1, p2)
    ax1.grid(alpha=0.3)

    # # plot resonant angles
    # theta1, theta2 = resonant_angle(resonance_p,resonance_q,p1.l, p2.l,
    #                                p1.omega_bar, p2.omega_bar)

    # ax2.plot(time, theta1, alpha=0.7)
    # ax2.plot(time, theta2, alpha=0.7)
    # ax2.set_xlabel("time")

    # fair_axes = []
    # fair_axes.append( fig.add_subplot(gs[2:4, 0:2]) )
    # fair_axes.append( fig.add_subplot(gs[2:4, 2:4]) )
    # fair_axes.append( fig.add_subplot(gs[5:7, 0:2]) )
    # fair_axes.append( fig.add_subplot(gs[5:7, 2:4]) )
    # ra_axes = [] # axes for resonant angles
    # ra_axes.append( fig.add_subplot(gs[4, 0:2]) )
    # ra_axes.append( fig.add_subplot(gs[4, 2:4]) )
    # ra_axes.append( fig.add_subplot(gs[7, 0:2]) )
    # ra_axes.append( fig.add_subplot(gs[7, 2:4]) )

    fair_axes = []
    fair_axes.append( fig.add_subplot(gs[2, 0]) )
    fair_axes.append( fig.add_subplot(gs[3, 0]) )
    fair_axes.append( fig.add_subplot(gs[4, 0]) )
    fair_axes.append( fig.add_subplot(gs[5, 0]) )
    ra_axes = [] # axes for resonant angles
    ra_axes.append( fig.add_subplot(gs[2, 1:]) )
    ra_axes.append( fig.add_subplot(gs[3, 1:]) )
    ra_axes.append( fig.add_subplot(gs[4, 1:]) )
    ra_axes.append( fig.add_subplot(gs[5, 1:]) )

    for n in range(4):
        plot_fair(fair_axes[n], n, p1, p2)
        plot_resonant_angles(ra_axes[n], n, p1, p2, resonance_p, resonance_q)

    fig.suptitle("FAIR method for {}:{} eccentricity type MMR resonance with p,q = {},{}".format(resonance_p+resonance_q, resonance_p, resonance_p, resonance_q))

    # global vars for selector
    global select_axes
    select_axes = []
    select_axes.append(ax0)
    select_axes.append(ax1)
    for ax in ra_axes:
        select_axes.append(ax)
    #select_axes.append(ax2)
    global xlims
    if hasattr(p1.time, "unit"):
        xlims = [p1.time.value[0], p1.time.value[-1]]
    else:
        xlims = [p1.time[0], p1.time[-1]]
    global rectangles_need_update
    rectangles_need_update = False
    # add hidden rectangles
    xmin, xmax = xlims
    color = ax1.get_lines()[0].get_color()
    for ax in select_axes:
        ymin, ymax = ax.get_ylim()
        ax.add_patch( Rectangle( (xmin, ymin), xmax-xmin, ymax-ymin, color=color, alpha=0.3, visible=False ))

   
    def mouse_button_press(event):
        global mouse_pressed
        mouse_pressed = True

    def mouse_button_release(event):
        global mouse_pressed
        mouse_pressed = False

    def onselect_fair_update(xmin, xmax):
        # get x values and ylimits
        ax = select_axes[0]
        x = ax.get_lines()[-1].get_xdata()
        if hasattr(x, "unit"):
            x = x.value
        ymin, ymax = ax.get_ylim()
        # update global xmin, xmax
        global xlims
        xlims = [xmin, xmax]

        # update fair plots
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)
        for n in range(4):
            fair_axes[n].clear()
            plot_fair(fair_axes[n], n, planets[0], planets[1], inds=[indmin,indmax])
            ra_axes[n].clear()
            plot_resonant_angles(ra_axes[n], n, p1, p2, resonance_p, resonance_q, inds=[indmin,indmax])
        global rectangles_need_update
        rectangles_need_update = True

    def enter_axes(event):
        # register a span selector when entering the axis
        ax = event.inaxes
        # remove all prior spanselectors
        global spanselector
        global spanselector_defined_on
        if ax in select_axes and not mouse_pressed:
            if spanselector is None or spanselector_defined_on != ax:
                #if spanselector is not None:
                #    spanselector.disconnect_events()
                spanselector_defined_on = ax
                color = select_axes[-1].get_lines()[-1].get_color()
                del(spanselector)
                spanselector = SpanSelector(ax,onselect_fair_update,
                        'horizontal', useblit=True,
                        props=dict(alpha=0.3, facecolor=color))
                #print("registered new span selector on", ax)
                ax.figure.canvas.draw()


    def update_rectangles(event):
        # only plot rectangle if user selected region
        global rectangles_need_update
        if not rectangles_need_update:
            return
        global xlims
        global select_axes
        xmin, xmax = xlims
        for ax in select_axes:
            try:
                p = ax.patches[0]
                p.set_x(xmin)
                p.set_width(xmax-xmin)
                p.set_visible(True)
                [p.remove() for p in ax.patches[1:]]
            except IndexError:
                pass
        rectangles_need_update = False
        ax.figure.canvas.draw()

    # register some gui events
    fig.canvas.mpl_connect('axes_enter_event', enter_axes)
    fig.canvas.mpl_connect('button_press_event', mouse_button_press)
    fig.canvas.mpl_connect('button_release_event', mouse_button_release)
    fig.canvas.mpl_connect('draw_event', update_rectangles)

    plt.show()



# plot semi-major axis
def plot_semimajor_axis(ax, p1, p2):
    plot_smoothed(ax, p1.time, p1.a, label='1')
    plot_smoothed(ax, p2.time, p2.a, label='2')
    ax.set_xlabel("time")
    ax.set_ylabel("semi-major axis")

# plot period ratio
def plot_period_ratio(ax, p1, p2):
    N = max(len(p1.time), len(p2.time))
    a1 = p1.a[:N]
    a2 = p2.a[:N]
    ratio = a2**1.5/a1**1.5
    t = p1.time[:N]
    plot_smoothed(ax, t, ratio)
    ax.set_xlabel("time")
    ax.set_ylabel("period ratio")

# plot one of the 4 possible mmr fair plots
# see table 1 in paper
# to axis ax
def plot_fair(ax, n, p1, p2, inds=None):
    plot_kw = {"color" : "tab:blue", "alpha" : 0.5 , "markeredgewidth" : 0.0, 'linestyle' : "", 'marker' :  '.'}
    if inds is None:
        inds = [0, len(p1.time)]
    if n == 0:
        X = map_M(p1.M)[inds[0]:inds[1]]
        Y = map_lambda(p2.l, p1.l)[inds[0]:inds[1]]
        ax.plot(X, Y, '.', **plot_kw)
        vert_count = "p+q"
        hor_count = "q"
        ax.set_xlabel("M1")
        ax.set_ylabel("Inner (count {})\n$\lambda_2 - \lambda_1$".format(vert_count))
        ax.set_title("count q")
    elif n == 1:
        X = map_M(p2.M)[inds[0]:inds[1]]
        Y = map_lambda(p2.l, p1.l)[inds[0]:inds[1]]
        ax.plot(X, Y, '.', **plot_kw)
        ax.set_xlabel("M2")
        vert_count = "p"
        hor_count = "q"
        ax.set_ylabel("Inner (count {})\n$\lambda_2 - \lambda_1$".format(vert_count))
    elif n == 2:
        X = map_M(p1.M)[inds[0]:inds[1]]
        Y = map_lambda(p1.l, p2.l)[inds[0]:inds[1]]
        ax.plot(X, Y, '.', **plot_kw)
        ax.set_xlabel("M1")
        vert_count = "p"
        hor_count = "q"
        ax.set_ylabel("Outer (count {})\n$\lambda_1 - \lambda_2$".format(vert_count))
    elif n == 3:
        X = map_M(p2.M)[inds[0]:inds[1]]
        Y = map_lambda(p1.l, p2.l)[inds[0]:inds[1]]
        ax.plot(X, Y, '.', **plot_kw)
        ax.set_xlabel("M2")
        vert_count = "p+q"
        hor_count = "q"
        ax.set_ylabel("Outer (count {})\n$\lambda_1 - \lambda_2$".format(vert_count))
    else:
        raise ValueError("plot type n has to be 0,1,2 or 3")

    #ax.set_xlim([0,2*np.pi])
    #ax.set_ylim([0,2*np.pi])
    ax.set_aspect('equal')
    # add the number of counts as extra labels
    # ax2 = ax.twiny()
    # ax2.set_xlim( ax.get_xlim() )
    # ax2.set_xlabel( hor_count )
    # ax3 = ax.twinx()
    # ax3.set_ylim( ax.get_ylim() )
    # ax3.set_ylabel( vert_count )

# plot one of the 4 possible mmr fair plots
# to axis ax
def plot_resonant_angles(ax, n, p1, p2, p, q, inds=None):
    plot_kw = {"color" : "tab:blue", "alpha" : 1.0 , "markersize" : 2.0 ,"markeredgewidth" : 0.0, 'linestyle' : "", 'marker' :  '.'}
    if inds is None:
        inds = [0, len(p1.time)]
        #inds = slice(0,len(p1.time))
    if n == 0:
        ra = resonant_angle(p, q, p1.l, p2.l, p1.omega_bar, p2.omega_bar)[0]
        plot_select_index(ax, p1.time, ra, inds, **plot_kw)
        ax.set_xlabel("time")
        ax.set_ylabel("$(p+q)\lambda_2 -p\lambda_1 - q (\omega_1+\Omega_1)$")
    elif n == 1:
        ra = resonant_angle(p, q, p1.l, p2.l, p1.omega_bar, p2.omega_bar)[1]
        plot_select_index(ax, p1.time, ra, inds, **plot_kw)
        ax.set_xlabel("time")
        ax.set_ylabel("$(p+q)\lambda_2 -p\lambda_1 - q (\omega_2+\Omega_2)$")
    elif n == 2:
        ra = resonant_angle(p, q, p2.l, p1.l, p2.omega_bar, p1.omega_bar)[1]
        plot_select_index(ax, p1.time, ra, inds, **plot_kw)
        ax.set_xlabel("time")
        ax.set_ylabel("$(p+q)\lambda_1 -p\lambda_2 - q (\omega_1+\Omega_1)$")
    elif n == 3:
        ra = resonant_angle(p, q, p2.l, p1.l, p2.omega_bar, p1.omega_bar)[0]
        plot_select_index(ax, p1.time, ra, inds, **plot_kw)
        ax.set_xlabel("time")
        ax.set_ylabel("$(p+q)\lambda_1 -p\lambda_2 - q (\omega_2+\Omega_2)$")
    else:
        raise ValueError("plot type n has to be 0,1,2 or 3")

    #ax.set_ylim(0, 360)
    ax.set_xlim(p1.time[inds[0]].value, p1.time[inds[1]-1].value)

def plot_select_index(ax, x, y, indminmax, *args, **kwargs):
    x = x[indminmax[0] : indminmax[1]]
    y = y[indminmax[0] : indminmax[1]]
    ax.plot(x, y, *args, **kwargs)

# calculate resonant angles for mean 2:1 motion resonance (p=1, q=1)
# formula taken from Forg´acs-Dajka et al 2018
# "A fast method to identify mean motion resonances"
def resonant_angle(p, q, l1, l2, omega_bar1, omega_bar2):
    return ( ((p+q)*l2 - p*l1 - q*omega_bar1)%(2*np.pi) ,
             ((p+q)*l2 - p*l1 - q*omega_bar2)%(2*np.pi)  )

# FAIR plot
def map_M(M):
    return M
def map_lambda(l1, l2):
    return (l2 - l1)%(2*np.pi)

class Planet:
    def __init__(self, time=None, a=None, Omega=None, omega=None, M=None, omega_bar=None, l=None):
        self.time = time
        self.a = a # semi major axis
        self.Omega = Omega # ascending node
        self.omega = omega # perehelion
        self.M = M # mean anomaly
        self.omega_bar = omega_bar # logitude of perehelion
        self.l = l # mean orbitl longitude

    def cut_to_length(self):
        N = np.inf
        # find the minimum length
        for name in ['time', 'Omega','omega','M','l','omega_bar']:
            if self.__dict__[name] is not None:
                N = min(N, len(self.__dict__[name]))
        # apply the minimum length
        for name in ['time', 'Omega','omega','M','l','omega_bar']:
            self.__dict__[name] = self.__dict__[name][:N]

    def set(self, name, time, value, tomap=False):
        if self.time is None:
            N = len(time)
        else:
            N = min(len(time), len(self.time))
        self.time = time[:N]
        self.__dict__[name] = value[:N]
        if tomap:
            X = self.__dict__[name]
            self.__dict__[name] = (X + (X<0)*2*np.pi)%(2*np.pi)

    def unify_length(self):
        N = len(self.time)
        for var in ['a','Omega','omega','M','omega_bar','l']:
            if len(self.__dict__[var]) > N:
                self.__dict__[var] = self.__dict__[var][:N]

def smooth(x, y, algo="savgol", window_length=51, s=1.3):
    from scipy import interpolate
    import scipy.signal
    # calculate the smoothed values
    if algo == "savgol":
        y_smoothed = scipy.signal.savgol_filter(y, window_length, 3)
        spl = None
    elif algo == "spline":
        spl = interpolate.UnivariateSpline(x, y, s=window_length)
        y_smoothed = spl(x)
    elif 'both':
        y_smoothed = scipy.signal.savgol_filter(y, window_length, 3)
        spl = interpolate.UnivariateSpline(x, y_smoothed, s=s/np.max(np.abs(y_smoothed)))
        y_smoothed = spl(x)
    else:
        raise TypeError("algo must be either 'savgol' or 'spline'")
    return (y_smoothed, spl)

def plot_smoothed(ax, x, y, algo="savgol", label = None, window_length = 51, s = 1.3, alpha=0.5, **kwargs):
    if len(y) == 0:
        return
    y_smoothed, spl = smooth(x,y, algo=algo, window_length=window_length, s=s)
    line, = ax.plot(x, y, alpha=alpha, **kwargs)
    if label is not None:
        kwargs["label"] = label
    kwargs["color"] = line.get_color()
    ax.plot(x, y_smoothed, **kwargs)
    if algo in ['spline', 'both']:
        return spl

if __name__=="__main__":
    main()
