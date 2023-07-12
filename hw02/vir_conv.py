import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc
import matplotlib.gridspec as gridspec

def x(t):
    # x_t = 0.42 + 0.5*np.cos(t)+0.08*np.cos(2*t)
    x_t = np.cos(t) + np.cos(2*t) - 1/2*np.cos(3*t) - 3/2*np.cos(4*t) + 2.75
    return x_t

def h(t):
    h_t = 1*(t>-0.5) - 1*(t>0.5)
    return h_t

def g(t):
    # mask = 1*(t>-0.5) - 1*(t>0.5)
    # f_abs = -np.abs(t) + 1
    # g_t = f_abs*mask
    g_t = (0.42 + 0.5*np.cos(t)+0.08*np.cos(2*t))/2
    return g_t 



def ht(t,d):
    h_t = 1*(t>(-2.5+d/10)) - 1*(t>(-1.5+d/10))
    return h_t

def gt(t,d):
    # mask = 1*(t>-2.5+d/10) - 1*(t>(-1.5+d/10))
    # f_abs = -np.abs(t+2-d/10) + 1
    # f_abs = -np.abs(t+2-d/10) + 1
    # g_t = f_abs*mask
    g_t = (0.42 + 0.5*np.cos(t-2+d/10)+0.08*np.cos(2*(t-2+d/10)))/2
    return g_t


def Conv_Animation(t,x_t,h_t,g_t):
    y_t = np.convolve(x_t,h_t)
    fig = plt.figure(figsize = (12,24))
    G   = gridspec.GridSpec(2, 9)
    ax1 = plt.subplot(G[0, :4])
    ax1.plot(t,x_t,"o", label = "x(t)",color = 'green')
    ax1.grid()
    line, = ax1.plot(t, h_t ,"o", label = "h(t)", color = 'blue')

    step = t+2
    ax2 = plt.subplot(G[1, :4])
    ax2.plot(step,y_t[(range(19,59))],"o", label = "y(t)", color = 'yellow')
    ax2.grid()
    ax2.set_xlim([-0.1,4.1])
    ax2.set_ylim([-0.1,45.2])
    area, = ax2.plot(t,y_t[(range(24,64))],"o", label = "y(t)", color = 'red')

    y2_t = np.zeros_like(x_t)
    for k in range(40):
        y2_t[k] = np.sum(gt(t,k)*x_t)
    ax3 = plt.subplot(G[0, 5:])
    ax3.plot(t,x_t,"o", label = "x(t)",color = 'green')
    ax3.grid()
    line2, = ax3.plot(t, h_t ,"o", label = "g(t)", color = 'blue')
    ax3.legend()

    step = t+2
    ax4 = plt.subplot(G[1, 5:])
    ax4.plot(step,y2_t,"o", label = "y(t)", color = 'yellow')
    ax4.grid()
    ax4.set_xlim([-0.1,4.1])
    ax4.set_ylim([-0.1,45.2])
    area2, = ax4.plot(t,y_t[(range(20,60))],"o", label = "y(t)", color = 'red')
    ax4.legend()
    
    def init():
        line.set_data([], [])
        area.set_data([],[])
        line2.set_data([], [])
        area2.set_data([],[])
        line.set_label("a")
        
        return (line,area,line2,area2,)
    def animate(i):
        hx = ht(t,i)
        line.set_data(t, hx)
        line.set_label("a")

        gx = gt(t,i)
        line2.set_data(t, gx)
        line2.set_label("a")


        yx = np.sum(hx*x_t)
        y_plot = np.zeros_like(x_t) - 1
        y_plot[i] = yx
        area.set_data(step, y_plot)

        yx2 = np.sum(gx*x_t)
        y_plot = np.zeros_like(x_t) - 1
        y_plot[i] = yx2
        area2.set_data(step, y_plot)
        return (line, area, line2, area2 )
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=40, interval=300, blit=True)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.show()

def main():
    t = np.linspace(-2,2,40)
    x_t = x(t)
    h_t = h(t)
    g_t = g(t)
    fig = plt.figure(figsize = (12,24))
    G   = gridspec.GridSpec(2, 9)
    ax0 = plt.subplot(G[0, :])
    ax1 = plt.subplot(G[1, :4])
    ax2 = plt.subplot(G[1, 5:])
    ax0.plot(t,x_t,'o',color='blue',label='x(t)')
    ax1.plot(t,h_t,'o',color='red',label='h(t)')
    ax2.plot(t,g_t,'o',color='yellow',label='g(t)')
    ax0.grid()
    ax1.grid()
    ax2.grid()
    ax0.legend()
    ax1.legend()
    ax2.legend()
    plt.show()
    y_t = np.convolve(x_t,h_t)
    Conv_Animation(t,x_t,h_t,g_t)

if __name__ == "__main__":
    main()