"""
Manual drive:
- W/S (accel/freno), A/D (sterzo) o frecce
- 5 raggi (3 frontali, 2 posteriori)
"""
import numpy as np, matplotlib.pyplot as plt, matplotlib.animation as animation
from matplotlib.patches import Polygon
from envs.parking2d.generators import gen_situation
from envs.parking2d.geometry import rect_corners, make_grid, raycast_grid, CAR_LEN, CAR_WID

def draw_rect(ax, r, **k): cx,cy,L,W,th=r; p=rect_corners(cx,cy,L,W,th); poly=Polygon(p,closed=True,**k); ax.add_patch(poly); return poly
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def main(layout="colonna", seed=123):
    s=gen_situation(layout=layout, seed=seed)
    P=np.vstack([rect_corners(*r) for r in np.vstack([s.bounds, s.obstacles])])
    m=2.0; x0,x1=P[:,0].min()-m,P[:,0].max()+m; y0,y1=P[:,1].min()-m,P[:,1].max()+m
    world=((x0+x1)/2,(y0+y1)/2,(x1-x0),(y1-y0),0.0); grid,meta=make_grid(world, s.obstacles)

    fig,ax=plt.subplots(); ax.set_aspect('equal','box'); ax.grid(True,alpha=.3)
    ax.set_title(f"layout={s.layout} seed={s.seed} — WASD/frecce"); ax.set_xlim(x0,x1); ax.set_ylim(y0,y1)
    [draw_rect(ax,b,fill=False,linewidth=2.0) for b in s.bounds]
    [draw_rect(ax,o,facecolor=(.5,.5,.5),edgecolor=None,alpha=.9) for o in s.obstacles]
    sp=s.spot_pose; draw_rect(ax,(sp[0],sp[1],5.0,2.4,sp[2]),facecolor=(0,1,0,.25),edgecolor=(0,.6,0))

    x,y,th=s.car_start_pose; v=0.0; L=2.6; steer=0.0
    car=draw_rect(ax,(x,y,CAR_LEN,CAR_WID,th),edgecolor=(0,0,1),facecolor=(.35,.6,1,.9),lw=2.0)
    rays=[ax.plot([0,0],[0,0],color="orange",lw=1.5,alpha=.85)[0] for _ in range(5)]
    ang_off=np.array([-0.5,0.0,0.5,np.pi-0.4,np.pi+0.4])

    pressed=set()
    def on_press(e): pressed.add(e.key)
    def on_release(e): pressed.discard(e.key)
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('key_release_event', on_release)

    def tick(_):
        nonlocal x,y,th,v,steer
        fwd = int(('w'in pressed) or ('up'in pressed)) - int(('s'in pressed) or ('down'in pressed))
        turn= int(('a'in pressed) or ('left'in pressed)) - int(('d'in pressed) or ('right'in pressed))
        acc=1.2*fwd; steer=np.clip(steer+0.03*turn, -0.6, 0.6)*0.98
        dt=0.1; v=np.clip(v+(acc-0.4*v)*dt, -2.0, 2.5)
        th=wrap(th+(v/L)*np.tan(steer)*dt); x+=v*np.cos(th)*dt; y+=v*np.sin(th)*dt
        car.set_xy(rect_corners(x,y,CAR_LEN,CAR_WID,th))
        ang=th+ang_off; d=raycast_grid(grid, np.array([x,y]), ang, meta, max_range=15.0)
        for i,a in enumerate(ang):
            ex,ey=x+float(d[i]*15*np.cos(a)), y+float(d[i]*15*np.sin(a))
            rays[i].set_data([x,ex],[y,ey])
        return rays+[car]

    ani = animation.FuncAnimation(fig, tick, interval=33, blit=False)
    plt.show(block=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--layout",choices=["colonna","spina","esse"],default="colonna")
    p.add_argument("--seed",type=int,default=123); a=p.parse_args(); main(a.layout,a.seed)
