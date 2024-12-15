import numpy as np
import warp as wp



dt = wp.constant(1/60)
width = wp.constant(640)
height = wp.constant(640)


@wp.func
def get_particle_w_bnd(s: wp.array2d(dtype=wp.vec3), ix: int, iy: int):
    ix = wp.clamp(ix, 0, width - 1)
    iy = wp.clamp(iy, 0, height - 1)
    return s[ix, iy]

@wp.func
def get_pressure_w_bnd(s: wp.array2d(dtype=float), ix: int, iy: int):
    ix = wp.clamp(ix, 0, width - 1)
    iy = wp.clamp(iy, 0, height - 1)
    return s[ix, iy]

@wp.func
def get_vel_w_bnd(u: wp.array2d(dtype=wp.vec2), ix: int, iy: int):
    if ix < 0 or ix >= width:
        return wp.vec2(0.0, 0.0)
    if iy < 0 or iy >= height:
        return wp.vec2(0.0, 0.0)
    return u[ix, iy]

@wp.func
def get_prev_velocity(u: wp.array2d(dtype=wp.vec2), prev_pos: wp.vec2):
    # index component of xy
    ix = int(wp.floor(prev_pos[0]))
    iy = int(wp.floor(prev_pos[1]))
    # decimal component of x
    dx = prev_pos[0] - float(ix)
    dy = prev_pos[1] - float(iy)
    
    velx0y0 = get_vel_w_bnd(u, ix, iy)
    velx1y0 = get_vel_w_bnd(u, ix+1, iy)
    velx0y1 = get_vel_w_bnd(u, ix, iy+1)
    velx1y1 = get_vel_w_bnd(u, ix+1, iy+1)
    z1 = wp.lerp(velx0y0, velx1y0, dx)
    z2 = wp.lerp(velx0y1, velx1y1, dx)
    return wp.lerp(z1, z2, dy)


@wp.func
def get_prev_particle(u: wp.array2d(dtype=wp.vec3), prev_pos: wp.vec2):
    # index component of xy
    ix = int(wp.floor(prev_pos[0]))
    iy = int(wp.floor(prev_pos[1]))
    
    # decimal component of x
    dx = prev_pos[0] - float(ix)
    dy = prev_pos[1] - float(iy)
    
    ptx0y0 = get_particle_w_bnd(u, ix, iy)
    ptx1y0 = get_particle_w_bnd(u, ix+1, iy)
    ptx0y1 = get_particle_w_bnd(u, ix, iy+1)
    ptx1y1 = get_particle_w_bnd(u, ix+1, iy+1)

    z1 = wp.lerp(ptx0y0, ptx1y0, dx)
    z2 = wp.lerp(ptx0y1, ptx1y1, dx)
    return wp.lerp(z1, z2, dy)

@wp.kernel
def pressure_step(p: wp.array2d(dtype=float), 
                   p_prev: wp.array2d(dtype=float), 
                   div: wp.array2d(dtype=float)):
    
    i, j = wp.tid()

    p0 = get_pressure_w_bnd(p_prev, i - 1, j)
    p1 = get_pressure_w_bnd(p_prev, i + 1, j)
    p2 = get_pressure_w_bnd(p_prev, i, j - 1)
    p3 = get_pressure_w_bnd(p_prev, i, j + 1)

    p[i,j] = (p0 + p1 + p2 + p3 - div[i, j])*0.25


# Advection step
@wp.kernel
def advect(u_prev: wp.array2d(dtype=wp.vec2),
           u: wp.array2d(dtype=wp.vec2),
           s_prev: wp.array2d(dtype=wp.vec3),
           s: wp.array2d(dtype=wp.vec3)):
    # Get the current thread index
    i, j = wp.tid()
     
    # Get the previous position
    prev_pos = wp.vec2(float(i), float(j)) - u_prev[i, j]*dt*0.5

    u[i, j] = get_prev_velocity(u_prev, prev_pos)
    s[i, j] = get_prev_particle(s_prev, prev_pos)


@wp.kernel
def external_forces(velocity_field: wp.array2d(dtype=wp.vec2), scalar_field: wp.array2d(dtype=wp.vec3), dt: float):
    i, j = wp.tid()
    f_g = wp.vec2(-100.0, 0.0) * wp.length(scalar_field[i, j])
    velocity_field[i, j] = velocity_field[i, j] + dt * f_g


@wp.kernel
def pressure_apply(p: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2)):
    i, j = wp.tid()
    if i == 0 or i == width - 1:
        return
    if j == 0 or j == height - 1:
        return
    f_p = wp.vec2(p[i + 1, j] - p[i - 1, j], p[i, j + 1] - p[i, j - 1]) * 0.5

    u[i, j] = u[i, j] - f_p


@wp.kernel
def divergence(u: wp.array2d(dtype=wp.vec2), div: wp.array2d(dtype=float)):
    i, j = wp.tid()

    if i == width - 1:
        return
    if j == height - 1:
        return

    dx = (u[i + 1, j][0] - u[i, j][0]) * 0.5
    dy = (u[i, j + 1][1] - u[i, j][1]) * 0.5

    div[i, j] = dx + dy
    
@wp.kernel
def sources(scalar_field: wp.array2d(dtype=wp.vec3), velocity_field: wp.array2d(dtype=wp.vec2), radius: int, dir: wp.vec2):
    i, j = wp.tid()

    top_source = wp.sqrt(wp.pow(float(i - (width - 20)), 2.0) + wp.pow(float(j - (height - 20)), 2.0))
    bottom_source = wp.length(wp.vec2(float(i - 20), float(j - 20)))
    
    if top_source < (radius + 3):
        scalar_field[i, j] = wp.vec3(0.0, 0.0, 1.0)
        velocity_field[i, j] = dir
        
    if bottom_source < radius:
        scalar_field[i, j] = wp.vec3(1.0, 0.0, 0.0)
        velocity_field[i, j] = -dir
        
class Fluid():
    def __init__(self):
        self.width = width
        self.height = height
        self.size = (height, width)
        self.u = wp.zeros(self.size, dtype=wp.vec2)
        self.u_prev = wp.zeros(self.size, dtype=wp.vec2)
        self.s = wp.zeros(self.size, dtype=wp.vec3)
        self.s_prev = wp.zeros(self.size, dtype=wp.vec3)
        self.p = wp.zeros(self.size, dtype=float)
        self.p_prev = wp.zeros(self.size, dtype=float)
        self.div = wp.zeros(self.size, dtype=float)
        
        self.occ_grip = wp.zeros(self.size, dtype=bool)
        
        self.sim_time = 0.0

    def step(self):
        for i in range(2):
            shape = (self.width, self.height)

            speed = 500.0
            angle = 4.0
            vel = wp.vec2(np.cos(angle) * speed, np.sin(angle) * speed)

            wp.launch(sources, dim=shape, inputs=[self.s, self.u, 10, vel])

            wp.launch(external_forces, dim=shape, inputs=[self.u, self.s, dt/2])
            wp.launch(divergence, dim=shape, inputs=[self.u, self.div])

            self.p.zero_()
            self.p_prev.zero_()

            for j in range(400):
                wp.launch(pressure_step, dim=shape, inputs=[self.p, self.p_prev, self.div])
                (self.p, self.p_prev) = (self.p_prev, self.p)
                
            wp.launch(pressure_apply, dim=shape, inputs=[self.p, self.u])

            wp.launch(advect, dim=shape, inputs=[self.u, self.u_prev, self.s, self.s_prev])
            (self.u, self.u_prev) = (self.u_prev, self.u)
            (self.s, self.s_prev) = (self.s_prev, self.s)

            self.sim_time += dt

    def step_and_render_frame(self, frame_num=None, img=None):
        self.step()

        if img:
            img.set_array(self.s.numpy())

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=100000, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    #with wp.ScopedDevice(args.device):
    fluid = Fluid()

    import matplotlib
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt

    fig = plt.figure()

    img = plt.imshow(
        fluid.s.numpy(),
        origin="lower",
        animated=True,
        interpolation="antialiased",
    )
    img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
    seq = anim.FuncAnimation(
        fig,
        fluid.step_and_render_frame,
        fargs=(img,),
        frames=args.num_frames,
        blit=True,
        interval=8,
        repeat=False,
    )

    plt.show()
