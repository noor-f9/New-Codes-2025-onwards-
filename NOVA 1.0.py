# N-body Orbit Visualisation & Analysis tool, written by Noor Alhasani, England, UK. alnoor587@Gmail.com

from numpy import array, float64, zeros_like, concatenate,random,clip,\
sum as np_sum, sqrt, inf,min as np_min, newaxis, fill_diagonal, linalg
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import njit
import time

"User Parameters ---------------------------------------------"

# Visualisation Parameters
ref_frame   = 'earth' # What to centre plot on. Choose 'inertial' or body name [completely case INsensitive]
animate     = 1     # If true, show animation, if false, show static plot
anim_length = 1e1   # Length of animation visualisation [seconds]
tail_length = 5e3   # For animation only
r,g,b,alpha = 1,1,1,0.1 #axes & ticks: colour and transparency

# Integration Parameters 
no_years  = 1e1 # How far to travel into the future 
step_days = 1 # Max integration step (its adaptive). 1 Day recommended, 0.1 to 5 is recommended region, too slow below and unstable above.

labels=["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"] #names of bodies
colors=["lightyellow","darkgray","gold","blue","red","saddlebrown","palegoldenrod","lightblue","blue"] #colours of bodies

#Initial Conds & Masses are from NASA JPL's Highly Accurate Ephemeris Data (18th Dec 2024)

m_sun,    m_mercury, m_venus   = 1.988410*1e30, 3.302000*1e23, 4.868500*1e24
m_earth,  m_mars,    m_jupiter = 5.972190*1e24, 6.417100*1e23, 1.898187*1e27
m_saturn, m_uranus,  m_neptune = 5.683400*1e26, 8.681300*1e25, 1.024090*1e26
masses=array([m_sun,m_mercury,m_venus,m_earth,m_mars,m_jupiter,m_saturn,m_uranus,m_neptune],dtype=float64)

# Initial positions in meters 
initial_positions = array([
    [ 6.878473812672654E+08,  6.323433304868377E+07, -8.286066140004821E+07],  # Sun
    [-4.080524994592670E+10,  2.963697065245490E+10,  6.139770476911131E+09],  # Mercury
    [ 9.564817601040664E+10,  5.206058332912648E+10, -4.848039203416785E+09],  # Venus
    [ 1.042668745649453E+10,  1.469400692033547E+11, -9.156352716957778E+07],  # Earth
    [-5.025028260030842E+10,  2.336219647830772E+11,  6.060636673623607E+09],  # Mars
    [ 1.742895137824861E+11,  7.395869492733686E+11, -7.038835075252473E+09],  # Jupiter
    [ 1.414512381110108E+12, -2.753792898669351E+11, -5.156471836717314E+10],  # Saturn
    [ 1.668617929052008E+12,  2.403453829458348E+12, -1.278356801932704E+10],  # Uranus
    [ 4.470574086838490E+12, -1.014297337587657E+11, -1.009978669361640E+11]   # Neptune
], dtype=float64)

# Initial velocities in meters per second
initial_velocities = array([
    [-1.723688120898555E+02,  1.984194030715345E+03,  8.334172147102781E+01],  # Sun
    [-3.839746532961555E+04, -3.566735673156449E+04,  5.124664962299228E+02],  # Mercury
    [-1.710167546151611E+04,  3.254980951616274E+04,  1.479956710502359E+03],  # Venus
    [-3.036976732880637E+04,  3.848533175039428E+03,  8.385324573891895E+01],  # Earth
    [-2.292972834619696E+04, -1.119452570174142E+03,  5.764158928685126E+02],  # Mars
    [-1.305684202147716E+04,  5.587564440370061E+03,  3.565886050008094E+02],  # Jupiter
    [ 1.132636355264521E+03,  1.145051902711364E+04, -1.332057467927479E+02],  # Saturn
    [-5.830012356249117E+03,  5.555528673823453E+03,  1.696559973452452E+02],  # Uranus
    [-9.594787531217168E+01,  7.457682961240392E+03, -3.063017534960855E+01]   # Neptune
], dtype=float64)

" End of User Parameters--------------------------------------"

t_final, start_year, G = no_years*365.2422*24*3600,2025,6.67430e-11 # m^3 kg^-1 s^-2
r_tol                = 1e3 #min distance [m] before simulation terminates
max_integration_step = step_days*86400  # secs
reference_frame      = ref_frame.title()
anim_length          = int(clip(anim_length,3,120)) #min & max duration of animation in real-time [secs]
tail_length,n_bodies,no_years = int(tail_length/step_days), len(masses), int(no_years)
max_dim              = 100   # Dimensions of plot

@njit
def n_body_system(t, y, n_bodies, G, masses):
    """ODE System for Scipy Integrator."""
    positions           = y[:3 * n_bodies].reshape(n_bodies, 3)
    velocities          = y[3 * n_bodies:].reshape(n_bodies, 3)
    dydt                = zeros_like(y)
    dydt[:3 * n_bodies] = velocities.flatten()
    diff                = positions[None, :, :] - positions[:, None, :]
    dist                = sqrt(np_sum(diff ** 2, axis=2) + 1e-60)
    factor              = (G * masses[None, :] / dist ** 3)[:, :, None]
    acceleration        = np_sum(factor * diff, axis=1)
    dydt[3 * n_bodies:] = acceleration.flatten()
    print('\nProgress %:')
    print(round(100 * t / t_final, 2))
    return dydt

@njit
def close_encounter_event(t, y, n_bodies, r_tol):
    """Event function to terminate simulation on close encounter."""
    positions   = y[:3 * n_bodies].reshape(n_bodies, 3)
    diff        = positions[:, newaxis, :] - positions[newaxis, :, :]
    dist_sq     = np_sum(diff ** 2, axis=2)
    fill_diagonal(dist_sq, inf)
    min_dist    = sqrt(np_min(dist_sq))
    return min_dist - r_tol

def integrate_n_body(initial_positions, initial_velocities, t_eval=None, **kwargs):
    """Integrates the ODE system, using the n_body_system function."""
    close_encounter_event.terminal  = True
    close_encounter_event.direction = -1
    y0 = concatenate((initial_positions.flatten(), initial_velocities.flatten()))
    sol = solve_ivp(lambda t, y: n_body_system(t, y, n_bodies, G, masses),(0, t_final), y0=y0,\
    t_eval=t_eval, dense_output=1,max_step=max_integration_step,atol=1e-12,rtol=1e-9,\
    events=lambda t, y: close_encounter_event(t, y, n_bodies, r_tol))
    times  = sol.t
    states = sol.y.T
    num_time_steps = states.shape[0]
    positions  = states[:, :3 * n_bodies].reshape(num_time_steps, n_bodies, 3)
    velocities = states[:, 3 * n_bodies:].reshape(num_time_steps, n_bodies, 3)
    if sol.t_events is None:
        close_time = array([])
    elif sol.t_events[0].size == 0:
        close_time = array([])
    else:
        close_time = sol.t_events[0]
    return positions, velocities, times, close_time

def plot_trajectories(positions, velocities, times):
    plt.close()
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.legend(fontsize=14)
    if (reference_frame == 'Inertial'):
        xyz = positions[:, :, :] / 1e9  # Calculate all positions at once
    else:
        ref_index = labels.index(reference_frame)
        xyz = (positions[:, :, :] - positions[:, ref_index, :][:, None, :]) / 1e9
    for j in range(n_bodies):
        x, y, z = xyz[:, j, 0], xyz[:, j, 1], xyz[:, j, 2]  # Extract x, y, z for each body
        ax.plot(x, y, z, color=colors[j], alpha=1, linewidth=1)  # Plot the entire trajectory
        ax.plot(x[-1:], y[-1:], z[-1:], "o", color=colors[j], label=labels[j], markersize=15)  # Mark the final position
    total_energies,energy_percentages,time_history = [],[],[]
    for i in range(len(times)):  # Loop over time steps
        v = velocities[i]
        x,y,z = xyz[i, :, 0],xyz[i, :, 1], xyz[i, :, 2]
        total_ke = 0.5 * np_sum(masses[:, newaxis] * np_sum(v * v, axis=1))
        total_gpe = 0
        for j in range(n_bodies):
            for k in range(j + 1, n_bodies):
                xj, yj, zj = x[j], y[j], z[j]
                xk, yk, zk = x[k], y[k], z[k]
                r_jk = linalg.norm([xk - xj, yk - yj, zk - zj])
                total_gpe -= G * masses[j] * masses[k] / r_jk
        total_energy = total_ke + total_gpe
        total_energies.append(total_energy)
        energy_percentage = (total_energy / total_energies[0]) * 100
        energy_percentages.append(energy_percentage)
        current_time = (times[i] / (365.2422 * 24 * 3600)) + start_year
        time_history.append(current_time)
    subplot_x, subplot_y = 0.2, 0.35
    subplot_width, subplot_height = 0.12, 0.1
    ax2 = fig.add_axes([subplot_x, subplot_y, subplot_width, subplot_height])
    ax2.plot(time_history, energy_percentages, color='w')
    ax2.set_xlim(start_year, start_year + no_years)
    ax2.set_ylim(80, 120),ax2.set_xlabel("Time (Year)"),ax2.set_title('Total System Energy %')
    final_year = int(times[-1] / (365.2422 * 24 * 3600)) + start_year  # Calculate final year
    time_text = ax.text2D(-0.2, 0.95, f"Year: {final_year}", transform=ax.transAxes, fontsize=15)
    ax.text2D(0.05, 0.95, time_text)
    ax.set_xlabel('X (Million km)'),ax.set_ylabel('Y (Million km)'),ax.set_zlabel('Z (Million\nkm)')
    ax.set_title('NOVA\nN-body Orbital Visualisation & Analysis tool', fontweight='bold', fontsize=15)
    ax.set_xlim(-max_dim, max_dim),ax.set_ylim(-max_dim, max_dim),ax.set_zlim(-max_dim, max_dim)
    k,num_stars = 100,1e3
    x_stars = random.uniform(-k * max_dim, k * max_dim, int(num_stars))
    y_stars = random.uniform(-k * max_dim, k * max_dim, int(num_stars))
    z_stars = random.uniform(-k * max_dim, k * max_dim, int(num_stars))
    ax.set_aspect('equal'),ax.legend(loc=(1, 0.8))
    ax.xaxis.set_pane_color((r, g, b, alpha)),ax.yaxis.set_pane_color((r, g, b, alpha)),ax.zaxis.set_pane_color((r, g, b, alpha))
    ax.xaxis.line.set_alpha(alpha),ax.yaxis.line.set_alpha(alpha),ax.zaxis.line.set_alpha(alpha)
    ax.xaxis.set_tick_params(color=[r,g,b], labelcolor=[r,g,b], labelsize=8)
    ax.yaxis.set_tick_params(color=[r,g,b], labelcolor=[r,g,b], labelsize=8)
    ax.zaxis.set_tick_params(color=[r,g,b], labelcolor=[r,g,b], labelsize=8)
    ax.xaxis._axinfo["grid"]['color'] = (r, g, b, alpha)
    ax.yaxis._axinfo["grid"]['color'] = (r, g, b, alpha)
    ax.zaxis._axinfo["grid"]['color'] = (r, g, b, alpha)
    ax.grid(color=[r,g,b], linewidth=0.5, alpha=alpha)
    ax.legend(loc=(1,0.7), fontsize=12)
    ax.scatter(x_stars, y_stars, z_stars, c='white', marker='.', s=2)
    plt.show()

def animate_trajectories(positions, velocities, times):
    plt.close(),plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.line.set_alpha(alpha),ax.yaxis.line.set_alpha(alpha),ax.zaxis.line.set_alpha(alpha)
    ax.xaxis.set_tick_params(color=[r,g,b], labelcolor=[r,g,b], labelsize=8)
    ax.yaxis.set_tick_params(color=[r,g,b], labelcolor=[r,g,b], labelsize=8)
    ax.zaxis.set_tick_params(color=[r,g,b], labelcolor=[r,g,b], labelsize=8)
    ax.grid(color=[r,g,b], linewidth=0.5, alpha=alpha)
    subplot_x, subplot_y = 0.2,0.35
    subplot_width, subplot_height = 0.12,0.1
    ax2 = fig.add_axes([subplot_x, subplot_y, subplot_width, subplot_height])  
    total_energy_line, = ax2.plot([], [], color='w')
    energy_percentages,total_energies,time_history,lines = [],[],[],[]
    for i in range(n_bodies):
        line, = ax.plot([], [], [], color=colors[i], alpha=1, linewidth=1)
        lines.append(line)
    points = [ax.plot([], [], [], "o", color=colors[i], label=labels[i], markersize=15)[0] for i in range(n_bodies)]
    ax.set_xlabel('X (Million km)'),ax.set_ylabel('Y (Million km)'),ax.set_zlabel('Z (Million km)')
    ax.set_title('NOVA\nN-body Orbital Visualisation & Analysis tool', fontweight='bold', fontsize=15)
    ax.xaxis.set_pane_color((r, g, b, alpha)),ax.yaxis.set_pane_color((r, g, b, alpha)),ax.zaxis.set_pane_color((r, g, b, alpha))
    ax.xaxis._axinfo["grid"]['color'] = (r, g, b, alpha)
    ax.yaxis._axinfo["grid"]['color'] = (r, g, b, alpha)
    ax.zaxis._axinfo["grid"]['color'] = (r, g, b, alpha)
    ax.set_xlim(-max_dim, max_dim),ax.set_ylim(-max_dim, max_dim),ax.set_zlim(-max_dim, max_dim)
    k,num_stars = 100,1e3
    x_stars = random.uniform(-k * max_dim, k * max_dim, int(num_stars))
    y_stars = random.uniform(-k * max_dim, k * max_dim, int(num_stars))
    z_stars = random.uniform(-k * max_dim, k * max_dim, int(num_stars))
    ax.scatter(x_stars, y_stars, z_stars, c='white', marker='.', s=2)
    ax.set_aspect('equal'),ax.legend(loc=(1, 0.8))
    time_text = ax.text2D(-0.2, 0.9, "", transform=ax.transAxes, fontsize=15)
    momentum_text = ax.text2D(-0.2, 0.80, "", transform=ax.transAxes, fontsize=12)
    ke_text = ax.text2D(-0.2, 0.70, "", transform=ax.transAxes, fontsize=12)
    gpe_text = ax.text2D(-0.2, 0.60, "", transform=ax.transAxes, fontsize=12)
    energy_text = ax.text2D(-0.2, 0.55, "", transform=ax.transAxes, fontsize=12)
    ax.legend(loc=(1,0.7), fontsize=12)
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        total_energy_line.set_data([], [])
        total_energies.clear()
        time_history.clear()
        energy_percentages.clear()
        return lines + points
    def animate(i):
        i = int(i * frame_step_size)
        start_index = int(max(0, i - tail_length))
        if reference_frame == 'Inertial':
            xyz = positions[start_index:i, :, :] / 1e9
        else:
            ref_index = labels.index(reference_frame)
            xyz = (positions[start_index:i, :, :] - positions[start_index:i, ref_index, :][:, None, :]) / 1e9
        if xyz.size == 0:
            return lines + points + [time_text, momentum_text, energy_text, ke_text, gpe_text, total_energy_line]
        x,y,z = xyz[-1, :, 0],xyz[-1, :, 1], xyz[-1, :, 2]
        for j, point in enumerate(points):
            x_j, y_j, z_j = xyz[:, j, 0], xyz[:, j, 1], xyz[:, j, 2]
            lines[j].set_data(x_j, y_j)
            lines[j].set_3d_properties(z_j)
            point.set_data(x_j[-1:], y_j[-1:])
            point.set_3d_properties(z_j[-1:])
        current_time = (times[i] / (365.2422 * 24 * 3600)) + start_year
        time_text.set_text(f"Year: {current_time:.0f}")
        v = velocities[i]
        total_momentum = np_sum(masses[:, newaxis] * v, axis=1)
        total_ke = 0.5 * np_sum(masses[:, newaxis] * np_sum(v * v, axis=1))
        total_gpe = 0
        for body_i in range(n_bodies):
            for body_j in range(body_i + 1, n_bodies):
                xi, yi, zi = x[body_i], y[body_i], z[body_i]
                xj, yj, zj = x[body_j], y[body_j], z[body_j]
                r_ij = linalg.norm([xj - xi, yj - yi, zj - zi])
                total_gpe -= G * masses[body_i] * masses[body_j] / r_ij
        total_energy = total_ke + total_gpe
        momentum_str = "{:.2e}".format(linalg.norm(total_momentum))
        ke_str = "{:.2e}".format(total_ke)
        gpe_str = "{:.2e}".format(total_gpe)
        total_energy_str = "{:.2e}".format(total_energy)
        momentum_text.set_text(f"Momentum:\n{momentum_str} Kg m/s\n")
        ke_text.set_text(f"Kinetic Energy (KE):\n{ke_str} J\n")
        gpe_text.set_text(f"\nGravitational\nPotential (GPE):\n{gpe_str} J\n")
        energy_text.set_text(f"Total (KE + GPE):\n{total_energy_str} J")
        total_energies.append(total_energy)
        energy_percentage = (total_energy/total_energies[0])*100
        energy_percentages.append(energy_percentage)
        time_history.append(current_time)
        total_energy_line.set_data(time_history, energy_percentages) 
        ax2.set_xlim(start_year, start_year + no_years)
        ax2.set_ylim(80,120),ax2.set_xlabel("Time (Year)"),ax2.set_title('Total System Energy %')
        return lines + points + [time_text, momentum_text, energy_text, ke_text, gpe_text, total_energy_line]
    fps = 10
    n_frames = int(anim_length*fps) #number of frames to show
    if len(times) < n_frames:
        n_frames = len(times)
        frame_step_size = 1
    else:
        frame_step_size = int(len(times) // n_frames)
    ani = animation.FuncAnimation(fig, animate, frames=n_frames,init_func=init, interval = int(1e3/fps), blit=False, repeat=True)
    return ani

if ((reference_frame not in labels) and (reference_frame != 'Inertial')): 
    raise Exception("The 'ref_frame variable (completely case IN-sensitive) must be either 'inertial' or the name of one of the bodies!")

if ( ((animate != 0) and (animate != 1)) or (type(animate) != int )):
    raise Exception("The 'animate' variable is a boolean, it MUST be either 0 or 1")
    
else:
    start_time = time.time()
    positions, velocities, times, close_time = integrate_n_body(initial_positions,initial_velocities)
    end_time = time.time()
    tot_time = round(end_time - start_time, 3)
    sim_time = int(times[-1] / (365.2422 * 24 * 3600))
    
    if (close_time.size > 0):
        print("\nSimulation was terminated early due to a close encounter at t = {} years\n".format((close_time[0]/(365.2422 * 24 * 3600))+start_year))        
    elif (sim_time < no_years):
        print("\nSimulation was terminated early due to a close encounter at t = {} years".format(sim_time + start_year))
    else:
        print("\n{} seconds to integrate {} years in time.\n".format(tot_time, sim_time))
    
if animate:
    ani = animate_trajectories(positions, velocities, times)
elif animate == 0:
    plot_trajectories(positions, velocities, times)
else:
    raise Exception("The 'animate' variable is a boolean, it MUST be either 0 or 1")