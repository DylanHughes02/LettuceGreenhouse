import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

from cmcrameri import cm

from aml_project.common.helper_functions import co2dens2ppm, vaporDens2rh

### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 24

plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('axes', unicode_minus=False)

def create_trajectory_figure(nvars, L, h, c, low_state_constraints, upper_state_constraints):
    """
    Creates base figure for all indoor outdoor variables and control inputs.
    """

    ylabels = [r"Lettuce DM (g/m$^{2}$)", r"CO$_{2, \mathregular{in}}$ (ppm) $\cdot 10^3$", r"T$_{\mathregular{in}}$ ($^\circ$C)", r"RH$_\mathregular{in}$ $(\%)$",\
        r"Sun radiation (W/m$^2$)", r"CO$_{2, \mathregular{out}}$ (ppm) $\cdot 10^3$", r"T$_{\mathregular{out}}$ ($^\circ$C)", r"RH$_\mathregular{out}$ $(\%)$",\
        r"CO$_{2, \mathregular{supply}}$ (mg/m$^2$/s) $\cdot 10^3$", r"Ventilation rate (mm/s)", r"Heating (W/m$^2$)"]

    # create figure
    fig = plt.figure(figsize=(18, 12), dpi=120)
    axes = [fig.add_subplot(3,4, i+1) for i in range(nvars)]

    # x-axis
    t = np.arange(0, L, h)
    x = t/c
    xlabel = "Time (days)"

    # plot state constraints
    if low_state_constraints is not None and upper_state_constraints is not None:
        axes[2].hlines(low_state_constraints[2], x[0], x[-1], linestyle='--', linewidth=3,  alpha=0.5, label='Lower bound')
        axes[2].hlines(upper_state_constraints[2], x[0], x[-1], linestyle='--', linewidth=3,  alpha=0.5, label='Upper bound')
        axes[3].hlines(low_state_constraints[3], x[0], x[-1], linestyle='--', linewidth=3, alpha=0.5)
        axes[3].hlines(upper_state_constraints[3], x[0], x[-1], linestyle='--', linewidth=3, alpha=0.5)

    # set x- and y-labels and limits on x-axis     
    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel(xlabel)

    fig.tight_layout()
    return fig, axes

def plot_multiple_mean_std_trajectories(fig, axes, trajectories_dict, L, h, c, start_day_plot, days2plot, n_per_day, labels):
    fig_lst, axes_lst, handles = [], [], []
    for alg_name, trajectories in trajectories_dict.items():
        fig, axes, traj_handles = plot_mean_std_trajectories(fig, axes, trajectories, L, h, c, start_day_plot, days2plot, n_per_day, alg_name)
        fig_lst.append(fig)
        axes_lst.append(axes)
        
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='lower right', bbox_to_anchor=(1.2, 0), fontsize=10)
    plt.legend(handles, labels)
    return fig_lst, axes_lst


def plot_mean_std_trajectories(fig, axes, trajectories, L, h, c, start_day_plot, days2plot, n_per_day, label):
    """
    Plots the mean and standard deviation of the trajectories.
    Trajectories is a 3D array of shape (n_runs, n_steps, n_vars).
    If n_runs > 1, it plots the mean and standard deviation of the trajectories.
    """
    t = np.arange(0, L, h)[start_day_plot*n_per_day:start_day_plot*n_per_day + n_per_day*days2plot]

    x = t/c

    means = trajectories[:, start_day_plot*n_per_day:start_day_plot*n_per_day + n_per_day*days2plot, :].mean(axis=0)
    std = trajectories[:, start_day_plot*n_per_day:start_day_plot*n_per_day + n_per_day*days2plot, :].std(axis=0)
    handles = []


    for i, var in enumerate(means.T):
        axes[i].step(x, var, linewidth=3)
        axes[i].fill_between(x, var - std[:, i], var + std[:, i], step="pre", alpha=0.3)
        axes[i].set_xlim((start_day_plot, start_day_plot + days2plot))
        


    lines = axes[2].get_lines()
    for line in lines:
        line.set_label(label)
    
    return fig, axes, handles

def plot_trajectory(fig, axes, trajectory, L, h, c, startDay, nDays, n_per_day, label):
    """
    Function that plots all the state and control variables of a trajectory.
    Given axes and figure. Usefull when one wants to create figure with multiple trajectories from different runs.
    """

    # x-axis
    t = np.arange(0, L, h)[startDay*n_per_day:startDay*n_per_day + n_per_day*nDays]
    x = t/c

    for i, var in enumerate(trajectory.T):
        axes[i].step(x, var[startDay*n_per_day:startDay*n_per_day + n_per_day*nDays], linewidth=3)

    lines = axes[2].get_lines()
    # for line in lines:
    lines[-1].set_label(label)
    return fig, axes

#TODO: broken
def plot_multiple_greenhouse_traj(values, L, h, c=86400, labels=[], test_traj=False, test_values=None, low_constraints=None, high_constraints=None, high_humid_constraints=None):
    """
    Function that plots greenhouse trajectories:
    - observations
    - outdoor weather
    - actions

    Args:
        values      - 3D array of several greenhouse trajectories, can come from 1 to N experiments
        L           - end of the simulation time
        h           - size of time steps
        test_traj   - boolean to specify whether test trajectory is plotted
        test_values - the test trajectories to visaulise
    """
    cmap = cm.berlin
    t = np.arange(0, L, h)[:values.shape[1]]
    ylabels = [r"Lettuce DM (g/m$^{2}$)", r"CO$_{2, \mathregular{in}}$ (ppm) $\cdot 10^3$", r"T$_{\mathregular{in}}$ ($^\circ$C)", r"RH$_\mathregular{in}$ $(\%)$",\
        r"Sun radiation (W/m$^2$)", r"CO$_{2, \mathregular{out}}$ (ppm) $\cdot 10^3$", r"T$_{\mathregular{out}}$ ($^\circ$C)", r"RH$_\mathregular{out}$ $(\%)$",\
        r"CO$_{2, \mathregular{ref}}$ (ppm) $\cdot 10^3$", r"Ventilation rate (mm/s)", r"T$_{\mathregular{ref}}$ ($^\circ$C)"]

    # convert CO2 concentration to parts per million (ppm)
    values[:,:,5] = co2dens2ppm(values[:,:,6], values[:,:,5])*1e-3
    # convert vapour pressure deficit to relative humidity
    values[:,:,7] = vaporDens2rh(values[:,:,6], values[:,:,7])

    xlabel = "Time (days)"
    fig = plt.figure(figsize=(18, 12), dpi=120)
    axes = [fig.add_subplot(3,4, i+1) for i in range(values.shape[-1])]

    for n_run, trajectories in enumerate(values):
        for i, trajectory in enumerate(trajectories.T):
            ax = axes[i]
            ax.step(t/c, trajectory, linewidth=2, c=cmap(n_run/values.shape[0]))
            ax.set_ylabel(ylabels[i])
            ax.set_xlabel(xlabel)
            ax.set_xlim((0,2))

        ax.xaxis.set_major_locator(LinearLocator(5))

    lines = axes[-1].get_lines()
    for i, line in enumerate(lines):
        line.set_label(labels[i])

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, .05), fontsize=10)
    plt.tight_layout() # Or equivalently,  "plt.tight_layout()"
    return fig, axes
