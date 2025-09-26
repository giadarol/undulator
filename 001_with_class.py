from wiggler_class import Wiggler
from scipy import signal
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.integrate import cumulative_trapezoid

from scipy.constants import c as clight
from scipy.constants import e as qe

import xtrack as xt
import xobjects as xo

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

# Class has been successfully tested for the sinusoidal region in the middle.
# The procedure is as follows: We fit the field on-axis (0,0).
# We fit the transverse second derivative at (0,0).
# This is a function of s, which is in turn fitted to a sinusoid.
# We pass the resulting function parameters to bpmeth.
# bpmeth correctly reproduces the data off-axis.

dz = 0.001  # Step size in the z direction for numerical differentiation.

# Create a Wiggler with parameters:
# file_path: Path to the field map file.
# xy_point: (x,y) coordinates of the axis where the field is evaluated.
# dz: Step size in the z direction for numerical differentiation.
#       In this case, dz=0.001 rescales the distances to m instead of mm.
# x_left_slices: Number of slices to the left in the x direction for fitting.
# x_right_slices: Number of slices to the right in the x direction for fitting.
# y_left_slices: Number of slices to the left in the y direction for fitting.
# y_right_slices: Number of slices to the right in the y direction for fitting.
# n_modes_x: Number of modes in the x direction for fitting the sinusoid.
# n_modes_y: Number of modes in the y direction for fitting the sinusoid.
print("FIELDS:")
test_wiggler = Wiggler(file_path='knot_map_test.txt',
                       xy_point=(0, 0),
                       dx=dz,
                       dy=dz,
                       ds=dz,
                       peak_window=(99, 2100),
                       n_modes=[3, 3, 1],
                       enge_deg=[[8, 5], [5, 5], [5, 5]],
                       der=False
)

test_wiggler.set()

test_wiggler_der = Wiggler(file_path='knot_map_test.txt',
                           xy_point=(0, 0),
                           dx=dz,
                           dy=dz,
                           ds=dz,
                           n_modes=[6, 4, 1],
                           enge_deg=[[8, 5], [8, 10], [5, 5]],
                           peak_window=(99, 2100),
                           der=True,
                           filter_params=(None, 2090, 7, 11, 3)
)

test_wiggler_der.set()

Bx_string, Cx = test_wiggler.export_piecewise_string(component="Bx")
By_string, Cy = test_wiggler.export_piecewise_string(component="By")
Bs_string, Cs = test_wiggler.export_piecewise_string(component="Bs")
Bx_der_string, Cx_der = test_wiggler_der.export_piecewise_string(component="Bx")
By_der_string, Cy_der = test_wiggler_der.export_piecewise_string(component="By")

a1 = Bx_string
b1 = By_string
bs = Bs_string

a2 = 0
b2 = 0

a3 = Bx_der_string
b3 = By_der_string

curv=0
wiggler = bp.GeneralVectorPotential(hs=f"{curv}",a=(f"{a1}", f"{a2}", f"{a3}"),b=(f"{b1}", f"{b2}", f"{b3}"), bs=f"{bs}")
Bxfun, Byfun, Bsfun = wiggler.get_Bfield()

class MyWiggler:
    def __init__(self, Bx_fun, By_fun, Bs_fun, s0=0):
        self.Bx_fun = Bx_fun
        self.By_fun = By_fun
        self.Bs_fun = Bs_fun
        self.s0 = s0

    def get_field(self, x, y, s):
        Bx = self.Bx_fun(x, y, s + self.s0)
        By = self.By_fun(x, y, s + self.s0)
        Bs = self.Bs_fun(x, y, s + self.s0)
        return Bx, By, Bs

mywig = MyWiggler(Bxfun, Byfun, Bsfun, s0=-1.2)

# Field on axis
# s_test = np.linspace(0, 2.4, 1000)
# Bx_test = 0 * s_test
# By_test = 0 * s_test
# Bs_test = 0 * s_test

# print("extracting field...")
# for i, s in enumerate(s_test):
#     if i % 10 == 0:
#         print(f"{i}/{len(s_test)}", end='\r', flush=True)
#     Bx_test[i], By_test[i], Bs_test[i] = mywig.get_field(0, 0, s)

# int_By = cumulative_trapezoid(By_test, s_test)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                  energy0=2.4e9)

p = p0.copy()

dt = 1e-11
n_steps = 150
s_cut = 2.4

from integrator import BorisIntegrator

wig = BorisIntegrator(fieldmap=mywig, s_cut=s_cut, dt=dt, n_steps_max=n_steps)


s_test = np.linspace(0, 2.4, 1000)
Bx_test = 0 * s_test
By_test = 0 * s_test
Bs_test = 0 * s_test
print("extracting field...")


env = xt.Environment()
env.elements['wiggler'] = wig

env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.

line = env.new_line(components=[
    env.new('corr1', 'Multipole', knl=['k0l_corr1'], ksl=['k0sl_corr1'], at=0.1),
    env.new('corr2', 'Multipole', knl=['k0l_corr2'], ksl=['k0sl_corr2'], at=0.2),
    env.place('wiggler', anchor='start', at=0.3),
    env.new('exit', 'Marker', at=3)
])
line.particle_ref = p0.copy()
line.twiss_default['include_collective'] = True

tw = line.twiss(betx=10, bety=10)

# opt = line.match(
#     solve=False,
#     betx=1, bety=1,
#     targets=[xt.TargetSet(x=0, px=0, at='exit')],
#     vary=[
#         xt.VaryList(['k0l_corr1', 'k0l_corr2'], step=1e-6)])