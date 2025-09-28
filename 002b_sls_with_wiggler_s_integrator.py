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

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                  energy0=2.4e9)

p = p0.copy()

n_steps = 500
s_cut = 2.4

from integrator_ds import BorisSpatialIntegrator

wig = BorisSpatialIntegrator(fieldmap=mywig, s_cut=s_cut, n_steps=n_steps)

# Field on axis
# s_test = np.linspace(0, 2.4, 1000)
# Bx_init, By_init, Bz_init = wig.get_field(0 * s_test, 0 * s_test, s_test)
# integral_By_init = cumulative_trapezoid(By_init, s_test)
# integral_Bx_init = cumulative_trapezoid(Bx_init, s_test)
# wig.By0 = -integral_By_init [-1] / s_cut
# wig.Bx0 = -integral_Bx_init [-1] / s_cut
# Bx, By, Bz = wig.get_field(0 * s_test, 0 * s_test, s_test)
# integral_By = cumulative_trapezoid(By, s_test)
# integral_Bx = cumulative_trapezoid(Bx, s_test)

env = xt.load('b075_2024.09.25.madx')
line = env.ring
line.particle_ref = p0.copy()

env.elements['wiggler'] = wig
env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
line.insert('wiggler', anchor='center', at=223.8)
line.insert([env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'],
                    at=-0.1, from_='wiggler@start'),
                env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'],
                    at=0.1, from_='wiggler@end'),
                env.new('mark', xt.Marker, at=0.2, from_='wiggler@end')
                ])
line.configure_bend_model(core='mat-kick-mat')
tw_no_wig = line.twiss4d()

# Kicks to be used without integral compensation and n_steps=500
line.vars.update(
{'k0l_corr1': np.float64(0.00018974542686277379),
 'k0l_corr2': np.float64(-0.00031876421147841136),
 'k0sl_corr1': np.float64(2.3978014875526436e-05),
 'k0sl_corr2': np.float64(0.000311552319146383)})

# # Kicks to be used without integral compensation and dt=0.5e-11
# line.vars.update(
# {'k0l_corr1': np.float64(0.00018887390895860494),
#  'k0l_corr2': np.float64(-0.00032214712121279116),
#  'k0sl_corr1': np.float64(2.7222520737453678e-05),
#  'k0sl_corr2': np.float64(0.00033011061736262727)})

# To compute the kicks
# opt = line.match(
#     solve=False,
#     init=tw_no_wig.get_twiss_init(0),
#     only_orbit=True,
#     include_collective=True,
#     vary=xt.VaryList(['k0l_corr1', 'k0l_corr2', 'k0sl_corr1', 'k0sl_corr2'], step=1e-6),
#     targets=xt.TargetSet(x=0, px=0, y=0, py=0., at='mark')
# )
# opt.step(2)

# tw_wig = line.twiss4d(include_collective=True)
tw_wig_open = line.twiss4d(include_collective=True, init=tw_no_wig.get_twiss_init(0))

p_co = tw_wig_open.particle_on_co.copy()
p_co.at_element=0
tw = line.twiss4d(include_collective=True, particle_on_co=p_co,
                  compute_chromatic_properties=False)

delta_chrom = 1e-4
p_co_plus = p_co.copy()
p_co_plus.delta += delta_chrom
p_co_plus.x += tw.dx[0] * delta_chrom
p_co_plus.px += tw.dpx[0] * delta_chrom
p_co_plus.y += tw.dy[0] * delta_chrom
p_co_plus.py += tw.dpy[0] * delta_chrom
p_co_plus.at_element=0
tw_plus = line.twiss4d(include_collective=True,
                       particle_on_co=p_co_plus,
                       compute_chromatic_properties=False)

p_co_minus = p_co_plus.copy()
p_co_minus.delta -= delta_chrom
p_co_minus.x -= tw.dx[0] * delta_chrom
p_co_minus.px -= tw.dpx[0] * delta_chrom
p_co_minus.y -= tw.dy[0] * delta_chrom
p_co_minus.py -= tw.dpy[0] * delta_chrom
p_co_minus.at_element=0
tw_minus = line.twiss4d(include_collective=True,
                        particle_on_co=p_co_minus,
                        compute_chromatic_properties=False)

cols_chrom, scalars_chrom = xt.twiss._compute_chromatic_functions(line, init=None,
                                      delta_chrom=delta_chrom,
                                      steps_r_matrix=None,
                                      matrix_responsiveness_tol=None,
                                      matrix_stability_tol=None,
                                      symplectify=None,
                                      tw_chrom_res=[tw_minus, tw_plus],
                                      on_momentum_twiss_res=tw)

tw._data.update(cols_chrom)
tw._data.update(scalars_chrom)
tw._col_names += list(cols_chrom.keys())