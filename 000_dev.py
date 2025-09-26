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

s_test = np.linspace(0, 2.4, 1000)
Bx_test = 0 * s_test
By_test = 0 * s_test
Bs_test = 0 * s_test

print("extracting field...")
for i, s in enumerate(s_test):
    if i % 10 == 0:
        print(f"{i}/{len(s_test)}", end='\r', flush=True)
    Bx_test[i], By_test[i], Bs_test[i] = mywig.get_field(0, 0, s)

int_By = cumulative_trapezoid(By_test, s_test)

ctx = xo.ContextCpu()

boris_knl_description = xo.Kernel(
    c_name='boris_step',
    args=[
        xo.Arg(xo.Int64,   name='N_sub_steps'),
        xo.Arg(xo.Float64, name='Dtt'),
        xo.Arg(xo.Float64, name='B_field', pointer=True),
        xo.Arg(xo.Float64, name='B_skew', pointer=True),
        xo.Arg(xo.Float64, name='xn1', pointer=True),
        xo.Arg(xo.Float64, name='yn1', pointer=True),
        xo.Arg(xo.Float64, name='zn1', pointer=True),
        xo.Arg(xo.Float64, name='vxn1', pointer=True),
        xo.Arg(xo.Float64, name='vyn1', pointer=True),
        xo.Arg(xo.Float64, name='vzn1', pointer=True),
        xo.Arg(xo.Float64, name='Ex_n', pointer=True),
        xo.Arg(xo.Float64, name='Ey_n', pointer=True),
        xo.Arg(xo.Float64, name='Bx_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='By_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='Bz_n_custom', pointer=True),
        xo.Arg(xo.Int64,   name='custom_B'),
        xo.Arg(xo.Int64,   name='N_mp'),
        xo.Arg(xo.Int64,   name='N_multipoles'),
        xo.Arg(xo.Float64, name='charge'),
        xo.Arg(xo.Float64, name='mass', pointer=True),
    ],
)

ctx.add_kernels(
    kernels={'boris': boris_knl_description},
    sources=[xt._pkg_root / '_temp/boris_and_solenoid_map/boris.h'],
)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                  energy0=2.4e9)

p = p0.copy()

dt = 1e-11
n_steps = 1500
s_cut = 2.4

x_log = []
y_log = []
z_log = []
px_log = []
py_log = []
pp_log = []
beta_x_log = []
beta_y_log = []
beta_z_log = []

for ii in range(n_steps):

    print(f"{ii}/{n_steps}", end='\r', flush=True)

    mask_inside = (p.s < s_cut) | (p.s < -1e-10)
    if not np.any(mask_inside):
        break

    mass0 = p.mass0
    q0 = p.q0
    p0c = p.p0c[mask_inside]
    x = p.x[mask_inside].copy()
    y = p.y[mask_inside].copy()
    z = p.s[mask_inside].copy()
    px = p.px[mask_inside].copy()
    py = p.py[mask_inside].copy()
    delta = p.delta[mask_inside].copy()
    energy = p.energy[mask_inside].copy()

    gamma = energy / mass0
    mass0_kg = mass0 * qe / clight**2
    charge0_coulomb = q0 * qe

    p0c_J = p0c * qe

    Pxc_J = px * p0c_J
    Pyc_J = py * p0c_J
    Pzc_J = np.sqrt((p0c_J*(1 + delta))**2 - Pxc_J**2 - Pyc_J**2)

    vx = Pxc_J / clight / (gamma * mass0_kg) # m/s
    vy = Pyc_J / clight / (gamma * mass0_kg) # m/s
    vz = Pzc_J / clight / (gamma * mass0_kg) # m/s

    Bx, By, Bz = mywig.get_field(x + vx * dt / 2,
                                y + vy * dt / 2,
                                z + vz * dt / 2)

    ctx.kernels.boris(
            N_sub_steps=1,
            Dtt=dt,
            B_field=np.array([0.]),
            B_skew=np.array([0.]),
            xn1=x,
            yn1=y,
            zn1=z,
            vxn1=vx,
            vyn1=vy,
            vzn1=vz,
            Ex_n=0 * x,
            Ey_n=0 * x,
            Bx_n_custom=Bx,
            By_n_custom=By,
            Bz_n_custom=Bz,
            custom_B=1,
            N_mp=len(x),
            N_multipoles=0,
            charge=charge0_coulomb,
            mass=mass0_kg * gamma,
    )

    p.x[mask_inside] = x
    p.y[mask_inside] = y
    p.s[mask_inside] = z
    p.px[mask_inside] = mass0_kg * gamma * vx * clight / p0c_J
    p.py[mask_inside] = mass0_kg * gamma * vy * clight / p0c_J
    pz = mass0_kg * gamma * vz * clight / p0c_J
    pp = np.sqrt(p.px**2 + p.py**2 + pz**2)

    beta_x_after = vx / clight
    beta_y_after = vy / clight
    beta_z_after = vz / clight

    x_log.append(p.x.copy())
    y_log.append(p.y.copy())
    z_log.append(p.s.copy())
    px_log.append(p.px.copy())
    py_log.append(p.py.copy())
    pp_log.append(pp)
    beta_x_log.append(beta_x_after)
    beta_y_log.append(beta_y_after)
    beta_z_log.append(beta_z_after)
