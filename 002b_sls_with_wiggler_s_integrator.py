from wiggler_class import Wiggler
from scipy import signal
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt


delta_chrom = 1e-4
# deltas = [-2*delta_chrom, -delta_chrom, 0, delta_chrom, 2*delta_chrom]
deltas = [-delta_chrom, delta_chrom]

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
                       poly_deg=[[4, 4], [4, 4], [4, 4]],
                       poly_pieces=[[19, 35], [19, 36], [8, 8]],
                       der=False
                       )

test_wiggler.set()
#Test_Wiggler.plot_fields()
#Test_Wiggler.plot_integrated_fields()

print("DERIVATIVES:")
test_wiggler_der = Wiggler(file_path='knot_map_test.txt',
                           xy_point=(0, 0),
                           dx=dz,
                           dy=dz,
                           ds=dz,
                           n_modes=[6, 4, 1],
                           poly_deg=[[4, 4], [4, 4], [4, 4]],
                           poly_pieces=[[15, 15], [15, 15], [15, 15]],
                           peak_window=(99, 2100),
                           der=True,
                           filter_params=(None, 2090, 7, 11, 3)
                           )

test_wiggler_der.set()

Bx_string = test_wiggler.export_piecewise_sympy(field="Bx")
Bx_der_string = test_wiggler_der.export_piecewise_sympy(field="Bx")
By_string = test_wiggler.export_piecewise_sympy(field="By")
By_der_string = test_wiggler_der.export_piecewise_sympy(field="By")
Bs_string = test_wiggler.export_piecewise_sympy(field="Bs")

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
        self.scale = 1.  # Scale to meters

    def get_field(self, x, y, s):
        Bx = self.scale * self.Bx_fun(x, y, s + self.s0)
        By = self.scale * self.By_fun(x, y, s + self.s0)
        Bs = self.scale * self.Bs_fun(x, y, s + self.s0)
        return Bx, By, Bs

mywig = MyWiggler(Bxfun, Byfun, Bsfun, s0=-1.1)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                  energy0=2.4e9)

p = p0.copy()

n_steps = 1000
l_wig = 2.2
n_slices = 1000

s_cuts = np.linspace(0, l_wig, n_slices + 1)
s_mid = 0.5 * (s_cuts[:-1] + s_cuts[1:])

wig_slices = []
for ii in range(n_slices):
    wig = xt.BorisSpatialIntegrator(fieldmap_callable=mywig.get_field, s_start=s_cuts[ii], s_end=s_cuts[ii + 1],
                                 n_steps=np.round(n_steps / n_slices).astype(int),
                                 verbose=True)
    wig_slices.append(wig)

Bx_mid, By_mid, Bs_mid = wig_slices[0].fieldmap_callable(0, 0, s_mid)

env = xt.load('b075_2024.09.25.madx')
line = env.ring
line.configure_bend_model(core='mat-kick-mat')
line.particle_ref = p0.copy()

for ii in range(n_slices):
    env.elements[f'wigslice_{ii}'] = wig_slices[ii]
wiggler = env.new_line(components=['wigslice_' + str(ii) for ii in range(n_slices)])

env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.
env['on_wig_corr'] = 1.0

env.new('corr1', xt.Multipole, knl=['on_wig_corr * k0l_corr1'], ksl=['on_wig_corr * k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['on_wig_corr * k0l_corr2'], ksl=['on_wig_corr * k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['on_wig_corr * k0l_corr3'], ksl=['on_wig_corr * k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['on_wig_corr * k0l_corr4'], ksl=['on_wig_corr * k0sl_corr4'])

wiggler.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)
wiggler.particle_ref = line.particle_ref

# Computed for 1000 slices, 1000 steps
env.vars.update(
{'k0l_corr1': np.float64(-0.0004540792291112204),
 'k0sl_corr1': np.float64(-1.213769189237666e-06),
 'k0l_corr2': np.float64(0.0008135172335552242),
 'k0sl_corr2': np.float64(0.00023470961164860475),
 'k0l_corr3': np.float64(-0.0001955197609031625),
 'k0sl_corr3': np.float64(-0.00021394733008765638),
 'k0l_corr4': np.float64(-0.00015806879956816854),
 'k0sl_corr4': np.float64(3.370506139561265e-05)})

# # To compute the kicks
# opt = wiggler.match(
#     solve=False,
#     betx=0, bety=0,
#     only_orbit=True,
#     include_collective=True,
#     vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
#                       'k0l_corr2', 'k0sl_corr2',
#                       'k0l_corr3', 'k0sl_corr3',
#                       'k0l_corr4', 'k0sl_corr4',
#                       ], step=1e-6),
#     targets=[
#         xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
#         xt.TargetSet(x=0, y=0, at='wiggler_167'),
#         xt.TargetSet(x=0, y=0, at='wiggler_833')
#         ],
# )
# opt.step(2)

print('Twiss wiggler only')
tw_wig_only = wiggler.twiss(include_collective=True, betx=1, bety=1)

wiggler_places = [
    'ars02_uind_0500_1',
    'ars03_uind_0380_1',
    'ars04_uind_0500_1',
    'ars05_uind_0650_1',
    'ars06_uind_0500_1',
    'ars07_uind_0200_1',
    'ars08_uind_0500_1',
    'ars09_uind_0790_1',
    'ars11_uind_0210_1',
    'ars11_uind_0610_1',
    'ars12_uind_0500_1',
]

tt = line.get_table()
for wig_place in wiggler_places:
    line.insert(wiggler, anchor='start', at=tt['s', wig_place])

env['on_wig_corr'] = 0
mywig.scale = 0
tw_no_wig = line.twiss4d(strengths=True)
tw_vs_momentum_no_wig = {}
for dd in deltas:
    tw_vs_momentum_no_wig[dd] = line.twiss4d(delta0=dd,
                                         compute_chromatic_properties=False)

env['on_wig_corr'] = 1.0
mywig.scale = 1.0

print('Twiss full line with wiggler')
p_co = tw_no_wig.particle_on_co.copy()
p_co.at_element=0

tw = line.twiss4d(include_collective=True, particle_on_co=p_co,
                  compute_chromatic_properties=False)



tw_vs_momentum = {}
for delta in deltas:
    print(f'Twiss off momentum, delta = {delta}')
    p_off = p_co.copy()
    p_off.delta += delta
    p_off.x += tw.dx[0] * delta
    p_off.px += tw.dpx[0] * delta
    p_off.y += tw.dy[0] * delta
    p_off.py += tw.dpy[0] * delta
    p_off.at_element=0
    tw_vs_momentum[delta] = line.twiss4d(include_collective=True,
                                         particle_on_co=p_off,
                                         compute_chromatic_properties=False)


cols_chrom, scalars_chrom = xt.twiss._compute_chromatic_functions(line, init=None,
                                      delta_chrom=delta_chrom,
                                      steps_r_matrix=None,
                                      matrix_responsiveness_tol=None,
                                      matrix_stability_tol=None,
                                      symplectify=None,
                                      tw_chrom_res=[tw_vs_momentum[-delta_chrom],
                                                    tw_vs_momentum[delta_chrom]],
                                      on_momentum_twiss_res=tw)

tw._data.update(cols_chrom)
tw._data.update(scalars_chrom)
tw._col_names += list(cols_chrom.keys())

dl = np.diff(s_cuts)
wig_mult_places = []
for ii, (bbx, bby) in enumerate(zip(Bx_mid, By_mid)):
    nn = f'wig_mult_{ii}'
    pp = env.new(nn, xt.Multipole,
                 knl=[dl[ii] * bby / p0.rigidity0[0]],
                 ksl=[dl[ii] * bbx / p0.rigidity0[0]],
                 at=s_mid[ii])
    wig_mult_places.append(pp)

wiggler_mult = wiggler.copy(shallow=True)
tt_slices = wiggler.get_table().rows['wigslic.*']

wiggler_mult.remove(tt_slices.name)
wiggler_mult.insert(wig_mult_places)

tw_wig_mult = wiggler_mult.twiss(betx=1, bety=1)




# qx_vs_delta = [tt.qx for tt in tw_vs_momentum.values()]
# qy_vs_delta = [tt.qy for tt in tw_vs_momentum.values()]
# qx_vs_delta_no_wig = [tt.qx for tt in tw_vs_momentum_no_wig.values()]
# qy_vs_delta_no_wig = [tt.qy for tt in tw_vs_momentum_no_wig.values()]

# plt.close('all')
# plt.figure(1)
# plt.plot(deltas, qx_vs_delta-tw.dqx*np.array(deltas)-tw.qx)
# plt.plot(deltas, qx_vs_delta_no_wig-tw_no_wig.dqx*np.array(deltas)-tw_no_wig.qx, '--')
