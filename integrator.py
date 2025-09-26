import xobjects as xo
import xtrack as xt
import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe

class BorisIntegrator:

    isthick = True

    def __init__(self, fieldmap, s_cut, dt, n_steps_max=10000,
                 Bx0=0, By0=0):
        self.fieldmap = fieldmap
        self.s_cut = s_cut
        self.dt = dt
        self.n_steps_max = n_steps_max
        self.ctx = None
        self.length = s_cut
        self.Bx0 = Bx0
        self.By0 = By0

    def _init_kernel(self):
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
        self.ctx = ctx

    def get_field(self, x, y, s):
        Bx, By, Bz = self.fieldmap.get_field(x, y, s)
        Bx += self.Bx0
        By += self.By0
        return Bx, By, Bz

    def track(self, p):

        if self.ctx is None:
            self._init_kernel()

        dt = self.dt
        n_steps = self.n_steps_max
        s_cut = self.s_cut

        x_log = []
        y_log = []
        z_log = []
        px_log = []
        py_log = []
        pp_log = []
        beta_x_log = []
        beta_y_log = []
        beta_z_log = []

        s_start = p.s.copy()
        p.s[:] = 0

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

            Bx, By, Bz = self.get_field(x + vx * dt / 2,
                                        y + vy * dt / 2,
                                        z + vz * dt / 2)

            self.ctx.kernels.boris(
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

        self.x_log = x_log
        self.y_log = y_log
        self.z_log = z_log
        self.px_log = px_log
        self.py_log = py_log
        self.pp_log = pp_log
        self.beta_x_log = beta_x_log
        self.beta_y_log = beta_y_log
        self.beta_z_log = beta_z_log

        # Backtrac to the cut
        ds = p.s - self.s_cut
        pz = np.sqrt((p.p0c*(1 + p.delta))**2 - p.px**2 - p.py**2)
        p.s = s_start + s_cut
        p.x -= p.px / pz * ds
        p.y -= p.py / pz * ds
