import xobjects as xo
import xtrack as xt
import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe

class BorisSpatialIntegrator:

    isthick = True

    def __init__(self, fieldmap, s_start, s_end, n_steps,
                 Bx0=0, By0=0):
        self.fieldmap = fieldmap
        self.s_start = s_start
        self.s_end = s_end
        self.ds = (s_end - s_start) / n_steps
        self.n_steps = n_steps
        self.ctx = None
        self.length = s_end - s_start
        self.Bx0 = Bx0
        self.By0 = By0

    def get_field(self, x, y, s):
        Bx, By, Bz = self.fieldmap.get_field(x, y, s)
        Bx += self.Bx0
        By += self.By0
        return Bx, By, Bz

    def track(self, p):

        x_log = []
        y_log = []
        z_log = []
        s_in = p.s.copy()
        p.s=self.s_start

        for ii in range(self.n_steps):

            print(f's = {p.s[0]:.3f}           ', end='\r', flush=True)

            q0 = p.q0
            x = p.x.copy()
            y = p.y.copy()
            z = p.s.copy()
            px = p.px.copy()
            py = p.py.copy()
            delta = p.delta.copy()
            energy = p.energy.copy()
            p0c = p.p0c

            charge0_coulomb = q0 * qe
            energy_J = energy * qe

            p0_J = p0c * qe / clight # in kg m/s

            Px_J = px * p0_J
            Py_J = py * p0_J

            w = np.zeros((Px_J.shape[0], 3), dtype=Px_J.dtype)
            w[:, 0] = Px_J
            w[:, 1] = Py_J
            w[:, 2] = energy_J / clight  # E/c

            x_new, y_new, z_new, w_new = step_spatial_boris_B(x, y, z, w,
                charge0_coulomb, self.ds,
                field_fn=self.get_field)
            p.x = x_new.copy()
            p.y = y_new.copy()
            p.s = z_new.copy()
            p.px = w_new[:, 0] / p0_J
            p.py = w_new[:, 1] / p0_J

            x_log.append(p.x.copy())
            y_log.append(p.y.copy())
            z_log.append(p.s.copy())
        p.s = s_in + self.length
        self.x_log = np.array(x_log)
        self.y_log = np.array(y_log)
        self.z_log = np.array(z_log)

import numpy as np
c = 299_792_458.0  # m/s

def step_spatial_boris_B(x, y, z, w, q, dz, field_fn):
    """
    Spatial Boris step for magnetic fields only (E = 0),
    computing pz from total energy U and transverse momenta.

    Parameters
    ----------
    x, y, z : (N,) arrays
        Particle positions [m]
    w : (N, 3) array
        (px, py, U/c)
    q : float
        Particle charge [C]
    dz : float
        Step in z [m]
    field_fn : callable(x, y, z) -> (Bx, By, Bz)
        Magnetic field function [T]
    m : float
        Particle rest mass [kg]
    """

    # --- Unpack
    px, py, Uc = w[:, 0], w[:, 1], w[:, 2]
    U = Uc * c  # total energy in J

    # --- Compute longitudinal momentum pz from energy–momentum relation
    pz = np.sqrt((U / c)**2 - px**2 - py**2)

    # --- Half drift of positions
    xh = x + (px / pz) * (dz * 0.5)
    yh = y + (py / pz) * (dz * 0.5)
    zh = z + dz * 0.5

    # --- Fields at midpoint
    Bx, By, Bz = field_fn(xh, yh, zh)

    # --- First half-kick from Bx, By
    pxm = px - 0.5 * q * dz * By
    pym = py + 0.5 * q * dz * Bx

    # --- Recompute pz at mid-step (since px,py changed)
    pz = np.sqrt((U / c)**2 - pxm**2 - pym**2)

    # --- Rotation due to Bz
    t = 0.5 * q * Bz * dz / pz
    t2 = t * t
    s = 2 * t / (1 + t2)
    c0 = (1 - t2) / (1 + t2)

    pxp = c0 * pxm - s * pym
    pyp = s * pxm + c0 * pym
    up  = Uc  # unchanged (E=0)

    # --- Second half-kick from Bx, By
    px1 = pxp - 0.5 * q * dz * By
    py1 = pyp + 0.5 * q * dz * Bx

    # --- Recompute pz again (for next step)
    pz1 = np.sqrt((U / c)**2 - px1**2 - py1**2)

    # --- Second half drift
    x1 = xh + (px1 / pz1) * (dz * 0.5)
    y1 = yh + (py1 / pz1) * (dz * 0.5)
    z1 = z + dz

    # --- Output
    w1 = np.stack([px1, py1, up], axis=1)
    return x1, y1, z1, w1
