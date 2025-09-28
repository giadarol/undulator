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

            if ii % 10 == 0:
                print(f"Step {ii}/{self.n_steps}           ", end='\r', flush=True)

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
            Pz_J = np.sqrt((p0_J * (1 + delta))**2 - Px_J**2 - Py_J**2)

            w = np.zeros((Px_J.shape[0], 3), dtype=Px_J.dtype)
            w[:, 0] = Px_J
            w[:, 1] = Py_J
            w[:, 2] = energy_J / clight  # E/c

            x_new, y_new, z_new, w_new = step_spatial_boris(
                x, y, z, w, Pz_J, charge0_coulomb, self.ds,
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

def step_spatial_boris(x, y, z, w, pz, q, dz, field_fn):
    """
    One Δz step with E≡0 using the spatial-Boris scheme.
    - State: x,y,z (N,), w=(px,py,U/c) (N,3), pz (N,) constant (since Ez=0).
    - field_fn(x,y,z) must return (Ex,Ey,Ez,Bx,By,Bz); we ignore E components.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    w = np.asarray(w); pz = np.asarray(pz)

    # Half drift: ds/dz = w/pz  (only px,py enter)
    xh = x + (w[:, 0] / pz) * (dz * 0.5)
    yh = y + (w[:, 1] / pz) * (dz * 0.5)
    zh = z + dz * 0.5

    # Fields at midpoint (E components ignored / assumed zero)
    Bx, By, Bz = field_fn(xh, yh, zh)

    # Magnetic half-kick from transverse B: b = q(-By, +Bx, 0)
    bx = -q * By
    by =  q * Bx
    # bz = 0 since E=0 ⇒ U/c unchanged during the “kick”
    wmx = w[:, 0] + 0.5 * dz * bx
    wmy = w[:, 1] + 0.5 * dz * by
    wmz = w[:, 2]                     # unchanged by the kick

    # Exact rotation in (px,py) due to Bz over Δz (no division by Bz):
    # t = tan(theta/2) with theta = 2*atan(delta/2), delta = q Bz dz / pz
    t = 0.5 * (q * Bz * dz / pz)
    denom = 1.0 + t*t
    s = 2.0 * t / denom               # sin(theta)
    c0 = (1.0 - t*t) / denom          # cos(theta)

    # Rotate (wmx, wmy) -> (wpx, wpy); wmz (U/c) unaffected by pure rotation
    wpx = c0 * wmx + s * wmy
    wpy = c0 * wmy - s * wmx
    wpz = wmz

    # Second magnetic half-kick
    w1x = wpx + 0.5 * dz * bx
    w1y = wpy + 0.5 * dz * by
    w1z = wpz

    w1 = np.stack([w1x, w1y, w1z], axis=1)

    # Second half drift with updated slopes
    x1 = xh + (w1[:, 0] / pz) * (dz * 0.5)
    y1 = yh + (w1[:, 1] / pz) * (dz * 0.5)
    z1 = z + dz

    return x1, y1, z1, w1
