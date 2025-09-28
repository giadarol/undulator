import xobjects as xo
import xtrack as xt
import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe

class BorisSpatialIntegrator:

    isthick = True

    def __init__(self, fieldmap, s_cut, n_steps,
                 Bx0=0, By0=0):
        self.fieldmap = fieldmap
        self.s_cut = s_cut
        self.ds = s_cut / n_steps
        self.n_steps = n_steps
        self.ctx = None
        self.length = s_cut
        self.Bx0 = Bx0
        self.By0 = By0

    def get_field(self, x, y, s):
        Bx, By, Bz = self.fieldmap.get_field(x, y, s)
        Bx += self.Bx0
        By += self.By0
        Ex = 0 * Bx
        Ey = 0 * By
        Ez = 0 * Bz
        return Ex, Ey, Ez, Bx, By, Bz

    def track(self, p):

        x_log = []
        y_log = []
        z_log = []
        s_start = p.s.copy()
        p.s=0

        for ii in range(self.n_steps):

            if ii % 10 == 0:
                print(f"Step {ii}/{self.n_steps}", end='\r', flush=True)

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

            x_new, y_new, z_new, w_new = step_spatial_boris_safe(
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
        p.s += s_start
        self.x_log = np.array(x_log)
        self.y_log = np.array(y_log)
        self.z_log = np.array(z_log)

import numpy as np
c = 299_792_458.0  # m/s

# ---------- helpers ----------
def _build_R_full(Ex, Ey, Bz, pz, q, dz):
    """R from Stoltz Eq. (22)–(27); requires |Bz| not too small."""
    delta = q * Bz * dz / pz
    Ex_over_Bc = Ex / (Bz * c)
    Ey_over_Bc = Ey / (Bz * c)
    Ex2_over_B2c2 = Ex_over_Bc**2
    Ey2_over_B2c2 = Ey_over_Bc**2

    b1 = 0.5 * delta * (Ex2_over_B2c2 - 1.0)
    b2 = 0.5 * delta * (Ey2_over_B2c2 - 1.0)
    b3 = 0.5 * delta * (Ex2_over_B2c2 + Ey2_over_B2c2)

    half_delta = 0.5 * delta
    den = 1.0 + (half_delta**2) * (1.0 - (Ex2_over_B2c2 + Ey2_over_B2c2))
    alpha = (2.0 * half_delta) / den

    ExEy_over_B2c2 = Ex_over_Bc * Ey_over_Bc
    N = Ex.shape[0]
    R = np.zeros((N, 3, 3), dtype=Ex.dtype)

    R[:, 0, 0] = b1
    R[:, 0, 1] = 1.0 + 0.5 * delta * ExEy_over_B2c2
    R[:, 0, 2] = 0.5 * delta * Ey_over_Bc

    R[:, 1, 0] = -1.0 + 0.5 * delta * ExEy_over_B2c2
    R[:, 1, 1] = b2
    R[:, 1, 2] = 0.5 * delta * Ex_over_Bc

    R[:, 2, 0] = Ex_over_Bc
    R[:, 2, 1] = Ey_over_Bc
    R[:, 2, 2] = b3

    R *= alpha[:, None, None]
    return R

def _build_R_linearized(Ex, Ey, Bz, pz, q, dz):
    """Small-|Bz| fallback: second-order symmetric linearized update (no division by Bz)."""
    coef = (q * dz) / pz
    N = Ex.shape[0]
    R = np.zeros((N, 3, 3), dtype=Ex.dtype)
    # M matrix entries (see derivation in previous message)
    R[:, 0, 1] = -coef * Bz
    R[:, 1, 0] =  coef * Bz
    R[:, 0, 2] =  coef * (Ex / c)
    R[:, 1, 2] =  coef * (Ey / c)
    R[:, 2, 0] = -coef * (Ex / c)
    R[:, 2, 1] = -coef * (Ey / c)
    return R

def _make_b(Bx, By, q):
    # b = q(-By, +Bx, 0) since Ez = 0
    return np.stack([-q*By, q*Bx, np.zeros_like(Bx)], axis=1)

# ---------- w-push with safe Bz handling (Ez = 0) ----------
def spatial_boris_w_push_safe(
    w, Ex, Ey, Bx, By, Bz, pz, q, dz,
    bz_abs_switch=1e-9,     # Tesla: switch to linearized if |Bz| < this
    bz_rel_switch=1e-6      # also switch if |E|/(|Bz| c) > this
):
    """
    Vectorized update of w=(p_x,p_y,U/c) by Δz, Ez=0.
    - 'Safe' particles: full spatial-Boris.
    - 'Small-Bz' particles: linearized symmetric (2nd order) fallback.
    """
    N = w.shape[0]
    b_vec = _make_b(Bx, By, q)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_x = np.abs(Ex) / (np.abs(Bz) * c)
        ratio_y = np.abs(Ey) / (np.abs(Bz) * c)

    small_mask = (np.abs(Bz) < bz_abs_switch) | (ratio_x > bz_rel_switch) | (ratio_y > bz_rel_switch)
    safe_mask = ~small_mask

    w_next = np.empty_like(w)

    # Safe: full spatial-Boris (Eqs. 28–30 with R from Eq. 22)
    if np.any(safe_mask):
        R_full = _build_R_full(Ex[safe_mask], Ey[safe_mask], Bz[safe_mask], pz[safe_mask], q, dz)
        b_s = b_vec[safe_mask]; w_s = w[safe_mask]
        w_minus = w_s + 0.5 * dz * b_s
        w_plus  = w_minus + np.einsum('nij,nj->ni', R_full, w_minus)
        w_next[safe_mask] = w_plus + 0.5 * dz * b_s

    # Small-Bz: linearized symmetric update (no division by Bz)
    if np.any(small_mask):
        R_lin = _build_R_linearized(Ex[small_mask], Ey[small_mask], Bz[small_mask], pz[small_mask], q, dz)
        b_l = b_vec[small_mask]; w_l = w[small_mask]
        w_minus = w_l + 0.5 * dz * b_l
        w_plus  = w_minus + np.einsum('nij,nj->ni', R_lin, w_minus)
        w_next[small_mask] = w_plus + 0.5 * dz * b_l

    return w_next, {"mask_safe": safe_mask, "mask_linear": small_mask}

# ---------- full Δz step (Ez = 0) ----------
def step_spatial_boris_safe(x, y, z, w, pz, q, dz, field_fn, **kwargs):
    """
    One Δz step with small-|Bz| protection. field_fn(x,y,z) -> Ex,Ey,Ez,Bx,By,Bz.
    Ez is ignored (assumed 0) here.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)

    # half drift: ds/dz = w/pz
    xh = x + (w[:, 0] / pz) * (dz * 0.5)
    yh = y + (w[:, 1] / pz) * (dz * 0.5)
    zh = z + dz * 0.5

    Ex, Ey, Ez, Bx, By, Bz = field_fn(xh, yh, zh)

    w1, aux = spatial_boris_w_push_safe(w, Ex, Ey, Bx, By, Bz, pz, q, dz, **kwargs)

    x1 = xh + (w1[:, 0] / pz) * (dz * 0.5)
    y1 = yh + (w1[:, 1] / pz) * (dz * 0.5)
    z1 = z + dz
    return x1, y1, z1, w1
