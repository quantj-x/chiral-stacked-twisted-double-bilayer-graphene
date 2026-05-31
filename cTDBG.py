"""
cTDBG.py  —  Continuum model for Twisted Double Bilayer Graphene (TDBG), AB-BA stacking.
Author: Jiao Xie, 2026.
Email: xiejiao@smail.nju.edu.cn
Version: 1.0 (2026-05-20)
"""

import numpy as np
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt

# ===================================================================================================
#   Reference:
#     M. Koshino, "Band structure and topological properties of twisted double bilayer graphene",
#     Phys. Rev. B 99, 235406 (2019).            <-- every block below is keyed to its equations.
#
#   Basis (8 components per moiré reciprocal-lattice site pair), Koshino's ordering:
#       (A1, B1, A2, B2, A3, B3, A4, B4)
#   layers 1,2 = first (bottom-block here)  bilayer ;  layers 3,4 = second (top-block) bilayer.
#   The moiré tunneling U couples layer 2  <->  layer 3   (Eq. 4).
# ===================================================================================================

# ------------------------------------
# Helper functions & small utilities
# ------------------------------------

def R(x):
    """Generic 2D rotation matrix."""
    return np.array([[np.cos(x), -np.sin(x)],
                     [np.sin(x),  np.cos(x)]])

I = np.identity(2)


def rot2d(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]], dtype=float)


def Et(theta, theta_s, e, delta):
    """
    Simultaneous small rotation and uniaxial strain operator acting on a 2D vector.
    See: Bi et al., Phys. Rev. B 100, 035448 (2019).

    theta   : rotation angle (rad)
    theta_s : strain axis angle (rad)
    e       : strain magnitude
    delta   : Poisson ratio
    """
    return R(-theta_s) @ np.array([[e, 0], [0, -delta * e]]) @ R(theta_s) \
           + np.array([[0, -theta], [theta, 0]])


# ------------------------------------
# Continuum model class
# ------------------------------------
class TDBGModel():

    def _parse_valley(self, valley):
        """Map a user-friendly valley spec into a list of xi values."""
        if valley is None:
            return [-1, 1]
        s = str(valley).strip().lower()
        if s in ('both', 'all', '±', '+-'):
            return [-1, 1]
        if s in ('k', 'solid', '-', '-1', 'm', 'minus'):
            return [-1]
        if s in ("k'", 'kprime', 'dashed', '+', '+1', 'p', 'plus'):
            return [1]
        raise ValueError("valley must be one of: 'both', 'K', \"K'\", 'dashed', or 'solid'.")

    """
    TDBG continuum model (Koshino PRB 99, 235406).

    Parameters
    ----------
    theta : float       twist angle (degrees) between the two bilayers.
    phi : float         heterostrain axis angle (degrees).            [extension; phi irrelevant if epsilon=0]
    epsilon : float     heterostrain magnitude (0 -> exact Koshino).  [extension]
    D : float           interlayer asymmetric potential.
                        * use_D_as_field=False (default route in RUN script): D IS Koshino's Delta
                          (the electrostatic-energy step between ADJACENT layers), in eV.
                          The 4 layers then sit at (+3/2, +1/2, -1/2, -3/2)*Delta  (Eq. 5).
                          e.g. D = 0.020  reproduces Koshino's Delta = 20 meV panels.
                        * use_D_as_field=True: D is the external field (V/nm); Delta = e*D*dt_nm/eps_s,
                          with dt_nm the SINGLE interlayer spacing (~0.335 nm).

    a : float           graphene lattice constant (nm).
    beta : float        two-center hopping modulus for the strain gauge field.
    delta : float       Poisson ratio.
    vf : float          velocity renormalization. vf=1 gives Koshino's hv/a = 2.1354 eV exactly.
    u, up : float       moiré AA / AB tunneling amplitudes (eV). Koshino: 0.0797 / 0.0975.
    cut : float         momentum cutoff in units of sqrt(3)*k_theta (>=4 converges).
    dt_nm : float       single interlayer spacing for the field->Delta conversion (nm).
    eps_s : float       effective dielectric constant for the field->Delta conversion.
    use_D_as_field : bool  see D above.
    Dp : float          dimer-site onsite potential Delta' (eV). Koshino full model: 0.050.
                        Set Dp=0 for Koshino's "minimal model".
    stacking : str      'AB-BA' (default) or 'AB-AB'.
    """

    def __init__(self, theta, phi, epsilon, D,
                 a=0.246, beta=3.14, delta=0.16,
                 vf=1, u=0.0797, up=0.0975, cut=4,
                 dt_nm=0.335, eps_s=4.0, use_D_as_field=True, e_charge_eV_per_V=1.0,
                 Dp=0.050, stacking='AB-BA'):

        # --- convert to radians ---
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180

        # --- store empirical params ---
        self.theta = theta
        self.phi = phi
        self.epsilon = epsilon
        self.D = D

        # field -> Delta conversion parameters
        self.dt_nm = dt_nm
        self.eps_s = eps_s
        self.use_D_as_field = use_D_as_field
        self.e_charge_eV_per_V = e_charge_eV_per_V  # 1 e x 1 V = 1 eV

        # --- graphene params ---
        self.a = a
        self.beta = beta
        self.delta = delta
        self.A = np.sqrt(3) * self.beta / (2 * a)  # strain gauge-field prefactor

        # --- continuum model params (Koshino values) ---
        self.v = vf * 2.1354 * a            # hv  (vf=1 -> Koshino hv/a = 2.1354 eV)
        self.v3 = np.sqrt(3) * a * 0.32 / 2  # from gamma3 = 0.32 eV
        self.v4 = np.sqrt(3) * a * 0.044 / 2  # from gamma4 = 0.044 eV
        self.gamma1 = 0.4                    # dimer coupling (eV)
        self.Dp = Dp                         # dimer-site potential Delta' (eV)
        self.stacking = str(stacking).strip().upper().replace('_', '-')

        self.omega = np.exp(1j * 2 * np.pi / 3)
        self.u = u
        self.up = up

        # --- graphene K-point magnitudes and reciprocal-lattice vectors ---
        k_d = 4 * np.pi / (3 * a)
        k1 = np.array([k_d, 0])
        k2 = np.array([np.cos(2 * np.pi / 3) * k_d, np.sin(2 * np.pi / 3) * k_d])
        k3 = -np.array([np.cos(np.pi / 3) * k_d, np.sin(np.pi / 3) * k_d])

        # --- strained moiré reciprocal vectors q_i ---
        q1 = Et(theta, phi, epsilon, delta) @ k1
        q2 = Et(theta, phi, epsilon, delta) @ k2
        q3 = Et(theta, phi, epsilon, delta) @ k3
        q = np.array([q1, q2, q3])
        self.q = q
        self.k_theta = np.max([norm(q1), norm(q2), norm(q3)])

        # --- basis vectors for the Q lattice (monolayer K lattices) ---
        b1 = q[1] - q[2]
        b2 = q[0] - q[2]
        b3 = q[1] - q[0]
        b = np.array([b1, b2, b3])
        self.b = b

        # --- generate Q lattice: two copies (l = 0,1) within cutoff ---
        Q = np.array([np.array(list([i, j, 0] @ b - l * q[0]) + [l])
                      for i in range(-100, 100)
                      for j in range(-100, 100)
                      for l in [0, 1]
                      if norm([i, j, 0] @ b - l * q[0]) <= np.sqrt(3) * self.k_theta * cut])
        self.Q = Q
        self.Nq = len(Q)

        # --- nearest neighbors on Q lattice (for moiré couplings) ---
        self.Q_nn = {}
        Q_round = np.round(Q[:, :2], 6)
        Q_list = Q_round.tolist()
        Q_index = {tuple(Q_list[idx]): idx for idx in range(len(Q_list))}

        for i in range(self.Nq):
            nbrs = []
            for j in range(len(q)):
                target = tuple(np.round(self.Q[i, :2] + q[j], 6))
                idx = Q_index.get(target)
                if idx is not None:
                    nbrs.append([idx, j])
            self.Q_nn[i] = nbrs

        # --- physical Fourier momenta associated with Q lattice points (for LDOS) ---
        Q2G = np.array([[l, l] for l in Q[:, 2]]) * q[0] + Q[:, :2]
        self.G = Q2G[Q[:, 2] == 1]

    # ------------------------------
    # Core: build Hamiltonian
    # ------------------------------
    def gen_ham(self, kx, ky, xi=1):
        """
        Build H(k) at momentum (kx, ky) for valley xi = +-1.  Returns a Hermitian matrix.

        Block layout follows Koshino Eqs. (1)-(6):
          * bottom bilayer (t==0)  ->  AB block  [H0, g_dag; g, H0']  on layers (1,2)
          * top bilayer    (t==1)  ->  AB  (stacking='AB-AB')   or
                                       BA  (stacking='AB-BA', default), obtained from the AB
                                       block by the layer-swap similarity P_L (= Eq. 6).
          * moiré coupling U (Eq. 4) connects layer 2 <-> layer 3 and is the SAME for both stackings.
          * V (Eq. 5): layer-asymmetric potential, tied to the PHYSICAL layer (NOT swapped by P_L).
          * Delta' (dimer-site potential, Eq. 2): tied to the dimer sublattice (IS relocated by P_L).
        """
        k = np.array([kx, ky])

        # --- moiré tunneling matrices (Koshino Eq. 4); each is Hermitian ---
        U1 = np.array([[self.u, self.up],
                       [self.up, self.u]])
        U2 = np.array([[self.u, self.up * self.omega ** (-xi)],
                       [self.up * self.omega ** (xi), self.u]])
        U3 = np.array([[self.u, self.up * self.omega ** (xi)],
                       [self.up * self.omega ** (-xi), self.u]])

        ham = np.zeros((4 * self.Nq, 4 * self.Nq), dtype=complex)

        # layer-swap permutation (AB -> BA for the top bilayer): swaps (0,1)<->(2,3)
        P_L = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]], dtype=float)

        # --- interlayer asymmetric potential V (Koshino Eq. 5), per-layer step = Delta ---
        if self.use_D_as_field:
            Delta = (self.D * self.dt_nm * self.e_charge_eV_per_V) / self.eps_s
        else:
            Delta = self.D                                 # D is already Koshino's Delta (eV)
        V_layer =xi * np.array([1.5, 0.5, -0.5, -1.5]) * Delta  # (L1, L2, L3, L4)

        is_ba = self.stacking in ('AB-BA', 'ABBA', 'BA')

        for i in range(self.Nq):
            t = self.Q[i, 2]              # 0 = first bilayer (layers 1,2), 1 = second bilayer (3,4)
            l = np.sign(2 * t - 1)        # -1 (t=0) / +1 (t=1): counter-rotation sign

            # strain+rotation operator; symmetric part -> strain tensor for the gauge field
            M = Et(l * xi * self.theta / 2, self.phi, l * xi * self.epsilon / 2, self.delta)
            E = (M + M.T) / 2
            exx, eyy, exy = E[0, 0], E[1, 1], E[0, 1]

            # gauge-shifted, rotated momentum
            kj = (I + M) @ (k + self.Q[i, :2] + xi * self.A * np.array([exx - eyy, -2 * exy]))
            km = xi * kj[0] - 1j * kj[1]   # k_-  = xi*kx - i*ky
            kp = xi * kj[0] + 1j * kj[1]   # k_+  = xi*kx + i*ky

            # ---- AB-reference KINETIC bilayer block (Koshino H0, H0', g) ----
            #   basis within block: (A_up, B_up, A_low, B_low); dimer = (B_up, A_low) = (1,2)
            Hk = np.zeros((4, 4), dtype=complex)
            Hk[0, 1] = -self.v * km        # Dirac, upper layer       (H0 off-diagonal)
            Hk[2, 3] = -self.v * km        # Dirac, lower layer       (H0' off-diagonal)
            Hk[1, 2] = self.gamma1         # gamma1 dimer  (B_up - A_low)
            Hk[0, 2] = self.v4 * km        # v4   (A_up - A_low)
            Hk[1, 3] = self.v4 * km        # v4   (B_up - B_low)
            Hk[0, 3] = self.v3 * kp        # v3   (A_up - B_low, non-dimer)
            Hk = Hk + Hk.conj().T          # Hermitize off-diagonals (diagonal still 0)
            # dimer-site potential Delta' on dimer sublattices (B_up=1, A_low=2)
            Hk[1, 1] += self.Dp
            Hk[2, 2] += self.Dp

            # ---- choose stacking for the top bilayer ----
            if t == 1 and is_ba:
                Hloc = P_L.T @ Hk @ P_L    # BA block: Dirac preserved, dimer/Delta' relocated (Eq. 6)
            else:
                Hloc = Hk                  # AB block (Eq. 1)

            # ---- add layer-asymmetric potential V on the diagonal (physical layer, NOT swapped) ----
            if t == 0:
                Hloc = Hloc + np.diag([V_layer[0], V_layer[0], V_layer[1], V_layer[1]])  # L1,L1,L2,L2
            else:
                Hloc = Hloc + np.diag([V_layer[2], V_layer[2], V_layer[3], V_layer[3]])  # L3,L3,L4,L4

            # place upper-triangle of this (Hermitian) block; final Hermitization restores it
            iu = np.triu_indices(4)
            blk = np.zeros((4, 4), dtype=complex)
            blk[iu] = Hloc[iu]
            ham[4 * i:4 * i + 4, 4 * i:4 * i + 4] = blk

            # ---- moiré coupling: layer 2 (of t=0 block) <-> layer 3 (of t=1 block); U UNCHANGED ----
            for (j, p) in self.Q_nn[i]:
                U = (p == 0) * U1 + (p == 1) * U2 + (p == 2) * U3
                ham[4 * j + 2:4 * j + 4, 4 * i:4 * i + 2] = U   # rows = L2@j, cols = L3@i

        return ham + ham.conj().T - np.diag(np.diag(ham))

    # ---------------------------------------------------------------------
    #  Band-structure paths along high-symmetry lines of the moiré BZ.
    # ---------------------------------------------------------------------
    def solve_along_path_one(self, res=300, plot_it=True, return_eigenvectors=False, ylim=None, valley='both'):
        """Band structure along K -> Gamma -> M -> K'."""
        l1 = int(res)                   # K -> Gamma
        l2 = int(np.sqrt(3) * res / 2)  # Gamma -> M
        l3 = int(res / 2)               # M -> K'

        kpath = []
        for i in np.linspace(0, 1, l1):
            kpath.append(i * (self.q[0] + self.q[1]))
        for i in np.linspace(0, 1, l2):
            kpath.append(self.q[0] + self.q[1] + i * (-self.q[0] / 2 - self.q[1]))
        for i in np.linspace(0, 1, l3):
            kpath.append(self.q[0] / 2 + i * self.q[0] / 2)

        kpath = np.array(kpath)
        valleys = self._parse_valley(valley)

        evals_m, evals_p = [], []
        if return_eigenvectors:
            evecs_m, evecs_p = [], []

        for kpt in kpath:
            if -1 in valleys:
                val, vec = eigh(self.gen_ham(kpt[0], kpt[1], -1))
                evals_m.append(val)
                if return_eigenvectors:
                    evecs_m.append(vec)
            if 1 in valleys:
                val, vec = eigh(self.gen_ham(kpt[0], kpt[1], 1))
                evals_p.append(val)
                if return_eigenvectors:
                    evecs_p.append(vec)

        def _pack(arr):
            return np.empty((len(kpath), 0)) if len(arr) == 0 else np.array(arr)

        evals_m = _pack(evals_m)
        evals_p = _pack(evals_p)

        if plot_it:
            plt.figure(1, figsize=(5, 4))
            plt.clf()
            if evals_m.shape[1] > 0:
                for i in range(evals_m.shape[1]):
                    plt.plot(evals_m[:, i], linestyle='solid')
            if evals_p.shape[1] > 0:
                for i in range(evals_p.shape[1]):
                    plt.plot(evals_p[:, i], linestyle='dashed')
            plt.ylim(*ylim) if ylim is not None else plt.ylim(-0.035, 0.035)
            plt.xticks([0, l1, l1 + l2, l1 + l2 + l3], ['K', r'$\Gamma$', 'M', "K'"])
            plt.ylabel('Energy (eV)')
            plt.tight_layout()
            plt.show()

        if return_eigenvectors:
            evecs_m = np.empty((len(kpath), 0, 0)) if (-1 not in valleys) else np.array(evecs_m)
            evecs_p = np.empty((len(kpath), 0, 0)) if (1 not in valleys) else np.array(evecs_p)
            return evals_m, evals_p, evecs_m, evecs_p, kpath
        return evals_m, evals_p, kpath

    def solve_along_path_two(self, res=300, plot_it=True, return_eigenvectors=False, ylim=None, valley='both'):
        """Band structure along K' -> M -> Gamma -> K."""
        l1 = int(res / 2)
        l2 = int(np.sqrt(3) * res / 2)
        l3 = int(res)

        kpath = []
        for i in np.linspace(0, 1, l1):
            kpath.append(self.q[0] + i * (-self.q[0] / 2))
        for i in np.linspace(0, 1, l2):
            kpath.append(self.q[0] / 2 + i * (self.q[0] / 2 + self.q[1]))
        for i in np.linspace(0, 1, l3):
            kpath.append(self.q[0] + self.q[1] + i * (-(self.q[0] + self.q[1])))

        kpath = np.array(kpath)
        valleys = self._parse_valley(valley)

        evals_m, evals_p = [], []
        if return_eigenvectors:
            evecs_m, evecs_p = [], []

        for kpt in kpath:
            if -1 in valleys:
                val, vec = eigh(self.gen_ham(kpt[0], kpt[1], -1))
                evals_m.append(val)
                if return_eigenvectors:
                    evecs_m.append(vec)
            if 1 in valleys:
                val, vec = eigh(self.gen_ham(kpt[0], kpt[1], 1))
                evals_p.append(val)
                if return_eigenvectors:
                    evecs_p.append(vec)

        def _pack(arr):
            return np.empty((len(kpath), 0)) if len(arr) == 0 else np.array(arr)

        evals_m = _pack(evals_m)
        evals_p = _pack(evals_p)

        if plot_it:
            plt.figure(1, figsize=(5, 4))
            plt.clf()
            if evals_m.shape[1] > 0:
                for i in range(evals_m.shape[1]):
                    plt.plot(evals_m[:, i], linestyle='solid')
            if evals_p.shape[1] > 0:
                for i in range(evals_p.shape[1]):
                    plt.plot(evals_p[:, i], linestyle='dashed')
            plt.ylim(*ylim) if ylim is not None else plt.ylim(-0.035, 0.035)
            plt.xticks([0, l1, l1 + l2, l1 + l2 + l3], [r'$K_{1}$', 'M', r'$\Gamma$', r'$K_{2}$'])
            plt.ylabel(r'$E(eV)$')
            plt.tight_layout()
            plt.show()

        if return_eigenvectors:
            evecs_m = np.empty((len(kpath), 0, 0)) if (-1 not in valleys) else np.array(evecs_m)
            evecs_p = np.empty((len(kpath), 0, 0)) if (1 not in valleys) else np.array(evecs_p)
            return evals_m, evals_p, evecs_m, evecs_p, kpath
        return evals_m, evals_p, kpath

    def solve_along_path_three(self, res=1200, plot_it=True, return_eigenvectors=False, ylim=None, valley='both'):
        """Band structure along K1 -> Gamma -> M -> K1."""
        l1 = int(res)
        l2 = int(np.sqrt(3) * res / 2)
        l3 = int(res / 2)

        kpath = []
        for i in np.linspace(0, 1, l1):
            kpath.append(self.q[0] + i * (self.q[1]))
        for i in np.linspace(0, 1, l2):
            kpath.append(self.q[0] + self.q[1] + i * (-self.q[0] / 2 - self.q[1]))
        for i in np.linspace(0, 1, l3):
            kpath.append(self.q[0] / 2 + i * (self.q[0] / 2))

        kpath = np.array(kpath)
        valleys = self._parse_valley(valley)

        evals_m, evals_p = [], []
        if return_eigenvectors:
            evecs_m, evecs_p = [], []

        for kpt in kpath:
            if -1 in valleys:
                val, vec = eigh(self.gen_ham(kpt[0], kpt[1], -1))
                evals_m.append(val)
                if return_eigenvectors:
                    evecs_m.append(vec)
            if 1 in valleys:
                val, vec = eigh(self.gen_ham(kpt[0], kpt[1], 1))
                evals_p.append(val)
                if return_eigenvectors:
                    evecs_p.append(vec)

        def _pack(arr):
            return np.empty((len(kpath), 0)) if len(arr) == 0 else np.array(arr)

        evals_m = _pack(evals_m)
        evals_p = _pack(evals_p)

        if plot_it:
            plt.figure(1, figsize=(5, 4))
            plt.clf()
            if evals_m.shape[1] > 0:
                for i in range(evals_m.shape[1]):
                    plt.plot(evals_m[:, i], linestyle='solid')
            if evals_p.shape[1] > 0:
                for i in range(evals_p.shape[1]):
                    plt.plot(evals_p[:, i], linestyle='dashed')
            plt.ylim(*ylim) if ylim is not None else plt.ylim(-0.04, 0.04)
            plt.xticks([0, l1, l1 + l2, l1 + l2 + l3],
                       [r'$K_{s}$', r'$\Gamma_{s}$', r'$M_{s}$', r'$K_{s}$'])
            plt.ylabel(r'$E(eV)$')
            plt.tight_layout()
            plt.show()

        if return_eigenvectors:
            evecs_m = np.empty((len(kpath), 0, 0)) if (-1 not in valleys) else np.array(evecs_m)
            evecs_p = np.empty((len(kpath), 0, 0)) if (1 not in valleys) else np.array(evecs_p)
            return evals_m, evals_p, evecs_m, evecs_p, kpath
        return evals_m, evals_p, kpath

    # ---------------------------------------------------------------------
    #  Layer-resolved DOS.  Layer <-> component map is now FIXED (physical):
    #     L1 = (A1,B1) = indices 0,1 of t=0 blocks ;  L2 = indices 2,3 of t=0
    #     L3 = (A3,B3) = indices 0,1 of t=1 blocks ;  L4 = indices 2,3 of t=1
    #  (independent of valley and of stacking, because V is added per physical layer).
    # ---------------------------------------------------------------------
    def _layer_masks(self):
        nL = 4 * self.Nq
        mL = [np.zeros(nL, dtype=bool) for _ in range(4)]
        for i in range(self.Nq):
            base = 4 * i
            if self.Q[i, 2] == 0:           # bottom bilayer -> L1, L2
                mL[0][base], mL[0][base + 1] = True, True
                mL[1][base + 2], mL[1][base + 3] = True, True
            else:                           # top bilayer -> L3, L4
                mL[2][base], mL[2][base + 1] = True, True
                mL[3][base + 2], mL[3][base + 3] = True, True
        return mL

    def solve_PDOS(self, nk=16, energies=np.round(np.linspace(-.1, .1, 201), 3), xi=1, plot_it=True):
        """Layer-resolved DOS at valley xi (histogrammed over a uniform moiré-BZ k-grid)."""
        energies = np.asarray(energies)
        de = energies[1] - energies[0] if len(energies) > 1 else 1.0
        edges = np.concatenate([energies - de / 2, [energies[-1] + de / 2]])
        mL = self._layer_masks()

        kpts = np.array([i * self.b[0] - j * self.b[1]
                         for i in np.linspace(0, 1, nk, endpoint=False)
                         for j in np.linspace(0, 1, nk, endpoint=False)])

        weights = [[] for _ in range(4)]   # per layer: (energy, weight) accumulation
        evals_all = [[] for _ in range(4)]
        for kpt in kpts:
            vals, vecs = eigh(self.gen_ham(kpt[0], kpt[1], xi))
            prob = np.abs(vecs) ** 2       # |psi|^2, columns = eigenstates
            for L in range(4):
                wL = prob[mL[L], :].sum(axis=0)   # layer weight of each eigenstate
                evals_all[L].append(vals)
                weights[L].append(wL)

        PDOS = []
        for L in range(4):
            ev = np.concatenate(evals_all[L])
            wt = np.concatenate(weights[L])
            hist, _ = np.histogram(ev, bins=edges, weights=wt)
            PDOS.append(hist / (len(kpts) * de))
        PDOS.append(sum(PDOS) / 4.0)

        if plot_it:
            plt.figure(1)
            plt.clf()
            for y in PDOS:
                plt.plot(energies, y)
            plt.xlabel('Energy (eV)')
            plt.ylabel('DOS')
            plt.legend(['L1', 'L2', 'L3', 'L4', 'Full'])
            plt.tight_layout()
            plt.show()
        return PDOS