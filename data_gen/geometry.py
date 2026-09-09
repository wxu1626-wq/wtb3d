"""Blade mesh construction + area-weighted surface sampling (pure numpy)."""
import numpy as np


def naca4_airfoil(m=0.04, p=0.4, t=0.12, n=40):
    """Closed 2D airfoil polygon (2*n-2, 2) in chord-normalized coords."""
    x = np.linspace(0.0, 1.0, n)
    yt = 5.0*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    top = np.column_stack([x - yt*np.sin(th), yc + yt*np.cos(th)])
    bot = np.column_stack([x + yt*np.sin(th), yc - yt*np.cos(th)])
    return np.vstack([top, bot[1:-1][::-1]])


def loft_blade(n_sections=40, n_airfoil=48, span=5.0, c_root=1.2,
               c_tip=0.4, twist_deg=8.0, m=0.04, p=0.4, t=0.12):
    """Loft airfoil sections along span (Z axis). Returns V(M,3), F(F,3), N(M,3) in meters."""
    base = naca4_airfoil(m=m, p=p, t=t, n=n_airfoil)
    naf = base.shape[0]
    rings = []
    for i in range(n_sections):
        s = i / (n_sections - 1)
        c = c_root + (c_tip - c_root)*s
        tw = np.deg2rad(twist_deg*(1.0 - s))
        u = (base[:, 0] - 0.5)*c
        v = base[:, 1]*c
        X = u*np.cos(tw) - v*np.sin(tw)
        Y = u*np.sin(tw) + v*np.cos(tw)
        Z = np.full(naf, s*span)
        rings.append(np.column_stack([X, Y, Z]))
    V = np.vstack(rings)
    F = []
    for i in range(n_sections - 1):
        a0, b0 = i*naf, (i+1)*naf
        for j in range(naf):
            j2 = (j + 1) % naf
            a1, a2, b1, b2 = a0+j, a0+j2, b0+j, b0+j2
            F.append((a1, b1, b2)); F.append((a1, b2, a2))
    F = np.array(F, dtype=np.int64)
    N = _vertex_normals(V, F)
    return V, F, N


def _vertex_normals(V, F):
    a, b, c = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    fn = np.cross(b-a, c-a)
    N = np.zeros_like(V)
    np.add.at(N, F[:,0], fn); np.add.at(N, F[:,1], fn); np.add.at(N, F[:,2], fn)
    nn = np.linalg.norm(N, axis=1, keepdims=True)
    return np.where(nn > 0, N/np.maximum(nn, 1e-12), 0.0)


def face_areas(V, F):
    a, b, c = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    return 0.5*np.linalg.norm(np.cross(b-a, c-a), axis=1)


def sample_surface(V, F, N, n, rng):
    """Area-weighted sampling. Returns pts, nrm, (fa,fb,fc), bary (for per-vertex interp)."""
    A = face_areas(V, F)
    cdf = np.cumsum(A); cdf /= cdf[-1]
    fi = np.clip(np.searchsorted(cdf, rng.random(n)), 0, F.shape[0]-1)
    fa, fb, fc = F[fi, 0], F[fi, 1], F[fi, 2]
    P0, P1, P2 = V[fa], V[fb], V[fc]
    Q0, Q1, Q2 = N[fa], N[fb], N[fc]
    r1 = rng.random(n); r2 = rng.random(n)
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1-r1[flip], 1-r2[flip]
    pts = P0 + r1[:,None]*(P1-P0) + r2[:,None]*(P2-P0)
    nrm = Q0 + r1[:,None]*(Q1-Q0) + r2[:,None]*(Q2-Q0)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    bary = np.column_stack([1-r1-r2, r1, r2])
    return pts, nrm, fa, fb, fc, bary
