"""
Local Signal Approximation and Trajectories [gu105.0]
=====================================================

Fits a high-degree polynomial locally to a rectangular test signal using
[`RLSAlssm`][lmlib.statespace.rls.RLSAlssm] and compares the Pascal monomial
basis ([`AlssmPoly`][lmlib.statespace.model.AlssmPoly]) against the numerically
superior Legendre basis ([`AlssmPolyLegendre`][lmlib.statespace.model.AlssmPolyLegendre]).

For each of three reference positions, the optimal state vector is extracted
from the full filter output and the corresponding trajectory is placed in the
output signal at the correct absolute location.  A wider window version shows
the trajectory extrapolated beyond the fitting window.
"""
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils.generator import gen_wgn, gen_rect

plt.close('all')

K = 2100
y = gen_rect(K, 570, 200) + gen_wgn(K, 0.01)

pd=7
g=100
cost = lm.CostSegment(lm.AlssmPoly(poly_degree=pd),
                      lm.Segment(0, 200, lm.BW, g))
cost_wide = lm.CostSegment(lm.AlssmPoly(poly_degree=pd),
                           lm.Segment(0 - 20, 220 + 20, lm.BW, g))

rls = lm.RLSAlssm(cost) 
rls.filter(y)
xs = rls.minimize_x(solver='lstsq')

cost_legendre = lm.CostSegment(lm.AlssmPolyLegendre(poly_degree=pd,a_seg=0,b_seg=200),
                      lm.Segment(0, 200, lm.BW, g))
cost_legendre_wide = lm.CostSegment(lm.AlssmPolyLegendre(poly_degree=pd,a_seg=0,b_seg=200),
                           lm.Segment(0 - 20, 220 + 20, lm.BW, g))

rls_legendre = lm.RLSAlssm(cost_legendre) 
rls_legendre.filter(y)
xs_legendre = rls_legendre.minimize_x(solver='lstsq')

K_refs = [500, 1130, 1800]
trajs = lm.Trajectory.eval_y(cost, xs[K_refs], K_refs, K, thd=0.01, merged_ks=True, merged_seg=True)
trajs_wide =lm.Trajectory.eval_y(cost_wide,xs[K_refs], K_refs, K, thd=0.01,merged_ks=True, merged_seg=True)

trajs_legendre = lm.Trajectory.eval_y(cost_legendre, xs_legendre[K_refs], K_refs, K, thd=0.01, merged_ks=True, merged_seg=True)
trajs_legendre_wide = lm.Trajectory.eval_y(cost_legendre_wide,xs_legendre[K_refs], K_refs, K, thd=0.01,merged_ks=True, merged_seg=True)

plt.title(f'Local Signal Approximation: Polynomial degree {pd}, RLSAlssm.minimize_x()')
plt.plot(y, lw=0.3, c='k', label='y')
plt.plot(trajs, lw=2, c='b', label='y_hat, Pascal Poly')
plt.plot(trajs_wide, lw=1, ls='--', c='b')
plt.plot(trajs_legendre, lw=2, c='g', label='y_hat, Legendre Poly')
plt.plot(trajs_legendre_wide, ls='--', lw=2, c='g')
plt.ylim([-3, 3])
plt.legend(loc=1)
plt.show()
