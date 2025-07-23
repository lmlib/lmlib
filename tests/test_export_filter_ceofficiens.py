import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.linalg import det
import lmlib as lm
from lmlib.utils import gen_slopes

from scipy.signal import ss2tf, lfilter

def shift_signal(y, a):
    y_shifted = np.zeros_like(y)
    # shift fir outputs
    s_signal = (
        a + 1
    )  # signal is shifted +1 compared to state space system (check recursion formulas)
    if s_signal == 0:
        return y
    if s_signal > 0:
        y_shifted[: K - s_signal] = y[s_signal:]
    if s_signal < 0:
        y_shifted[-s_signal:] = y[: K + s_signal]
    return y_shifted


def FIR_filter(b, y, b_axis=0):
    N = np.shape(b)[b_axis]
    out = np.zeros(np.insert(np.shape(y), 1, N))
    a = np.zeros_like(b[0])
    a[0] = 1
    for n in range(N):
        out[:, n] = lfilter(b[n], a, y)
    return out


def IIR_filter(a, y, y_axis=1):
    N = np.shape(y)[y_axis]
    out = np.zeros_like(y)
    b = np.zeros_like(a)
    b[0] = 1
    for n in range(N):
        out[:, n] = lfilter(b, a, y[:, n])
    return out


def ss_to_tf_coef(
    alssm: lm.ModelBase, segment: lm.Segment
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = segment.a
    b = segment.b
    gamma = segment.gamma
    A = alssm.A
    C = alssm.C
    N = alssm.N

    Css = np.eye(N)
    Dss = np.zeros((N, 1))

    # TF of system using ss2tf
    # ------- a boundary --------

    # Bss = np.linalg.matrix_power(gamma * A, a).T @ C.reshape(1, N).T  # use this if the input signal is filtered directly
    Bssgamma1 = (
        np.linalg.matrix_power(A, a-1).T @ C.reshape(1, N).T
    )  # use this if the input signal is multiplied with gamma^s
    Ass = np.linalg.inv(A * gamma).T
    q_a, p = ss2tf(Ass, Bssgamma1, Css, Dss)

    # ------- b boundary --------
    Css = np.eye(N)
    Dss = np.zeros((N, 1))
    # Bss = np.linalg.matrix_power(gamma * A, b).T @ C.reshape(1, N).T  # use this if the input signal is filtered directly
    Bssgamma1 = (
        np.linalg.matrix_power(A, b).T @ C.reshape(1, N).T
    )  # use this if the input signal is multiplied with gamma^s
    Ass = np.linalg.inv(A * gamma).T
    q_b, _ = ss2tf(Ass, Bssgamma1, Css, Dss)

    return q_a, q_b, p

def sympy_det_coef(alssm, segment):
    N = alssm.N

    gamma = sympy.symbols('gamma')
    z = sympy.symbols('z')
    I = sympy.eye(N)
    A = sympy.Matrix(alssm.A)
    Atilde = 1/gamma * A.inv().T  # note that this would normally be gamma^{-1} in backward recursions
    c = sympy.Matrix(alssm.C).T
    # p_sympy = sympy.det(z * I - Atilde)
    p_sympy = Atilde.charpoly() # advanced
    p = sympy.Poly(p_sympy.subs([('gamma', segment.gamma)])).all_coeffs()
    p_len = len(p)

    # TF Coefficients for Boundary b Numerator
    b = sympy.symbols('b')
    q_b = []
    for n in range(N):
        ei = sympy.zeros(1, N)
        ei[n] = 1
        barei = sympy.eye(N)
        barei[n, n] = 0
        cAbTei = (c * A ** b).T * ei
        zIAbarei = (z * I - Atilde) * barei
        q_i = sympy.det(zIAbarei + cAbTei)
        q_ = sympy.Poly(q_i.subs([('gamma', segment.gamma), ('b', segment.b)])).all_coeffs()
        q_ = np.concatenate([np.zeros(p_len-len(q_)), q_]) if p_len > len(q_) else q_
        q_b.append(q_)

    a = sympy.symbols('a')
    q_a = []
    for n in range(N):
        ei = sympy.zeros(1, alssm.N)
        ei[n] = 1
        barei = sympy.eye(alssm.N)
        barei[n, n] = 0
        cAaTei = (c * A ** (a - 1)).T * ei
        zIAbarei = (z * I - Atilde) * barei
        tildeqi = sympy.det(zIAbarei + cAaTei)
        q_ = sympy.Poly(tildeqi.subs([('gamma', segment.gamma), ('a', segment.a)])).all_coeffs()
        q_ = np.concatenate([np.zeros(p_len-len(q_)), q_]) if p_len > len(q_) else q_
        q_a.append(q_)

    if len(q_a[0]) > p_len:
        p = np.concatenate([np.zeros(len(q_a[0])-p_len), p])

    return np.asarray(q_a), np.asarray(q_b), p


def scipy_det_coef(alssm, segment):
    A = alssm.A
    C = alssm.C
    N = alssm.N
    a = segment.a
    b = segment.b
    gamma = segment.gamma

    I = np.eye(N)
    M = 1/gamma * np.linalg.inv(A).T
    p = np.flip(np.polynomial.Polynomial.fromroots(np.linalg.eigvals(M)).coef)
    p_len = len(p)

    # TF Coefficients for Boundary b Numerator
    q_b = []
    for n in range(N):
        ei = np.zeros(N)
        ei[n] = 1
        cAbTei =np.outer(C@np.linalg.matrix_power(A,b), ei)
        p1 = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(M + cAbTei))
        p2 = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(M))
        q_ = -np.flip((p1-p2).coef)
        q_ = np.concatenate([np.zeros(p_len-len(q_)), q_]) if p_len > len(q_) else q_
        q_b.append(q_)

    q_a = []
    for n in range(N):
        ei = np.zeros(N)
        ei[n] = 1
        cAaTei = np.outer(C@np.linalg.matrix_power(A, a - 1),  ei)
        p1 = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(M + cAaTei))
        p2 = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(M))
        q_ = -np.flip((p1 - p2).coef)
        q_ = np.concatenate([np.zeros(p_len-len(q_)), q_]) if p_len > len(q_) else q_
        q_a.append(q_)

    return np.asarray(q_a), np.asarray(q_b), p

# Signal
K = 450
k = np.arange(K)
ks = [140, 180, 230, 260, 320]
deltas = [0, 5, -8.5, 3, -3]
y = gen_slopes(K, ks, deltas)  # + gen_wgn(K, sigma=0.2, seed=3141)


alssm = lm.AlssmPoly(poly_degree=2)
segment = lm.Segment(a=-10, b=-1, direction=lm.FW, g=1000, delta=0)
cost = lm.CostSegment(alssm, segment)
rls = lm.RLSAlssm(cost, steady_state=False)
xs_numpy = rls.filter_minimize_x(y)
Winv = np.linalg.inv(cost.get_steady_state_W())

# ss2tf option
colors = ('b', 'purple', 'orange')
lss = (':', '-.', '--')
for i, flag in enumerate(('ss2tf', 'sympy', 'scipy_det')):
    print('\n-------'+flag+'--------')
    if flag == 'ss2tf':
        q_a, q_b, p = ss_to_tf_coef(alssm, segment)
        print(f'Q(z, a) =  \n {q_a} \n  Q(z, b) = \n  {q_b} \n P(z) = {p}')
    if flag == 'sympy':
        q_a, q_b, p = sympy_det_coef(alssm, segment)
        print(f'Q(z, a) =  \n {q_a} \n  Q(z, b) =  \n {q_b} \n P(z) = {p}')
    if flag == 'scipy_det':
        q_a, q_b, p = scipy_det_coef(alssm, segment)
        print(f'Q(z, a) =  \n {q_a} \n  Q(z, b) =  \n {q_b} \n P(z) = {p}')

    FIR_b = shift_signal(FIR_filter(q_b, y * segment.gamma**segment.b), segment.b)
    # FIR_b = FIR_filter(q_b, shift_signal(y * segment.gamma**segment.b, segment.b))
    FIR_a = shift_signal(FIR_filter(q_a, y * segment.gamma**(segment.a-1)), segment.a-1)
    # FIR_a = FIR_filter(q_a, shift_signal(y * segment.gamma**(segment.a-1), segment.a-1))
    FIR_diff = FIR_b - FIR_a
    xi = IIR_filter(p, FIR_diff)
    xs = np.einsum('ij,kj->ki', Winv, xi)
    y_hat = cost.eval_alssm_output(xs)
    plt.plot(y_hat, c=colors[i], ls=lss[i], lw=1, label='lfilter coefficients ' + flag)

y_hat_true = cost.eval_alssm_output(xs_numpy)
plt.plot(y, c="grey", lw=0.8, label='raw')
plt.plot(y_hat_true, c="g", lw=0.8, label='reference')
plt.legend()
plt.show()
