import unittest
import lmlib as lm
import numpy as np


class TestRLSAlssm(unittest.TestCase):

    # single set
    def test_eval_error_single_output_ss(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False)
        xs = rls.filter_minimize_x(y)
        error = [0.00078396, 0.00175143, 0.00267136, 0.00290315, 0.00288889, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.00288889, 0.00290315, 0.00267136,
                 0.00175143, 0.00078396]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_multi_output_ss(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))[:, None]
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False)
        xs = rls.filter_minimize_x(y)
        error = [0.00078396, 0.00175143, 0.00267136, 0.00290315, 0.00288889, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.00288889, 0.00290315, 0.00267136,
                 0.00175143, 0.00078396]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_single_output_tf(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = [0.00078396, 0.00175143, 0.00267136, 0.00290315, 0.00288889, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.00288889, 0.00290315, 0.00267136,
                 0.00175143, 0.00078396]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_multi_output_tf(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))[:, None]
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = [0.00078396, 0.00175143, 0.00267136, 0.00290315, 0.00288889, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.00288889, 0.00290315, 0.00267136,
                 0.00175143, 0.00078396]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    # steady state single set
    def test_eval_error_single_output_steady_state_ss(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True)
        xs = rls.filter_minimize_x(y)
        error = [0.09217478, 0.17659045, 0.22973486, 0.18450274, 0.07276776, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.07276776, 0.18450274, 0.22973486,
                 0.17659045, 0.09217478]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_multi_output_steady_state_ss(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))[:, None]
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True)
        xs = rls.filter_minimize_x(y)
        error = [0.09217478, 0.17659045, 0.22973486, 0.18450274, 0.07276776, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.07276776, 0.18450274, 0.22973486,
                 0.17659045, 0.09217478]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_single_output_steady_state_tf(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = [0.09217478, 0.17659045, 0.22973486, 0.18450274, 0.07276776, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.07276776, 0.18450274, 0.22973486,
                 0.17659045, 0.09217478]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_multi_output_steady_state_tf(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 20))[:, None]
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = [0.09217478, 0.17659045, 0.22973486, 0.18450274, 0.07276776, 0.00629653,
                 0.02259697, 0.05396076, 0.08716118, 0.10819699, 0.10819699, 0.08716118,
                 0.05396076, 0.02259697, 0.00629653, 0.07276776, 0.18450274, 0.22973486,
                 0.17659045, 0.09217478]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    # multi set
    def test_eval_error_set_single_output_ss(self):
        y = np.column_stack([np.sin(np.linspace(0, 2 * np.pi, 20)),
                             1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))])
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False)
        xs = rls.filter_minimize_x(y)
        error = [
            [0.00078396, 0.00198539],
            [0.00175143, 0.00838537],
            [0.00267136, 0.02624551],
            [0.00290315, 0.06540435],
            [0.00288889, 0.13690168],
            [0.00629653, 0.24825427],
            [0.02259697, 0.21157827],
            [0.05396076, 0.14100976],
            [0.08716118, 0.06630881],
            [0.10819699, 0.01897823],
            [0.10819699, 0.01897823],
            [0.08716118, 0.06630881],
            [0.05396076, 0.14100976],
            [0.02259697, 0.21157827],
            [0.00629653, 0.24825427],
            [0.00288889, 0.13690168],
            [0.00290315, 0.06540435],
            [0.00267136, 0.02624551],
            [0.00175143, 0.00838537],
            [0.00078396, 0.00198539]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_set_multi_output_ss(self):
        y = np.dstack([np.sin(np.linspace(0, 2 * np.pi, 20)[:, None]),
                       1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))[:, None]])
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False)
        xs = rls.filter_minimize_x(y)
        error = [
            [0.00078396, 0.00198539],
            [0.00175143, 0.00838537],
            [0.00267136, 0.02624551],
            [0.00290315, 0.06540435],
            [0.00288889, 0.13690168],
            [0.00629653, 0.24825427],
            [0.02259697, 0.21157827],
            [0.05396076, 0.14100976],
            [0.08716118, 0.06630881],
            [0.10819699, 0.01897823],
            [0.10819699, 0.01897823],
            [0.08716118, 0.06630881],
            [0.05396076, 0.14100976],
            [0.02259697, 0.21157827],
            [0.00629653, 0.24825427],
            [0.00288889, 0.13690168],
            [0.00290315, 0.06540435],
            [0.00267136, 0.02624551],
            [0.00175143, 0.00838537],
            [0.00078396, 0.00198539]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_set_single_output_tf(self):
        y = np.column_stack([np.sin(np.linspace(0, 2 * np.pi, 20)),
                             1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))])
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = [
            [0.00078396, 0.00198539],
            [0.00175143, 0.00838537],
            [0.00267136, 0.02624551],
            [0.00290315, 0.06540435],
            [0.00288889, 0.13690168],
            [0.00629653, 0.24825427],
            [0.02259697, 0.21157827],
            [0.05396076, 0.14100976],
            [0.08716118, 0.06630881],
            [0.10819699, 0.01897823],
            [0.10819699, 0.01897823],
            [0.08716118, 0.06630881],
            [0.05396076, 0.14100976],
            [0.02259697, 0.21157827],
            [0.00629653, 0.24825427],
            [0.00288889, 0.13690168],
            [0.00290315, 0.06540435],
            [0.00267136, 0.02624551],
            [0.00175143, 0.00838537],
            [0.00078396, 0.00198539]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_set_multi_output_tf(self):
        y = np.dstack([np.sin(np.linspace(0, 2 * np.pi, 20)[:, None]),
                       1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))[:, None]])
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=False, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = [
            [0.00078396, 0.00198539],
            [0.00175143, 0.00838537],
            [0.00267136, 0.02624551],
            [0.00290315, 0.06540435],
            [0.00288889, 0.13690168],
            [0.00629653, 0.24825427],
            [0.02259697, 0.21157827],
            [0.05396076, 0.14100976],
            [0.08716118, 0.06630881],
            [0.10819699, 0.01897823],
            [0.10819699, 0.01897823],
            [0.08716118, 0.06630881],
            [0.05396076, 0.14100976],
            [0.02259697, 0.21157827],
            [0.00629653, 0.24825427],
            [0.00288889, 0.13690168],
            [0.00290315, 0.06540435],
            [0.00267136, 0.02624551],
            [0.00175143, 0.00838537],
            [0.00078396, 0.00198539]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    # steady state multi set
    def test_eval_error_set_single_output_steady_state_ss(self):
        y = np.column_stack([np.sin(np.linspace(0, 2 * np.pi, 20)),
                             1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))])
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True)
        xs = rls.filter_minimize_x(y)
        error =\
            [[0.09217478, 1.79342604]
            , [0.17659045, 1.28098271]
            , [0.22973486, 0.91485587]
            , [0.18450274, 1.04946409]
            , [0.07276776, 1.24696433]
            , [0.00629653, 0.24825427]
            , [0.02259697, 0.21157827]
            , [0.05396076, 0.14100976]
            , [0.08716118, 0.06630881]
            , [0.10819699, 0.01897823]
            , [0.10819699, 0.01897823]
            , [0.08716118, 0.06630881]
            , [0.05396076, 0.14100976]
            , [0.02259697, 0.21157827]
            , [0.00629653, 0.24825427]
            , [0.07276776, 1.24696433]
            , [0.18450274, 1.04946409]
            , [0.22973486, 0.91485587]
            , [0.17659045, 1.28098271]
            , [0.09217478, 1.79342604]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_set_multi_output_steady_state_ss(self):
        y = np.dstack([np.sin(np.linspace(0, 2 * np.pi, 20)[:, None]),
                       1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))[:, None]])
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True)
        xs = rls.filter_minimize_x(y)
        error = \
            [[0.09217478, 1.79342604]
                , [0.17659045, 1.28098271]
                , [0.22973486, 0.91485587]
                , [0.18450274, 1.04946409]
                , [0.07276776, 1.24696433]
                , [0.00629653, 0.24825427]
                , [0.02259697, 0.21157827]
                , [0.05396076, 0.14100976]
                , [0.08716118, 0.06630881]
                , [0.10819699, 0.01897823]
                , [0.10819699, 0.01897823]
                , [0.08716118, 0.06630881]
                , [0.05396076, 0.14100976]
                , [0.02259697, 0.21157827]
                , [0.00629653, 0.24825427]
                , [0.07276776, 1.24696433]
                , [0.18450274, 1.04946409]
                , [0.22973486, 0.91485587]
                , [0.17659045, 1.28098271]
                , [0.09217478, 1.79342604]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_set_single_output_steady_state_tf(self):
        y = np.column_stack([np.sin(np.linspace(0, 2 * np.pi, 20)),
                             1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))])
        alssm = lm.AlssmPoly(poly_degree=2)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = \
            [[0.09217478, 1.79342604]
                , [0.17659045, 1.28098271]
                , [0.22973486, 0.91485587]
                , [0.18450274, 1.04946409]
                , [0.07276776, 1.24696433]
                , [0.00629653, 0.24825427]
                , [0.02259697, 0.21157827]
                , [0.05396076, 0.14100976]
                , [0.08716118, 0.06630881]
                , [0.10819699, 0.01897823]
                , [0.10819699, 0.01897823]
                , [0.08716118, 0.06630881]
                , [0.05396076, 0.14100976]
                , [0.02259697, 0.21157827]
                , [0.00629653, 0.24825427]
                , [0.07276776, 1.24696433]
                , [0.18450274, 1.04946409]
                , [0.22973486, 0.91485587]
                , [0.17659045, 1.28098271]
                , [0.09217478, 1.79342604]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())

    def test_eval_error_set_multi_output_steady_state_tf(self):
        y = np.dstack([np.sin(np.linspace(0, 2 * np.pi, 20)[:, None]),
                       1.5 * np.cos(np.linspace(0, 2 * np.pi, 20))[:, None]])
        alssm = lm.AlssmPoly(poly_degree=2, force_MC=True)
        segment_fw = lm.Segment(a=-5, b=-1, direction=lm.FW, g=10, delta=0)
        segment_bw = lm.Segment(a=0, b=5, direction=lm.BW, g=10, delta=0)
        cost = lm.CompositeCost([alssm], [segment_fw, segment_bw], F=[[1, 1]])
        rls = lm.RLSAlssm(cost, steady_state=True, backend='lfilter')
        xs = rls.filter_minimize_x(y)
        error = \
            [[0.09217478, 1.79342604]
                , [0.17659045, 1.28098271]
                , [0.22973486, 0.91485587]
                , [0.18450274, 1.04946409]
                , [0.07276776, 1.24696433]
                , [0.00629653, 0.24825427]
                , [0.02259697, 0.21157827]
                , [0.05396076, 0.14100976]
                , [0.08716118, 0.06630881]
                , [0.10819699, 0.01897823]
                , [0.10819699, 0.01897823]
                , [0.08716118, 0.06630881]
                , [0.05396076, 0.14100976]
                , [0.02259697, 0.21157827]
                , [0.00629653, 0.24825427]
                , [0.07276776, 1.24696433]
                , [0.18450274, 1.04946409]
                , [0.22973486, 0.91485587]
                , [0.17659045, 1.28098271]
                , [0.09217478, 1.79342604]]
        self.assertTrue(np.isclose(rls.eval_errors(xs), error).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
