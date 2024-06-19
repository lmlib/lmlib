import numpy as np

import lmlib
from lmlib.irrls.section import SectionBase
from lmlib.irrls.message_passing import init_bw_msg, init_fw_msg

__all__ = ['FactorGraph', 'MBF', 'BIFM']

MBF = 'MBF'
"""str : Modified Bryson–Frazier Message Passing Algorithm. :code:`lm.MBF='MBF'`"""

BIFM = 'BIFM'
"""str : Backward Recursion with Time-Reversed Information Filter, Forward Recursion with Marginals Message Passing Algorithm. :code:`lm.BIFM='BIFM'` """


class FactorGraph(object):
    r"""
    (Forney) factor graph to define a recursively computable cost function. 
    A :class:`.FactorGraph` object requires two additional objects to function:

    - A Section (of type :class:`.SectionBase`, or mostly of type :class:`.SectionContainer` containing of multiple :class:`.SectionBase` instances) that defines the state transition from :math:`x_k` to :math:`x_{k+1}`

    - A message passing method such as :class:`.MBF_MessagePassingBase` propagating Gaussian messages forward and backward along the graph.
    
    The :code:`FactorGraph.optimize(iterations=100)` runs 100 times the forward and backward message passing and calls the internal update routines of each Section to update internal states (if needed).
    After optimization, the optimized internal states might be accessed via generic and Section-specific access methods.
    
    For the example factor graph

    .. code-block:: python

       # sections
       sc_sys = lm.SectionSystem(A)

       sc_input = lm.SectionInput_NUV(B, save_input_estimate=True)

       sc_output = lm.SectionOutput(C, y, sigma2=1.0,save_output_estimate=True)

       sc = lm.SectionContainer([sc_sys, sc_input, sc_output], save_state_marginal=True)

       # message passing
       fg = lm.FactorGraph(sc)
       fg.initialize_mp(lm.MBF, K)
       fg.optimize(iterations=100)
    
    the following generic and Section-specific properties are accessible:
    
    .. code-block:: python

       X = sc.get_state_marginal() # access to state marginals after the section

       U = sc_input.get_input_marginal() # access to input marginals

       Yt = sc_output.get_output_marginal() # access to output marginals



    Parameters
    ----------
    section : SectionBase
        section for one time step (section)
    left_side_prior : tuple, optional
        left side prior with mean and covariance.
        Scalars will be multiplied by a mean vector with ones and an identity covariance. Default = (0, 1e9)
    left_side_prior : tuple, optional
        right side prior with precision-weighted mean and precision matrix (inverse of covariance matrix).
        Scalars will be multiplied by a  precision-weighted mean vector with ones and an identity precision matrix.
        Default = (0, 0)
    """

    def __init__(self, section, left_side_prior=(0, 1e9), right_side_prior=(0, 0)):
        self._N = None
        self._K = None
        self._mp_type = None

        self._section = section
        self._left_side_prior = left_side_prior
        self._right_side_prior = right_side_prior

    def initialize_mp(self, mp_type, K):
        """
        Message Passing initializations

        Parameters
        ----------
        mp_type : str
            Message Passing type weather 'MBF' or 'BIFM'
        K : int
            Length of Factor Graph
        """
        if not isinstance(K, int):
            raise ValueError("K is not an positive integer")
        self._K = K

        self._section._setup_mp(mp_type, self._K)
        self._mp_type = mp_type
        self._N = self._section.N

    def optimize(self, iterations=1):
        """optimize factor graph

        Parameters
        ----------
        iterations : int
            Number of iterations (default = 1).
            If `iteration` is 1 the solution equals to Modified Bryson–Frazier smoother.
            else if `iteration` is larger the variance updated in an Expectation Maximization (EM)
            or Alternate Maximization (AM) algorithm. This can be by the parameter `update_method`
            in the corresponding section.
        """

        # initialize forward and backward initial states
        msg_fw = init_fw_msg(self.N, self._left_side_prior)
        msg_bw = init_bw_msg(self.N, self._right_side_prior)

        mp = self._section.mp
        if self._mp_type == 'MBF':
            # iterations for EM Algorithm  (-1 for last propagation which saves states)
            for i in range(iterations - 1):
                for k in range(self.K):
                    mp.propagate_forward(k, msg_fw)
                for k in reversed(range(self.K)):
                    mp.propagate_backward(k, msg_bw)

            for k in range(self.K):
                mp.propagate_forward_save_states(k, msg_fw)
            for k in reversed(range(self.K)):
                mp.propagate_backward_save_states(k, msg_bw)

        if self._mp_type == 'BIFM':
            # iterations for EM Algorithm  (-1 for last propagation which saves states)
            for i in range(iterations - 1):
                for k in range(self.K):
                    mp.propagate_backward(k, msg_bw)
                for k in reversed(range(self.K)):
                    mp.propagate_forward(k, msg_fw)

            for k in range(self.K):
                mp.propagate_backward_save_states(k, msg_bw)
            for k in reversed(range(self.K)):
                mp.propagate_forward_save_states(k, msg_fw)
    @property
    def N(self):
        """int : Model Order"""
        return self._N

    @property
    def K(self):
        """int : Length of Factor-graph"""

        return self._K
