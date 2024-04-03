import numpy as np
from .section import SectionBase
from .message_passing import MessagePassing, MassagePassingSection
from lmlib.utils.check import all_equal

__all__ = ['FactorGraph']


class FactorGraph(object):
    r"""
    (Forney) factor graph to define a recursively computable cost function. 
    A :class:`.FactorGraph` object requires two additional objects to function:

    - A Section (of type :class:`.Section`, or mostly of type :class:`.SectionContainer` containing of multiple :class:`.Section` instances) that defines the state transition from :math:`x_k` to :math:`x_{k+1}`

    - A message passing method such as :class:`.message_passing` propagating Gaussian messages forward and backward along the graph.
    
    The :code:`optimize(iterations=100)` runs 100 times the forward and backward message passing and calls the internal update routines of each Section to update internal states (if needed).
    After optimization, the optimized internal states might be accessed via generic and Section-specific access methods.
    
    For the example factor graph

    .. code-block:: python
    
       blk_system = lm.SectionSystem(A, label="system")
       blk_input = lm.SectionInputNUV(B, sigma2_init=1.0, estimate_input=True, save_deployed_sigma2=True)
       blk_output = lm.SectionOutput(C, sigma2_init=1.0, y=y, estimate_output=True)
       blk = lm.SectionContainer(Sections=[blk_system, blk_input, blk_output], save_marginals=True)

       # message passing
       fg = lm.FactorGraph(blk, lm.MBF, K)
       fg.optimize(iterations=100)
    
    the following generic and Section-specific properties are accessible:
    
    .. code-block:: python
    
       X = fg.get_mp_Section(blk).get_marginal() # access to marginals after each Section


    Parameters
    ----------
    section : SectionBase
        section for one time step (section)
    left_side_prior : tuple, optional
        left side prior with mean and covariance.
        Scalars will be multiplied by a mean vector with ones and an identity covariance.
    left_side_prior : tuple, optional
        right side prior with precision-weighted mean and precision matrix (inverse of covariance matrix).
        Scalars will be multiplied by a  precision-weighted mean vector with ones and an identity precision matrix.
    """

    def __init__(self, section, left_side_prior=(0, 1e9), right_side_prior=(0, 1e-9)):
        self._message_passing = None
        self.N = None
        self.K = 0
        self._mp_section = None

        self.section = section
        self._left_side_prior = left_side_prior
        self._right_side_prior = right_side_prior

    def _get_initialized_messages(self):
        msg_fw = self.message_passing.get_forward_initial_state(self.N, *self._left_side_prior)
        msg_bw = self.message_passing.get_backward_initial_state(self.N, *self._right_side_prior)
        return msg_fw, msg_bw

    def initialize_mp(self, message_passing, K):
        """
        Message Passing initialization

        Parameters
        ----------
        message_passing : MBF
            Message Passing object containing the solver
        K : int
            Length of Factor Graph


        """
        self.message_passing = message_passing
        self.K = K
        self.mp_section = message_passing.create_mp_section(self.section, K)
        self.N = self.mp_section.section.N

    @property
    def mp_section(self):
        return self._mp_section

    @mp_section.setter
    def mp_section(self, mp_section):
        assert isinstance(mp_section, MassagePassingSection), 'mp_section must be a subclass of MassagePassingSection'
        self._mp_section = mp_section

    @property
    def message_passing(self):
        return self._message_passing

    @message_passing.setter
    def message_passing(self, message_passing):
        assert issubclass(message_passing, MessagePassing)
        self._message_passing = message_passing

    def optimize(self, iterations=1):

        # initialize forward and backward initial states
        msg_fw, msg_bw = self._get_initialized_messages()

        # iterations for EM Algorithm  (-1 for last propagation which saves states)
        for i in range(iterations - 1):
            for k in range(self.K):
                self.mp_section.propagate_forward(k, msg_fw)
            for k in reversed(range(self.K)):
                self.mp_section.propagate_backward(k, msg_bw)

        for k in range(self.K):
            self.mp_section.propagate_forward_save_states(k, msg_fw)
        for k in reversed(range(self.K)):
            self.mp_section.propagate_backward_save_states(k, msg_bw)

    def get_mp_section(self):
        """
        Returns the message passing section of the factor graph

        Returns
        -------
        out : MessagePassingSection
            return the message passing section
        """
        return self._mp_section
