import numpy as np
from .block import BlockBase
from .message_passing import MessagePassing
from lmlib.utils.check import all_equal

__all__ = ['FactorGraph']


class FactorGraph(object):
    r"""
    (Forney) factor graph to define a recursively computable cost function. 
    A :class:`.FactorGraph` object requires two additional objects to function:

    - A block (of type :class:`.Block`, or mostly of type :class:`.BlockContainer` containing of multiple :class:`.Block` instances) that defines the state transition from :math:`x_k` to :math:`x_{k+1}`

    - A message passing method such as :class:`.message_passing` propagating Gaussian messages forward and backward along the graph.
    
    The :code:`optimize(iterations=100)` runs 100 times the forward and backward message passing and calls the internal update routines of each block to update internal states (if needed).
    After optimization, the optimized internal states might be accessed via generic and block-specific access methods.
    
    For the example factor graph

    .. code-block:: python
    
       blk_system = lm.BlockSystem(A, label="system")
       blk_input = lm.BlockInputNUV(B, sigma2_init=1.0, estimate_input=True, save_deployed_sigma2=True)
       blk_output = lm.BlockOutput(C, sigma2_init=1.0, y=y, estimate_output=True)
       blk = lm.BlockContainer(blocks=[blk_system, blk_input, blk_output], save_marginals=True)

       # message passing
       fg = lm.FactorGraph(blk, lm.MBF, K)
       fg.optimize(iterations=100)
    
    the following generic and block-specific properties are accessible:
    
    .. code-block:: python
    
       X = fg.get_mp_block(blk).get_marginals() # access to marginals after each block
    
    
    

    Parameters
    ----------
    block : BlockBase
        block for one time step (section)
    message_passing : MBF
        Message Passing object containing the solver
    K : int
        Length of Factor Graph
    """

    def __init__(self, block, message_passing, K):
        # self.block = block
        self.message_passing = message_passing
        self.K = K
        self.mp_block = message_passing.create_mp_block(block, K)
        self.N = self.mp_block.block.N

    @property
    def message_passing(self):
        return self._message_passing

    @message_passing.setter
    def message_passing(self, message_passing):
        assert issubclass(message_passing, MessagePassing)
        self._message_passing = message_passing

    def optimize(self, iterations=1, init_msg_fw=None, init_msg_bw=None):

        # initialize forward and backward initial states
        if init_msg_fw is None:
            msg_fw = self.message_passing.get_forward_initial_state(self.N)
        else:
            msg_fw = init_msg_fw

        if init_msg_bw is None:
            msg_bw = self.message_passing.get_backward_initial_state(self.N)
        else:
            msg_bw = init_msg_bw

        # iterations for EM Algorithm  (-1 for last propagation which saves states)
        for i in range(iterations - 1):
            for k in range(self.K):
                self.mp_block.propagate_forward(k, msg_fw)
            for k in reversed(range(self.K)):
                self.mp_block.propagate_backward(k, msg_bw)

        for k in range(self.K):
            self.mp_block.propagate_forward_save_states(k, msg_fw)
        for k in reversed(range(self.K)):
            self.mp_block.propagate_backward_save_states(k, msg_bw)

    def get_mp_block(self, block):
        """
        Returns the Message Passing Block of the corresponding cost describing Block by label or instance

        Parameters
        ----------
        block : str or BlockBase
            label of the block or the block instance itself

        Returns
        -------
        out : MessagePassingBlock
            return the corresponding message passing block
        """
        # create a mutable variable to store the mp_block into
        out_dict = dict(mp_block=None)

        # case if block is an instance of BlockBase
        if isinstance(block, BlockBase):
            self.mp_block.get_block_by_obj(block, out_dict)
            return out_dict["mp_block"]

        # case if block is a block label
        if isinstance(block, str):
            self.mp_block.get_block_by_label(block, out_dict)
            return out_dict["mp_block"]

