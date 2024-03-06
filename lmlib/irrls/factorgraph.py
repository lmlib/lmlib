import numpy as np
from .block import BlockBase
from .message_passing import MessagePassing
from lmlib.utils.check import all_equal

__all__ = ['FactorGraph']


class FactorGraph(object):
    """
    Factor Graph for Message Passing

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

