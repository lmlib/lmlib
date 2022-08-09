from lmlib.statespace.cost import CompositeCost, Segment
from lmlib.statespace.model import AlssmPoly

from abc import ABC
import numpy as np

__all__ = ['TSLM']

class TSLM(ABC):
    """
    Two Sided Line Model

    DraftDoc

    """

    H_Free = np.array([[1, 0, 0, 0],  # x_A,left : offset of left line
                       [0, 1, 0, 0],  # x_B,left : slope of left line
                       [0, 0, 1, 0],  # x_A,right : offset of right line
                       [0, 0, 0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type free"""

    H_Continuous = np.array(
        [[1, 0, 0],  # x_A,left : offset of left line
         [0, 1, 0],  # x_B,left : slope of left line
         [1, 0, 0],  # x_A,right : offset of right line
         [0, 0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type continuous"""

    H_Straight = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type straight"""

    H_Horizontal = np.array(
        [[1],  # x_A,left : offset of left line
         [0],  # x_B,left : slope of left line
         [1],  # x_A,right : offset of right line
         [0]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type horizontal"""

    H_Left_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 0],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type left horizontal"""

    H_Right_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 0]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type right horizontal"""

    H_Peak = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 1],  # x_B,left : slope of left line
                       [1, 0],  # x_A,right : offset of right line
                       [0, -1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type peak"""

    H_Step = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 0],  # x_B,left : slope of left line
                       [0, 1],  # x_A,right : offset of right line
                       [0, 0]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : H Constrain Matrix type step"""

    @staticmethod
    def create_cost(ab, gs):
        """
        Returns a TSLM (Two Sided Line Model) Cost Model

        DraftDoc

        The TSLM cost model is setup by two Line Models (AlssmPoly of order 2) and  by left- and a right-sided Segments.
        The Segments have no gap inbetween, meaning the left segment has a right boundary at -1 and the right one a
        left boundary at 0.

        Parameters
        ----------
        ab : array_like of integer, of shape (2,)
            Array/Tuple of two integers defining the left boundary of the left line model and the right boundary of the right line
            model, respectively. The right boundary of the left model is set to -1 and the left boundary of the right
            model to 0.
        gs : array_like of integer, of shape (2,)
            Array/Tuple of two integers defining window weight `g` of the left line model and the right line model,
            respectively.

        Returns
        -------
        out : CompositeCost
            TSLM (Two Sided Line Model) Cost Model
        """
        return CompositeCost([AlssmPoly(1), AlssmPoly(1)],
                             [Segment(ab[0], -1, 'fw', gs[0]), Segment(0, ab[1], 'bw', gs[1])],
                             np.eye(2))