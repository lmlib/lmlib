"""Provides useful and specific applications based on :mod:`lmlib.statespace` methods""" 


from lmlib.statespace.cost import CompositeCost, Segment
from lmlib.statespace.model import AlssmPoly

from abc import ABC
import numpy as np

__all__ = ['TSLM']

class TSLM(ABC):
    """
    Full implementation of the Two-Sided Line Model (TSLM) as published in "Onset Detection of Pulse-Shaped Bioelectrical Signals Using Linear State Space Models" [Waldmann2022]_.
    
    Comprehensive examples on how to apply TSLMs are provided at :ref:`Onset and Peak Detection With Two-Sided Line Models (TSLMs) <onset>`.

    .. image:: /static/images/api/tslm_fig1.png
      :width: 400
      :alt: TSLM Fig. 1 [Waldmann2022]_

    .. image:: /static/images/api/tslm_tab2.png
      :width: 400
      :alt: TSLM Tab. 2 [Waldmann2022]_

    """

    H_Free = np.array([[1, 0, 0, 0],  # x_A,left : offset of left line
                       [0, 1, 0, 0],  # x_B,left : slope of left line
                       [0, 0, 1, 0],  # x_A,right : offset of right line
                       [0, 0, 0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Free"`, see [Waldmann2022]_."""

    H_Continuous = np.array(
        [[1, 0, 0],  # x_A,left : offset of left line
         [0, 1, 0],  # x_B,left : slope of left line
         [1, 0, 0],  # x_A,right : offset of right line
         [0, 0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Continuous"`, see [Waldmann2022]_."""

    H_Straight = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Straight"`, see [Waldmann2022]_."""

    H_Horizontal = np.array(
        [[1],  # x_A,left : offset of left line
         [0],  # x_B,left : slope of left line
         [1],  # x_A,right : offset of right line
         [0]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Horizontal"`, see [Waldmann2022]_."""

    H_Left_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 0],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Left_Horizontal"`, see [Waldmann2022]_."""

    H_Right_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 0]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Right_Horizontal"`, see [Waldmann2022]_."""

    H_Peak = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 1],  # x_B,left : slope of left line
                       [1, 0],  # x_A,right : offset of right line
                       [0, -1]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Peak"`, see [Waldmann2022]_."""

    H_Step = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 0],  # x_B,left : slope of left line
                       [0, 1],  # x_A,right : offset of right line
                       [0, 0]])  # x_B,right : slope of right line
    """:class:`numpy.ndarray` : Defines constrain matrix :math:`H` of type :code:`"Step"`, see [Waldmann2022]_."""
                       
                       
                       

    @staticmethod
    def create_cost(ab, gs):
        """
        Returns a TSLM (Two Sided Line Model) Cost Model

        Sets up Composite Cost (CC) Model using two ALSSM straight line models (AlssmPoly of order 2) and 2 concatenated segments with exponentially decaying windows to instantiate a TSLM. 


        .. code-block:: text


                        gs[0]  gs[1]  
                     __,--->   <---.__   
                    --------------------
                A_L |   c_L  |    0    |
                    --------------------
                A_R |   0    |    c_R  |
                    --------------------
                    :        :         :
                   a=ab[0]   0        b=ab[1]
        
        


        Parameters
        ----------
        ab : array_like of integer, of shape (2,)
            Array/Tuple of two integers defining the left and right boundary of the left line model and the right line model, respectively.
            (The right boundary of the left model is set to :code:`b=-1` and the left boundary of the right model to :code:`a=0`.)
        gs : array_like of float, of shape (2,)
            Array/Tuple of two floats defining the window weight of the left ( :code:`g_left = g[0]` )  and the right :code:`g_right = g[0]` line model,
            respectively.

        Returns
        -------
        out : CompositeCost 
              Object representing a TSLM (Two Sided Line Model)
        
        
        
        Example
        --------
        >>> ccost = lm.TSLM.create_cost( ab=(-15, 30), gs=(50, 50) )
        >>> print(ccost)
        CompositeCost(label=TSLM)
        └- ['AlssmPoly(A=[[1,1],[0,1]], C=[1,0], label=left line model)', 'AlssmPoly(A=[[1,1],[0,1]], C=[1,0], label=right line model)'],
        └- ['Segment(a=-15, b=-1, direction=fw, g=50, delta=0, label=left segment)', 'Segment(a=0, b=30, direction=bw, g=50, delta=0, label=right segment)']
        
        """
        return CompositeCost([AlssmPoly(1, label='left line model'), AlssmPoly(1, label='right line model')],
                             [Segment(ab[0], -1, 'fw', gs[0], label='left segment'),
                              Segment(0, ab[1], 'bw', gs[1], label='right segment')],
                             np.eye(2), label='TSLM')