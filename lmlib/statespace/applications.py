from lmlib.statespace.cost import CompositeCost
from lmlib.statespace.segment import Segment
from lmlib.statespace.model import AlssmPoly

from abc import ABC
import numpy as np

__all__ = ['TSLM']

class TSLM(ABC):
    """
    Full implementation of the Two-Sided Line Model (TSLM) as published in 
    "Onset Detection of Pulse-Shaped Bioelectrical Signals Using Linear State Space Models"
    [Waldmann2022](../bibliography.md#waldmann2022).

    Comprehensive examples on how to apply TSLMs are provided at 
    [Onset and Peak Detection With Two-Sided Line Models](../examples/onset_detection.md).

    ![TSLM Fig. 1](../images/tslm_fig1.png)
    *Figure 1: Illustration of the Two-Sided Line Model.*

    ![TSLM Tab. 2](../images/tslm_tab2.png)
    *Table 2: Parameters of the TSLM.*
    """

    H_Free = np.array([[1, 0, 0, 0],  # x_A,left : offset of left line
                       [0, 1, 0, 0],  # x_B,left : slope of left line
                       [0, 0, 1, 0],  # x_A,right : offset of right line
                       [0, 0, 0, 1]])  # x_B,right : slope of right line
    """
    ```python
    H_Free = np.array([
        [1, 0, 0, 0],  # x_A,left : offset of left line
        [0, 1, 0, 0],  # x_B,left : slope of left line
        [0, 0, 1, 0],  # x_A,right : offset of right line
        [0, 0, 0, 1]   # x_B,right : slope of right line
    ])
    ```
    """

    H_Continuous = np.array(
        [[1, 0, 0],  # x_A,left : offset of left line
         [0, 1, 0],  # x_B,left : slope of left line
         [1, 0, 0],  # x_A,right : offset of right line
         [0, 0, 1]])  # x_B,right : slope of right line
    """
    ```python
    H_Continuous = np.array(
        [[1, 0, 0],  # x_A,left : offset of left line
         [0, 1, 0],  # x_B,left : slope of left line
         [1, 0, 0],  # x_A,right : offset of right line
         [0, 0, 1]])  # x_B,right : slope of right line
    ```
    """

    H_Straight = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    """
    ```python
    H_Straight = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    ```
    """

    H_Horizontal = np.array(
        [[1],  # x_A,left : offset of left line
         [0],  # x_B,left : slope of left line
         [1],  # x_A,right : offset of right line
         [0]])  # x_B,right : slope of right line
    """
    ```python
    H_Horizontal = np.array(
        [[1],  # x_A,left : offset of left line
         [0],  # x_B,left : slope of left line
         [1],  # x_A,right : offset of right line
         [0]])  # x_B,right : slope of right line
    ```
    """

    H_Left_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 0],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    """
    ```python
    H_Left_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 0],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 1]])  # x_B,right : slope of right line
    ```
    """

    H_Right_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 0]])  # x_B,right : slope of right line
    """
    ```python
    H_Right_Horizontal = np.array(
        [[1, 0],  # x_A,left : offset of left line
         [0, 1],  # x_B,left : slope of left line
         [1, 0],  # x_A,right : offset of right line
         [0, 0]])  # x_B,right : slope of right line
    ```
    """

    H_Peak = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 1],  # x_B,left : slope of left line
                       [1, 0],  # x_A,right : offset of right line
                       [0, -1]])  # x_B,right : slope of right line
    """
    ```python
    H_Peak = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 1],  # x_B,left : slope of left line
                       [1, 0],  # x_A,right : offset of right line
                       [0, -1]])  # x_B,right : slope of right line
    ```
    """

    H_Step = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 0],  # x_B,left : slope of left line
                       [0, 1],  # x_A,right : offset of right line
                       [0, 0]])  # x_B,right : slope of right line
    """
    ```python
    H_Step = np.array([[1, 0],  # x_A,left : offset of left line
                       [0, 0],  # x_B,left : slope of left line
                       [0, 1],  # x_A,right : offset of right line
                       [0, 0]])  # x_B,right : slope of right line
    ```
    """
                       
                       
                       

    @staticmethod
    def create_cost(ab, gs):
        """
        Returns a TSLM (Two Sided Line Model) Cost Model

        Sets up Composite Cost (CC) Model using two ALSSM straight line models (AlssmPoly of order 2) and 2 concatenated segments with exponentially decaying windows to instantiate a TSLM. 


        ```

                        gs[0]  gs[1]  
                     __,--->   <---.__   
                    --------------------
                A_L |   c_L  |    0    |
                    --------------------
                A_R |   0    |    c_R  |
                    --------------------
                    :        :         :
                   a=ab[0]   0        b=ab[1]
        ```
        
        


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
        ```python
        >>> ccost = lm.TSLM.create_cost( ab=(-15, 30), gs=(50, 50) )
        >>> print(ccost)
        CompositeCost(label=TSLM)
        ['AlssmPoly(A=[[1,1],[0,1]], C=[1,0], label=left line model)', 'AlssmPoly(A=[[1,1],[0,1]], C=[1,0], label=right line model)'],
        ['Segment(a=-15, b=-1, direction=fw, g=50, delta=0, label=left segment)', 'Segment(a=0, b=30, direction=bw, g=50, delta=0, label=right segment)']
        ```
        
        """
        return CompositeCost([AlssmPoly(1, label='left line model'), AlssmPoly(1, label='right line model')],
                             [Segment(ab[0], -1, 'fw', gs[0], label='left segment'),
                              Segment(0, ab[1], 'bw', gs[1], label='right segment')],
                             np.eye(2), label='TSLM')