.. _lmlib_getting_started:

Getting Started
===============

Localized Model Approximation
-----------------------------

Introduction: Least Square Approximation
****************************************
Least squares signal approximation is a method used to find the best-fitting function that minimizes the sum of the squared error (SE) between observed data points and a proposed model.
The cost equation is

.. math:: J(x) = \sum_{i=a}^b (s_i(x) - y_i)^2
   :label: SE

where :math:`a` and :math:`b` are the range of the SE sum, :math:`s_i(x)` is the model which the unknown (to be estimated) parameter :math:`x` and :math:`y_i` the observed data/signal.
Least Square Optimization is widely applied in signal processing to approximate signals with simpler forms or basis functions.

Localized Model Approximations
******************************
Localized model approximation uses the least square approximation in combination of a sliding window.
We can extend :eq:`SE` with the window weight :math:`w(\cdot)` and the localization index :math:`k` to

.. math:: J(x) = \sum_{i=k+a}^{k+b} w(i-k) (s_{i-k}(x_k) - y_i)^2 \quad .
   :label: LocalizedSE

Note that the unknown parameters are :math:`x_k` for :math:`k = 0, 1, \dots, K-1`, where :math:`K` is the data/signal length.

Figure 1 demonstrated a localized polynomial model approximation of an signal :math:`y`.
The boundaries :math:`k+a` and :math:`k+b` depicts the sliding window, and the blue line the fitted model.

.. image:: /static/getting_started/animation_approx.gif
  :width: 800


Autonomous Linear State Space Model
***********************************
The State-Space representation is well-known in signal processing and it is commonly used in Kalman filter and control systems.
The standard equations are

.. math::
    x_{k+1} &= A x_{k} + B u_k \\
    y_{k+1} &= C x_{k+1} + D u_{k+1}

where :math:`x_k` is the state-space vector, :math:`u_k` the input vector, :math:`y_k` the output vector, :math:`A` the state-transition / system matrix, :math:`B` the input matrix,
:math:`C` the output matrix, :math:`D` the feed-through matrix.
The use of autonomous state-space representation rejects the input vector and is composed of

.. math::
    x_{k+1} &= A x_{k} \\
    y_{k+1} &= C x_{k+1} \quad .

We are able to write the equations into a recursive form by using the matrix power:

.. math::
    x_{i} &= A^i x_{0} \\
    s_i(x_0) &= CA^i x_{0}


where :math:`x_{0}` is the initial state which defines the model, solely.

*lmlib* uses such predefined autonomous linear state space models, short **ALSSM** and is able to create compositions of different models.
Commonly used models are:
    - Polynomial Models :class:`~lmlib.AlssmPoly` and :class:`~lmlib.AlssmPolyJordan`
    - Exponential Models :class:`~lmlib.AlssmExp`
    - Sinusoidal Models :class:`~lmlib.AlssmSin`
    - Native Models :class:`~lmlib.Alssm`

Such a model can be created with

.. code-block::

    import lmlib as lm
    alssm = lm.AlssmPoly(3, label='polynomial-model')
    print(alssm)

.. exec::
    import lmlib as lm
    alssm = lm.AlssmPoly(3, label='polynomial-model')
    print(alssm)

Exponential Window Weights
**************************
There are various types of window functions (Hamming, Boxcar, Kaiser, Gaussian, etc.) which are commonly used in a sliding window practise.
Due to the received recursive ALSSM form, we use a recursive window function which simply boils down to a exponential function

.. math::
    w(i) = \gamma^i \quad .

*lmlib* uses :class:`~lmlib.Segment` object which defines not only the weight :math:`\gamma` but also gives information about the boundaries :math:`a` and :math:`b`.
Instead of using :math:`\gamma` the  :class:`~lmlib.Segment` expects the parameter :py:attr:`g` which implicit defines :math:`\gamma`, depending on the specified calculation :py:attr:`direction`.
(The direction defines if the segment is calculated forwards or backward in the cost recursions and will be discussed later).
:py:attr:`g` illustrates the area under the window curve :math:`[a, b)`. The exponential weight is therefore given

.. math::

    \overrightarrow{\gamma} &=  \frac{g}{g-1} \quad (\textsf{forward}) \\
    \overleftarrow{\gamma} &= \frac{g-1}{g} \quad (\textsf{backward})

.. code-block::

    import lmlib as lm

    seg = lm.Segment(a=-30, b=0, direction=lm.FW, g=80, label='left-sided-segment')
    print(seg)

.. exec::
    import lmlib as lm
    seg = lm.Segment(a=-30, b=0, direction=lm.FW, g=80, label='left-sided-segment')
    print(seg)


Building the Cost Models Using ALSSMs and Segments
**************************************************
The simplest cost model is a relation between one ALSSM and one segment.
:class:`~lmlib.CostSegment` creates such a behaviour as the following code shows.


.. code-block::

    import lmlib as lm
    alssm = lm.AlssmPoly(3, label='polynomial-model')
    seg = lm.Segment(a=-30, b=0, direction=lm.FW, g=80, label='left-sided-segment')
    cost = lm.CostSegment(alssm, seg)
    print(cost)


Recursive Optimization
**********************
