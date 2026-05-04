# State-Space Tutorial

This is a beginner's tutorial to start using the model-based signal
processing library lmlib.

To start, [install lmlib](installation.md) .

``` ipython3
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils.generator import load_lib_csv
```

## Description

This tutorial takes the following steps:

1. Loading ECG signal (from the lmlib library);
2. Setting up an autonomous linear state space model (ALSSM) for a polynomial model;
3. Setting up segments and defining the window shape for each segment;
4. Assigning the model to the segments;
5. Setting up the cost function;

## A simple example

### 1. Loading ECG signal

We use an exemplary electrocardiogram (ECG) signal, provided by lmlib;
the function `load_lib_csv` returns such an ECG sniped in a
1-dimensional numpy array.

``` ipython3
y = load_lib_csv('EECG_FILT_1CH_10S_FS2400HZ.csv')
K = len(y)  # read number of samples
k = np.arange(K)  # generate discrete-time index array

# plotting the signal
fig, ax = plt.subplots(figsize=(7, 1.7))
ax.plot(k, y)
plt.show()
```

<!--
![image](state-space-tutorial_files/state-space-tutorial_5_0.png)
-->

### 2. Setting up an ALSSM model

An ALSSM model can produce a wide class of discrete-time signals like
*polynomial*, *sinusoidal*, and *exponential* shapes, including linear
combinations and multiplications of those.

#### Theory:

The output sequence of an ALSSM is given, for an initial state $x_0$:

`\begin{equation*}
    s_k(x_0) = c A^kx_0 \ ,       
\end{equation*}`{.interpreted-text role="raw-latex"}

where `` \begin{equation}
s_k(x_0) \in \mathbb{R} 
\end{equation}`is the output at time index  ``{.interpreted-text
role="raw-latex"}[k]{.title-ref}\`;

:raw-latex:\`begin{equation} c in mathbb{R}\^{1 times N}
end{equation}\`is the output vector;

:raw-latex:\`begin{equation} A in mathbb{R}\^{N times N}
end{equation}\`is the state-transition matrix, and

:raw-latex:\`begin{equation} x_k = A\^kx_0 in mathbb{R}\^N
end{equation}\`is the state space vector (independent variable);

#### Application:

We assume here that the signal can be approximated by a 3th order
polynomial. Such a model corresponds to the following state-transition
matrix and output vector: `\begin{equation}
A = \begin{bmatrix}1 & 1 & 1 & 1\\ 0 & 1 & 2 & 3 \\ 0 & 0 & 1 & 3\\ 0 & 0 &0 &1\end{bmatrix}
\end{equation}`{.interpreted-text role="raw-latex"}

and `\begin{equation}
c=\begin{bmatrix}1 & 0 & 0 & 0\end{bmatrix}.
\end{equation}`{.interpreted-text role="raw-latex"}

Using the lmlib library, this can be simply generated with the following

``` ipython3
alssm_poly = lm.AlssmPoly(poly_degree=1)
alssm_poly.update() ## still needed?
print(alssm_poly)
```

<!-->
::: parsed-literal
AlssmPoly(A=\[\[1,1\],\[0,1\]\], C=\[1,0\], label=n/a)
:::
<!-->

### 3. Setting up segments and defining the window shape

A segment provides localization to a ALSSM, i.e. it adds a weighted
window of finite or infinite borders.

A segment consists of interval borders and a window shape:

- The segment boundaries: `a` and `b`: they can be declared as finite or
  infinite (at least on of needs to be finite).
- The window shape acts as a weight in the cost function. We note that a
  direction for the recursion is needed.

Several segments can be combined to model different components of the
signal. This freedom allows us to achieve a high modular signal
processing.

Let us generate two segments - a left-sided segment where the window
weight lower as we go lef, and a right-sided segment where the window
weight lowe as we go right.

``` ipython3
a = -300
b = 300
segment_left = lm.Segment(a=a, b=0, direction=lm.FORWARD, g=20)
segment_right = lm.Segment(a=1, b=b, direction=lm.BACKWARD, g=20)
```

#### Visualization of the window shape

``` ipython3
# choice of the reference time indices (for visualization purposes only):
k0 = [10000, 11000]
```

``` ipython3
# define the window shape at these time indices
win_lr= lm.map_windows([segment_left.window(), segment_right.window()], k0, K, merge_ks=True)
```

``` ipython3
# plotting segments and signal
_, axs = plt.subplots(2, 1, sharex='all')
axs[0].plot(k, win_lr[0], label="left-sided segment")
axs[0].plot(k, win_lr[1], label="right-sided segment")
axs[1].plot(k, y)
axs[0].legend()

plt.xlim(k0[0]+2*a, k0[-1]+b) # show just a specific time interval of the time series
plt.show()
```
<!--
![image](state-space-tutorial_files/state-space-tutorial_15_0.png)
-->

### 4. Assign Model to Segment

The assignment is done with a matrix `F`. This notation enables us to
describe in a concise way which model is assigned to each segment. For
instance, if we have data that we would like to represent with a
combination of 2 models (M1, M2) over 3 segments (left, middle, right).
Assuming that the left and right segments can be modeled with M1, while
the middle segment is a composite model made of M1 and M2, the mapping
matrix will have this shape: `\begin{equation}
F = \begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \ ,
\end{equation}`{.interpreted-text role="raw-latex"} where the columns
refer to the models and the row to the segments.

In our case, we assign the same model to both segment, as follows:

``` ipython3
F = [[1, 1]]
```

### 5. Setting up the Cost Function

defining the squared error cost function to be minimized when doing the
model fit by combining the previously defined ALSSM with the segment
configuration.

The cost function is given for each time index`k`.

The cost function is obtained by computing the squared error between the
model output ($s_{j-k}(x_k)$) and the signal, over the window
($w_{j-k}$) defined over the segment boundaries (a,b):

`\begin{equation}
    J_a^b(k,x_k) = \sum_{j=k+a}^{k+b} w_{j-k}(y_j - cA^{j-k}x_k)^2,
\end{equation}`{.interpreted-text role="raw-latex"} where `x_k` is the
state vector containg the parameters that need to be tuned to minimize
the cost function. The value of this state vector can be different for
every time index k.

This cost function is for one segment only. In case of multiple
segments, we define the *composite cost*, where `p` runs over the number
of segments: `\begin{equation}
    J_P(k, x_k) = \sum_{p=1}^P J_{a_p}^{b_p} (k,x_k).
\end{equation}`{.interpreted-text role="raw-latex"}

Within the lmlib library, setting up the cost function is done in two
steps. We first define the part that contain all the model and window
parameters and then we link it to the signal. This separation allows us
to distinguish and more easily handle the engineering part from the
signal itself.

The model and window parameters can be called in the method
CompositeCost:

``` ipython3
cost = lm.CompositeCost((alssm_poly,), (segment_left, segment_right), F)
print(cost)
```


<!-->
::: parsed-literal

CompositeCost(label=n/a)

:   └- \[\'AlssmPoly(A=\[\[1,1\],\[0,1\]\], C=\[1,0\], label=n/a)\'\],
    └- \[\'Segment(a=-300, b=0, direction=fw, g=20, delta=0,
    label=n/a)\', \'Segment(a=1, b=300, direction=bw, g=20, delta=0,
    label=n/a)\'\]
:::
<!-->

As you can see by printing the output of this method, all the parameters
chosen in our model are summarized in this variable.

The next step is to link this variable to the signal itself. This is
done by defining an instance of the class Recursive Least Square Alssm
and calling the method filter from the lmlib library. The method
filter(y) stores internal variables that will be needed later on to
minimize the cost function:

``` ipython3
rls = lm.RLSAlssm(cost)
rls.filter(y)
```

### 6. Fit the model parameters on the data

We aim at finding the optimal state vector parameters that minimize the
cost function. These parameters form the initial state of the ALSSM
($x_0$), whose value at any time index k can be recursively computed.

Here we use the output of the filter method to find the optimal $x_0$,
whose parameters minimize the cost function. This is done by using e.g.,
minimize_x().

``` ipython3
xs = rls.minimize_x()
```

#### Visualize the result of the fit

``` ipython3
#plot the trajectories at specific time indices
trajs = lm.map_trajectories(cost.trajectories(xs[k0]), k0, K, True, True)
_, axs = plt.subplots(1, 1, sharex='all')
axs.plot(k, y, c='grey', lw=0.3, label="data")
axs.plot(k, trajs, lw=0.8, color="blue", label=r'trajectory at k0=%s'%k0)
axs.legend()
axs.set_xlabel("time index")
plt.xlim(k0[0] + 2*a, k0[-1] + 2*b)
plt.show()
```

<!--
![image](state-space-tutorial_files/state-space-tutorial_26_0.png)
-->


``` ipython3
#plot the full estimate
_, axs = plt.subplots(1, 1, sharex='all')
axs.plot(k, y, c='grey', lw=0.3, label="data")
axs.plot(k, cost.eval_alssm_output(xs), lw=0.8, color="blue", label=r'$\hat{y}$')
axs.legend()
axs.set_xlabel("time index")
plt.xlim(k0[0] + 2*a, k0[-1] + 2*b)
plt.show()
```

<!--
![image](state-space-tutorial_files/state-space-tutorial_27_0.png)
-->
