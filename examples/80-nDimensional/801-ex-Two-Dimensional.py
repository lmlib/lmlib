"""
Text Recognition Ex.801
-----------------------
@author: christof, frederic

"""
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
# import lmlib_multivar_V_2_0_4 as lmmulti
import copy
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import random
from lmlib.statespace.backend import get_backend, BACKEND_TYPES, available_backends

from lmlib import RLSAlssm


# import colorcet as cc


# Create a reference / interferer image with a given letter
def create_letter_image(letter):
    img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 0))  #
    draw = ImageDraw.Draw(img)

    # Load a font
    # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 80, encoding="unic")
    font = ImageFont.truetype("/System/Library/Fonts/Courier.ttc", 80, encoding="unic")

    # Calculate the size of the text to center it
    text = letter
    position = (10, 0)

    # Draw the letter 'A' in the center
    draw.text(position, text, font=font, fill=(0, 0, 0, 255), align='center')  # Fill with white color

    return img


# Create the synthetic image
def create_synthetic_image_randomletters(grid_size=(8, 4), cell_size=(100, 100)):
    grid_image = Image.new('RGBA', (grid_size[0] * cell_size[0], grid_size[1] * cell_size[1]), color=(255, 255, 255, 0))
    # refletter='Z'
    refletter = 'B'
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Randomly rotate and stretch the reference image
            angle = random.uniform(-5, 5)  # Random rotation between -30 and 30 degrees
            scale = random.uniform(1.0, 1.3)  # Random scaling between 80% and 120%
            xpos = random.uniform(-30, 30)  # random position
            ypos = random.uniform(-30, 30)  # random position
            letter = random.choice(alphabet)

            if i == 0 and j == 0:
                angle = 0;
                scale = 1;
                xpos = 0;
                ypos = 0  # fix for reference image
                letter = refletter
            if (i == 5 and j == 1) or (i == 1 and j == 2) or (i == 3 and j == 3):
                letter = refletter
            if (i == 6 and j == 0):
                letter = refletter
                angle = -10
                scale = 1.0  # 1.05
            image = create_letter_image(letter)
            transformed_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
            transformed_image = transformed_image.resize(
                (int(transformed_image.width * scale), int(transformed_image.height * scale)), Image.BICUBIC)

            # Calculate position to paste the transformed image into the grid
            pos_x = int(xpos) + i * cell_size[0] + (cell_size[0] - transformed_image.width) // 2
            pos_y = int(ypos) + j * cell_size[1] + (cell_size[1] - transformed_image.height) // 2

            # grid_image.paste(transformed_image, (pos_x, pos_y))
            grid_image.alpha_composite(transformed_image, (pos_x, pos_y))

    return grid_image


alphabet = 'abcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ'

dpi = 300
# plt.close("all")
celllength = 120
synthetic_image = create_synthetic_image_randomletters(cell_size=(celllength, celllength))
Y_nonoise = np.array(synthetic_image)[:, :, 3] * 1.0
Y_nonoise = Y_nonoise / np.nanmax(Y_nonoise)
K1 = Y_nonoise.shape[0]
K2 = Y_nonoise.shape[1]
k1 = np.arange(K1)
k2 = np.arange(K2)

Y_image_gaussian = np.copy(Y_nonoise)

# Gaussian Noise (row-wise)
rows, cols = Y_image_gaussian.shape[:2]
for i in range(rows):
    std_dev = (i / rows) * 1.0 * 0.125  # Noise level increases with row index
    noise = np.random.normal(0.0, std_dev, cols)
    Y_image_gaussian[i] = np.clip(Y_image_gaussian[i] + noise, 0, 1)

Y = Y_image_gaussian.copy()
Ynans = copy.deepcopy(Y)
Ynans[Ynans < 0.1] = np.NaN

plot_raw = False
if plot_raw:
    # Display the synthetic image
    plt.figure(figsize=(8, 8))
    plt.imshow(synthetic_image, cmap='gray')
    # plt.axis('off')
    plt.show()

K1_REF = 55  # letter pixel position, x
K2_REF = 47  # letter pixel position, y

# ALSSM Definition
g = 100
l_side = 35
poly_degree = 2
alssm_1dpoly = lm.AlssmPolyJordan(poly_degree=poly_degree)
alssm_k1 = alssm_1dpoly
alssm_k2 = alssm_1dpoly
segment_left_k1 = lm.Segment(a=-l_side, b=-1, direction=lm.FORWARD, g=g)
segment_right_k1 = lm.Segment(a=0, b=l_side, direction=lm.BACKWARD, g=g)
segment_left_k2 = lm.Segment(a=-l_side, b=-1, direction=lm.FORWARD, g=g)
segment_right_k2 = lm.Segment(a=0, b=l_side, direction=lm.BACKWARD, g=g)
F = [[1, 0],  # mixing matrix, turning on and off models per segment (1=on, 0=off)
     [0, 1]]

print('start alssm')
# filter signal in first dimension
cost_1d = lm.CompositeCost((alssm_k1, alssm_k1), (segment_left_k1, segment_right_k1), F)
cost_2d = lm.CompositeCost((alssm_k1, alssm_k1), (segment_left_k1, segment_right_k1), F)
consts_12d = [cost_1d, cost_2d]

class RLSAlssmND:

    def __init__(self, cost_models, dimension_order=None, filter_form='auto', backend=None):

        self.cost_models = cost_models
        self.dimension_order = dimension_order if dimension_order is not None else list(range(len(cost_models)))
        self.filter_form = filter_form
        self._backend = backend if backend else get_backend()
        self._is_multichannel = None

        self.Ns = [cm.get_model_order() for cm in self.cost_models]
        self._xi0 = None
        self._xi1 = None
        self._xi2 = None

        self.betas = None

    @property
    def cost_models(self):
        """CostBase : Cost Model"""
        return self._cost_model

    @cost_models.setter
    def cost_models(self, cost_models):
        assert isinstance(cost_models, (list, tuple)), 'cost_models is not a instance of List/Tuple'
        self._cost_models = cost_models

    def get_model_orders(self):
        return self.Ns

    @property
    def filter_form(self):
        """str : Set the form of filter to be used. Options:'parallel', 'cascade' 'auto' (Default)"""
        return self._filter_form

    @filter_form.setter
    def filter_form(self, filter_form):
        assert filter_form in ('parallel', 'cascade',
                               'auto'), 'Unknown filter_form value. Options: parallel, cascade, auto.'
        self._filter_form = filter_form

    def _check_output_dimensions(self, y):
        C_nDims = []
        for cm in self.cost_models:
            C = lm.AlssmSum(cm.alssms).C
            C_nDims.append(np.ndim(C))

        assert np.all(C_nDims == C_nDims[0]), "Output dimension of C-Matrix is not consistent over all dimensions."

        self._is_multichannel = C_nDims[0] == 2

        if self._is_multichannel:
                assert C_nDims[0] == np.shape(y)[-1], \
                    'Model output and observation shape does not match. ' \
                    'Multi-channel system expect shapes: ' \
                    'C shape (L, N) and y shape (K1,..., L)'
        else:
            assert np.ndim(y) == len(self.cost_models), \
                'Model output and observation shape does not match. ' \
                'Scalar systems (non multi-channel/set) expect shapes:' \
                'C shape (N,) and y shape (K1,,...)'


    def _allocate_parameter_storage(self, input_shape):
        Ks  = input_shape if ~self._is_multichannel else input_shape[:-1]
        Ns = self.get_model_orders()

        if self._calc_W and not self._steady_state:
            self._xi2 = np.zeros([N ** 2 for N in Ns])

        if self._steady_state:
            self._xi2 = np.zeros([N ** 2 for N in Ns])

        if self._calc_xi:
            self._xi1 = np.zeros(Ks + (np.prod(Ns),))

        if self._calc_kappa:
            self._xi0 = np.zeros(Ks)

        if self._calc_nu:
            self._nu = np.zeros(Ks)

    def _temporary_parameter_storage(self, input_shape):
        Ks = input_shape if ~self._is_multichannel else input_shape[:-1]
        Ns = self.get_model_orders()

        _xi2, _xi1, _xi0, _nu = None, None, None, None

        if self._calc_W and not self._steady_state:
            _xi2 = np.zeros([N ** 2 for N in Ns])

        if self._steady_state:
            _xi2 = np.zeros([N ** 2 for N in Ns])

        if self._calc_xi:
            _xi1 = np.zeros(Ks + (np.prod(Ns),))

        if self._calc_kappa:
            _xi0 = np.zeros(Ks)

        if self._calc_nu:
            _nu = np.zeros(Ks)

        return _xi2, _xi1, _xi0, _nu

    def n_rls(self, cost_model, k_indices, Y, V):
        rls_tmp = RLSAlssm(cost_model,
                           steady_state=self.steady_state,
                           calc_W=self.calc_W,
                           calc_xi=self.calc_xi,
                           calc_kappa=self.calc_kappa,
                           calc_nu=self.calc_nu,
                           kappa_diag=False,
                           betas=self.betas,
                           filter_form=self.filter_form,
                           backend=self._backend)

        rls_tmp.filter(Y[k_indices], V[k_indices])
        return rls._xi2, rls._xi1, rls._xi0, rls._nu

    def filter(self, Y):
        self._check_output_dimensions(Y)
        self._allocate_parameter_storage(np.shape(Y))

        for cost_model in self.cost_models:
            cm = cost_model
            xi2, xi1, xi0, nu = _temporary_parameter_storage(Y_xi_shape)
            xi2[k_indices], xi1[k_indices], xi0[k_indices], nu[k_indices] = self.n_rls(cm, k_indices,  xiXYZ, V)





class ndRLSAlssm(RLSAlssm):

    def __init__(self, nd_cost_model: list, **kwargs):
        super().__init__(nd_cost_model.cost_models[0], **kwargs)
        self._nd_cost_model = nd_cost_model

    @property
    def cost_models(self):
        """List of CostBase : Cost Models"""
        return self._nd_cost_model.cost_models


    def get_model_order(self):
        """int : Order of the (stacked) Alssm Model"""
        return lm.AlssmSum(self.cost_models[0].alssms).N


    def filter(self, Y, V=None):
        self._check_output_dimensions(Y)
        self._allocate_parameter_storage(np.shape(Y))

        for n, cost_model in enumerate(self.cost_models):
            if n == 0:
                self._ndfilter(cost_model, Y, V)
            else:
                for m in range(self._xi1.shape[0]):
                    self._ndfilter(cost_model, self._xi1[m])

    def _ndfilter(self,cost_model, Y, V=None):

        segments = cost_model.segments
        alssms = cost_model.alssms
        F = cost_model.F

        A = lm.AlssmSum(alssms).A

        if V is None:
            V = 1

        betas = np.ones(len(segments)) if self.betas is None else self.betas

        for i, (segment, beta) in enumerate(zip(segments, betas)):

            # calculate output matrix C for the segment
            tmp_c = []
            for j, alssm in enumerate(alssms):
                tmp_c.append(F[j, i] * alssm.C)
            C = np.hstack(tmp_c)

            if segment.direction == lm.FW:
                self._forward_recursion(A, C, segment, Y, V, beta)
            elif segment.direction == lm.BW:
                self._backward_recursion(A, C, segment, Y, V, beta)
            else:
                ValueError('segment.direction has wrong value.')


# rlsk1.filter(Y)
# rlsk2 = lm.RLSAlssm(costs1d, steady_state=True)
# rlsk2.filter(rlsk1.xi)

# filter signal in second dimension
alssm_k1dummy = lm.AlssmSum((alssm_k1, alssm_k1))
alssm_k2dummy = lm.AlssmProd((alssm_k1dummy, alssm_k2))  # dummy, only used to have correct dimensions
costs2d = lm.CompositeCost((alssm_k2dummy, alssm_k2dummy), (segment_left_k2, segment_right_k2),
                           F)  # only used for storage allocation
rls = ndRLSAlssm(nd_cost, calc_W=False, calc_kappa=False, steady_state=False, backend='lfilter')
rls.filter(Y)

print('start alssm 2d')
lmmulti.rls_filter2d(rls1d, rls2d, (alssm_k2, alssm_k2), Y)  # filter in second dimension
print('finish alssm filtering')

# minimize unconstrained
minimize = 'vanilla'
if minimize == 'vanilla':
    xhat_H1 = rls2d.minimize_x()
    xs_ref = xhat_H1[K1_REF, :, K2_REF]  # store state variables as reference pulse shape
print('finish minimization')

N1tilde = alssm_k1dummy.N
H_A = np.concatenate(([1], np.zeros(N1tilde ** 2 - 1))).reshape(
    (N1tilde ** 2, 1))  # constrain matrix to find pulses of same shape as the reference pulse
h = xs_ref.reshape((N1tilde ** 2, 1)).copy()
h[0] = 0
v_A = rls2d.minimize_v(H_A, h=h)
xs_A = H_A @ v_A + h
J_A = rls2d.eval_errors(xs_A)  # get SE (squared error) for hypothesis 1

xs_B = rls2d.minimize_x()
J_B = rls2d.eval_errors(xs_B)

cr = J_B / J_A

# ------------ Plotting -------------------------------
plot_ref = True
if plot_ref:
    width, height = 40, 40
    k1plot = np.arange(K1_REF - width, K1_REF + width)
    k2plot = np.arange(K2_REF - height, K2_REF + height)
    k1alssm = np.arange(K1_REF - l_side, K1_REF + l_side)
    k2alssm = np.arange(K2_REF - l_side, K2_REF + l_side)
    traj = lmmulti.trajectory_4seg((alssm_k1, alssm_k1), (alssm_k2, alssm_k2), xs_ref, k1alssm, k2alssm, K1_REF, K2_REF)
    mappedtraj = lmmulti.maptraj(traj, k1alssm, k2alssm, k1, k2, K1, K2)

    figsize = (6.5, 4.8)
    fig = plt.figure(layout='constrained', figsize=figsize, dpi=dpi)
    k1k1_, k2k2_ = np.meshgrid(k1plot, k2plot, indexing='ij')
    ax = fig.add_subplot(121)
    cset = ax.imshow(Ynans[k1k1_, k2k2_], cmap='gray_r')
    # csetlcr = ax.imshow(mappedtraj[k1k1_, k2k2_], cmap=cc.cm.fire, alpha=0.5)
    csetlcr = ax.imshow(mappedtraj[k1k1_, k2k2_], cmap=plt.cm.get_cmap('hot'), alpha=0.5)

    # 3D plot
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(k1k1_, k2k2_, Ynans[k1k1_, k2k2_], cmap='gray_r', alpha=0.6, zorder=1)

    ax.plot_surface(k1k1_, k2k2_, mappedtraj[k1k1_, k2k2_], cmap='plasma', alpha=0.7, zorder=3)
    ax.plot_wireframe(k1k1_, k2k2_, mappedtraj[k1k1_, k2k2_], colors='b', zorder=3)

    ax.set_title("3D Surface Plot of Cropped Section")
    ax.set_xlabel("Y axis")  # note the swapped axis
    ax.set_ylabel("X axis")
    ax.view_init(azim=-1, elev=46)  # Change azimuth (horizontal angle) and elevation (vertical angle)
    ax.set_zlabel("Intensity")

plot_costratio = True
if plot_costratio:
    # Convert grayscale image to RGB by stacking it three times along the last dimension
    image_rgb = np.stack([1 - Y] * 3, axis=-1)
    crdisplay = copy.copy(cr)
    crdisplayalpha = cr

    cvals = [0, 0.5, 0.8, 0.85, 1]
    colors = ["white", "yellow", "orange", "red", "red"]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    colored_overlay = cmap(crdisplay)
    colored_overlay_rgb = colored_overlay[...,
                          :3] * 1.0  # We only need the RGB channels, so discard the alpha channel for now

    alpha = crdisplayalpha[..., np.newaxis]
    alpha = alpha / np.nanmax(alpha)  # normalize to 1
    highlighted_image = (1 - alpha) * image_rgb + alpha * colored_overlay_rgb

    # Convert back to float for display
    highlighted_image = np.clip(highlighted_image, 0.0, 1.0)

    # Plot the original and highlighted images
    figsize = (7.8 * 0.7, 4.9 * 0.7)
    fig = plt.figure(layout='constrained', figsize=figsize, dpi=dpi)
    axs = fig.subplots(1, 1)
    cset = axs.imshow(highlighted_image)

    # Get the colormap colors
    alphas_fullrange = np.linspace(0, 1, num=cmap.N)  # sigmoid(np.linspace(0,1,num=cmap.N))
    alphas_fullrange = np.clip(alphas_fullrange, 0.5, 1)
    colors = cmap(np.arange(cmap.N))
    colors[:, -1] = alphas_fullrange
    cmap_with_alpha = matplotlib.colors.ListedColormap(colors)
    cmap_test = matplotlib.colors.ListedColormap(cmap(np.arange(cmap.N)))

    norm = plt.Normalize(0, 1)  # Set the normalization from -1 to 0
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_with_alpha), ax=axs, location='bottom', aspect=40)

    # plot reference area
    col_reference = 'xkcd:blue'
    k1boxmin = K2_REF - l_side
    k1boxmax = K2_REF + l_side
    k2boxmin = K1_REF - l_side
    k2boxmax = K1_REF + l_side
    axs.plot([k1boxmin, k1boxmin, k1boxmax, k1boxmax, k1boxmin], [k2boxmin, k2boxmax, k2boxmax, k2boxmin, k2boxmin],
             c=col_reference, lw=1, ls='--')
    plt.show()

