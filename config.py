import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

DTYPE_NP = np.float32
DTYPE_TORCH = torch.float32

r = 0.1          # Rate of interest
q = 0.0          # Dividend yield

T_s_ref = 0.25         # Reference auxiliary sampling horizon (thesis Sec 3.4.1: T_s)
K_ref = 10.0           # Reference strike price
kappa_ref = 5.0        # Reference mean-reversion speed
theta_ref = 0.16       # Reference long-run average variance (v_bar)
sigma_v_ref = 0.9      # Reference vol-of-vol
rho_ref = 0.1          # Reference correlation
v0_ref = 0.25          # Reference initial variance

S_BOUNDS = (0.0, 2.0)
V_BOUNDS = (0.0, 1.0)
TAU_BOUNDS = (0.0, 1.0)

# Power-law exponents for collocation point clustering
POWER_S = 3
POWER_TAU = 1.6

KAPPA_BOUNDS   = (1.0, 10.0)
THETA_BOUNDS   = (0.01, 0.5)
SIGMA_V_BOUNDS = (0.1, 1.0)
RHO_BOUNDS     = (-0.9, 0.1)
T_s_BOUNDS     = (0.1, 1.0)

FELLER_MARGIN = 0.95

PARAM_CURRICULUM_STAGES = [
    0.05,   # Stage A: +-5% of full range, centered on *_ref
    0.4,    # Stage B: 40% of full range
    1.0,    # Stage C: full range
]

# Reproducibility
random_seed = 59
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if device.type == 'cuda':
    torch.cuda.manual_seed_all(random_seed)
