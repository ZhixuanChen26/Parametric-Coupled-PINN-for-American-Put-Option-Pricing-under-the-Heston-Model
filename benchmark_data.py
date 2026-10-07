"""
Single source of truth for published reference values at the benchmark
Heston configuration (Table 4.1: K=10, r=0.10, q=0, tau=0.25, kappa=5,
theta=0.16, sigma_v=0.90, rho=0.10, S in {8,9,10,11,12}).

Currently populated with only the Ikonen and Toivanen reference prices
(Table 4.3's "Ikonen and Toivanen" row), at the precision shown in that
table. The thesis text (Sec 4.2.1) notes the d^(a) statistics in Table 4.4
were computed from Ikonen and Toivanen's values reported to five decimal
places -- if that fifth-decimal source is available, replace these entries
with it before this module is used for anything beyond a plot marker
(where 4-decimal precision is visually indistinguishable).

The other literature methods and the three Rohan configurations are
deliberately left out for now rather than back-filled from the rounded
4-decimal display in the current Table 4.3 -- see the discussion with the
user before adding them.
"""

S_VALS = [8.0, 9.0, 10.0, 11.0, 12.0]

# Ikonen and Toivanen [12], Table 4.3, at the benchmark point (tau=0.25).
IKONEN_TOIVANEN = {
    0.0625: [2.0000, 1.1076, 0.5199, 0.2135, 0.0820],
    0.25:   [2.0785, 1.3336, 0.7959, 0.4482, 0.2427],
}
