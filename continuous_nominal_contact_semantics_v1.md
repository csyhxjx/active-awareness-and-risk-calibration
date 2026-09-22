# Additive MuJoCo 2.3.7 Contact-Distance Correction

Date: 2026-09-22. Status: prospective correction before the nominal grid.
The previously pushed nominal amendment incorrectly states that equal
`geom_margin=geom_gap=0.25 m` makes positive-distance contacts available.
MuJoCo 2.3.7 uses the legacy `margin - gap` inclusion rule: equal values
exclude nonpenetrating pairs. The old 72-case dynamic-controller scan used
that setting and its reported absence of boundary cases is therefore a
**measurement limitation**, not proof that no boundary geometry exists.
The old raw results, summary, and interpretation remain in history unchanged.

For this new kinematic fixture only, require exactly MuJoCo `2.3.7` and set
`geom_margin=0.25 m, geom_gap=0` for every robot/obstacle collision geom.
No dynamics step is taken under the altered collision model; contact forces
cannot alter the prescribed joint poses. When no pair is reported, record a
right-censored `>=0.25 m` clearance rather than a point estimate. Before the
grid, demonstrate that at offsets `0.085, 0.090, 0.095 m` on the same route,
the nearest positive/negative distances do not disappear discontinuously.
The 8-vs-16 interpolation refinement and `0.001 m` fast-model error gate from
the main amendment are unchanged.
