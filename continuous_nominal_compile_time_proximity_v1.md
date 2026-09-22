# Additive Compile-Time Proximity Clarification

Date: 2026-09-22. Status: prospective before the fixed grid. The 2.3.7
legacy `margin - gap` correction alone did not make positive distances appear
when `geom_margin` was changed *after* model compilation: the broadphase
continued to omit nearby pairs. The fixed grid has not been run.

Apply `margin=0.25`, `gap=0` to robot and obstacle collision geoms in the
fixture XML **before compiling** a dedicated `NominalProximityEnv`. Keep the
original Phase 6C environment and its previous results unchanged. Verify
the resulting compiled geoms have these values and that fixed-pose distances
at offsets `0.085`, `0.090`, and `0.095 m` are finite and locally continuous.
If the check still fails, stop rather than infer safe from missing contacts.
