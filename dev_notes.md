# Development Notes

## TODOs

## Open Questions
- Move beta parameter used in recursions from RLSAlssm to CostSegment/CompositeCost/NDCompositeCost?
  Decision:
- Change the variable of sample weight "v" to "w". Close usage to another variable named v (rls.minimize_v())
  Decision:
- Change minimize_v/x to minimize(output='x'/'v') and filter_minimize_x/v/yhat to filter_minimize(output='x'/'v'/'yhat')
  -> less functions/documentation
  Decision:
- 