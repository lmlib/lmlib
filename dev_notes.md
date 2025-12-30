# Development Notes

## TODOs
- Change all get_model_... to get_alssm_...
- Change all sample weights from v to w if decided
- Nu implementation
- Covariance steady state for limited sum
- Complete beta implementation if decided
- Trajectory and Window Class plot method implementation
- Trajectory and Window Class extended for nd-cost

## Open Questions / Ideas to Discuss
- Move beta parameter used in recursions from RLSAlssm to CostSegment/CompositeCost/NDCompositeCost.
  ```
  cost = lm.CostSegment(alssm, segment, beta=0.8) # default beta=1
  cost = lm.CompositeCost(alssms, segments, F, betas=[1, 1, ...])
  cost = lm.NDCompsiteCost([cost_1d, cost_2d]) # betas aleady set in compsite-cost objects
  ```

  Remove argument betas in `RLSAlssm(betas=...)` Constructor

  **Decision:**
---

- Change the variable of sample weight "v" to "w". Close usage to another variable named v (rls.minimize_v())
  ```
  rls.minimize_filter_v(y, v, H, h) # v has both different meaning
  ```
  **Decision:**
---
- Change minimize_v/x to minimize(output='x'/'v') same for filter_minimize_x/v/yhat.
  -> less functions/documentation
  ```
  xs = rls.minimize(H, h, output='x') # for states
  vs = rls.minimize(H, v, output='v') # for constrain stated
  y_hat = rls.minimize(H, v, output='y_hat') # for model output --> new
  
  xs = rls.filter_minimize(y, H, h, output='x')
  vs = rls.filter_minimize(y, H, h, output='x')
  y_hat = rls.filter_minimize(y, H, h, output='y_hat')
  ```
  **Decision:**
---

- change name of filter_minimize() --> suggestion: rls.solve(y)
  ```
  rls = RLSAlssm(cost)
  xs = rls.solve(y)
  vs = rls.solve(y, H, output='v')
  y_hat = rls.solve(y, output='y_hat')
  ```
  other names rls.estimate(y) or rls.fit(y) (both imply y_hat as output)
  
  **Decision:**
---

- Trajectory and Window class method naming and default values. Suggestion:
  ```
  trajs = Trajectory.get_local(cost, xs, F=None, thd=1e-6)
  trajs = Trajectory.get_mapped(cost, xs, ks, K, merged_ks=True, merged_seg=True, F=None, thd=1e-6, fill_value=np.nan)
  ```
  Window analog.
  New: merged_ks, merged_seg arguments are set to true, such that a fast plot without indexing is at hand. 
  If user likes to deep-dive with plot change parameters.
  
  **Decision:**
---

- Naming of cost.eval_alssm_output(xs) and alssm.eval_states(xs)?
  Concerns: alssm.eval_states(xs) results in an alssm output. Why not use alssm.output(xs)?
  Then cost.alssm_output(xs) makes sense.
  eval_....(xs) implies more an assessment/judgment of the output in compare to xs

- **Decision:**
---

---

- Naming of NDCompsiteCost.costs or other name? nd_costs?
- Naming if RLSAlssm(cost?) or better cost_model?
- **Decision:**
---
