# Development Notes

## TODOs
- RLSAlssm lcr method implementation
- Nu implementation
- Trajectory and Window Class plot method implementation

## Open Questions / Ideas to Discuss
- Trajectory and Window Class eval_y xs already selected or gets selected by ks?
  ```
  trajs = Trajectory.eval_y(cost, xs, ks, K, merged_ks=True, merged_seg=True, F=None, thd=1e-6, fill_value=np.nan)
  OR
  trajs = Trajectory.eval_y(cost, xs[ks], ks, K, merged_ks=True, merged_seg=True, F=None, thd=1e-6, fill_value=np.nan)

  ```
  **Decision:**  when not possible both options then only second one: Trajectory.eval_y(cost, xs[ks], ks, K, m ...)
  **Added**  Both functionalities and as well if ks is of type integer
-

- Move beta parameter used in recursions from RLSAlssm to CostSegment/CompositeCost/NDCompositeCost.
  ```
  cost = lm.CostSegment(alssm, segment, beta=0.8) # default beta=1
  cost = lm.CompositeCost(alssms, segments, F, betas=[1, 1, ...])
  cost = lm.NDCompsiteCost([cost_1d, cost_2d]) # betas aleady set in compsite-cost objects
  ```

  Remove argument betas in `RLSAlssm(betas=...)` Constructor

  **Decision:** As suggested.
  **Added** beta parameter to CostSegment and CompositeCost.
  **Removed** beta from RLSAlssm constructor.
---

- Change the variable of sample weight "v" to "w". Close usage to another variable named v (rls.minimize_v())
  ```
  rls.minimize_filter_v(y, v, H, h) # v has both different meaning
  ```
  **Decision:** sample_weights as name instead of v rls.minimize_filter_v(y, H, h, sample_weights)
  **Renamed** v to sample_weights.
-
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
  **Decision:**  Rejected
---

- change name of filter_minimize() --> suggestion: rls.solve(y)
  ```
  rls = RLSAlssm(cost)
  y_hat = rls.solve(y)


  vs = rls.solve(y, H, output='v')

  y_hat, ... = rls.solve(y, output=('y_hat', 'x', 'error'))
  ```
  other names rls.estimate(y) or rls.fit(y) (both imply y_hat as output)

  **Decision:**   y_hat, ...  = rls.fit(y, output=str, list of names ()) (fitler, minizimze, alssm_output)
  **Added function**  
---

- lcr = rls.lcr(y, H0, H1=Identity, h0=None, h1=None, output=lcr, (....))

  **Decision:**  Accepted

- Trajectory and Window class method naming and default values. Suggestion:
  ```
  trajs = Trajectory.get_local(cost, xs, F=None, thd=1e-6)
  trajs = Trajectory.get_mapped(cost, xs, ks, K, merged_ks=True, merged_seg=True, F=None, thd=1e-6, fill_value=np.nan)
  ```
  Window analog.
  New: merged_ks, merged_seg arguments are set to true, such that a fast plot without indexing is at hand.
  If user likes to deep-dive with plot change parameters.

  **Decision:**
  ```
  trajs = Trajectory.eval(cost, xs, F=None, thd=1e-6)
  trajs = Trajectory.eval_y(cost, xs, ks, K, merged_ks=True, merged_seg=True, F=None, thd=1e-6, fill_value=np.nan)
  ```
  **Renamed** to eval and eval_y.

---

- Naming of cost.eval_alssm_output(xs) and alssm.eval_states(xs)?
  Concerns: alssm.eval_states(xs) results in an alssm output. Why not use alssm.output(xs)?
  Then cost.alssm_output(xs) makes sense.
  eval_....(xs) implies more an assessment/judgment of the output in compare to xs

  **Decision:** alssm.eval_output(xs)
  **Renamed** alssm.eval_states(xs) to alssm.eval_output(xs)"

---

---

- Naming of NDCompsiteCost.costs or other name? nd_costs?
- Naming if RLSAlssm(cost?) or better cost_model?


  **Decision:** nameing cost_terms for nd_costs or cost_model
  **Renamed** to costs to cost_terms
---
