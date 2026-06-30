# References — Gradient Quality of Differentiable NMPC

Curated, citation-grounded bibliography for the project (see `NOTE.md`). Every
entry was verified by a research agent against the primary source (arXiv / PMLR /
IEEE / proceedings) during the literature review on 2026-06-15. Verification
caveats from that pass are preserved verbatim at the bottom. One-line "→"
annotations state the *specific* contribution we rely on, not a generic summary.

Citation key convention: `[FirstAuthorYear]`.

---

## A. Differentiable optimization & MPC — the mechanism

- **[AmosKolter2017]** B. Amos, J. Z. Kolter. *OptNet: Differentiable Optimization
  as a Layer in Neural Networks.* ICML 2017, PMLR v70:136–145. arXiv:1703.00443.
  → QP layer via **implicit differentiation of the KKT conditions** (incl.
  complementarity); backward reuses the interior-point KKT factorization. Thm 1 /
  App. C.1: differentiable except on a **measure-zero** set = weakly-active
  (strict-complementarity-failing) points where the **KKT matrix is singular**.

- **[Amos2018]** B. Amos, I. D. J. Rodriguez, J. Sacks, B. Boots, J. Z. Kolter.
  *Differentiable MPC for End-to-end Planning and Control.* NeurIPS 2018.
  arXiv:1810.13400. → Forward = box-constrained iLQR (box-DDP); gradients by
  **differentiating the fixed point** (backward pass is itself an LQR solve).
  Active box bounds → corresponding control gradient **zeroed**. Explicit caveat:
  *"differentiating through the final iLQR iterate that's not a fixed point will
  usually give the wrong gradients."* MPC policy "significantly more data-efficient
  than a generic neural network" (imitation/sysID, not full RL).

- **[Agrawal2019]** A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond,
  J. Z. Kolter. *Differentiable Convex Optimization Layers (cvxpylayers).*
  NeurIPS 2019. arXiv:1910.12430. → Disciplined parametrized programming +
  affine-solver-affine; differentiates a **cone-program** reformulation; falls
  back to "a heuristic quantity" at non-differentiable points.

- **[Agrawal2019cone]** A. Agrawal, S. Barratt, S. Boyd, E. Busseti, W. M. Moursi.
  *Differentiating Through a Cone Program (diffcp).* J. Appl. Numer. Optim.
  1(2):107–115, 2019. arXiv:1904.09043. → Implicit differentiation of the
  **residual map of the homogeneous self-dual embedding**; LSQR fallback when the
  derivative map is not invertible.

- **[Barratt2018]** S. Barratt. *On the Differentiability of the Solution to Convex
  Optimization Problems.* arXiv:1804.05098, 2018. → Sufficient conditions: Slater,
  nonsingular KKT Jacobian, and explicitly the **degenerate set
  {λᵢ=0 and fᵢ=0} empty** (strict complementarity).

- **[Frey2025]** J. Frey, K. Baumgärtner, G. Frison, D. Reinhardt, J. Hoffmann,
  L. Fichtner, S. Gros, M. Diehl. *Differentiable Nonlinear Model Predictive
  Control.* arXiv:2505.01353, 2025 (acados). → IFT on **interior-point-smoothed
  optimality conditions** inside SQP; forward+adjoint sensitivities for general
  nonconvex constrained OCPs; >3× faster than mpc.pytorch/cvxpygen. Remark 2 + Fig 2:
  *"At active set changes … the solution map is nondifferentiable"* and may jump.

- **[Magoon2024]** A. Magoon, Q. Yang, N. Aigerman, S. Kovalsky. *Differentiation
  Through Black-Box Quadratic Programming Solvers (dQP).* arXiv:2410.06324, 2024.
  → "Differentiability requires … **strict complementary slackness**"; the
  derivative system "**degenerates exactly in the presence of weakly active
  constraints**"; implicit differentiation "becomes **severely ill-conditioned**"
  near active-set changes. Benchmarks duality gap (not gradient error) across 15+
  solvers.

- **[Pineda2022]** L. Pineda et al. *Theseus: A Library for Differentiable
  Nonlinear Optimization.* NeurIPS 2022. arXiv:2207.09442. → Compares
  Unroll / Truncated / Implicit / DLM; truncation is biased; Implicit is exact only
  "at an optimal solution" and "more unstable during training."

## B. Sensitivity analysis — the differentiability conditions

- **[Fiacco1983]** A. V. Fiacco. *Introduction to Sensitivity and Stability
  Analysis in Nonlinear Programming.* Academic Press, 1983. → **LICQ + strict
  complementarity + SOSC ⇒** locally unique primal-dual solution that is a
  **C¹ function** of the parameter, Jacobian by implicit differentiation of KKT.

- **[Robinson1980]** S. M. Robinson. *Strongly Regular Generalized Equations.*
  Math. Oper. Res. 5(1):43–62, 1980. DOI:10.1287/moor.5.1.43. → **LICQ + SSOSC ⇒
  strong regularity ⇒ locally unique, Lipschitz solution map** (no strict
  complementarity needed; loses C¹).

- **[BonnansShapiro2000]** J. F. Bonnans, A. Shapiro. *Perturbation Analysis of
  Optimization Problems.* Springer, 2000. → Comprehensive reference: value/solution
  map continuity & **directional differentiability under second-order conditions**.

- **[DontchevRockafellar2014]** A. L. Dontchev, R. T. Rockafellar. *Implicit
  Functions and Solution Mappings.* Springer, 2nd ed. 2014. → IFT for **generalized
  equations / variational inequalities** (KKT); single-valued Lipschitz / directional
  localizations.

- **[Pacaud2025]** F. Pacaud. *Sensitivity Analysis for Parametric Nonlinear
  Programming: A Tutorial.* arXiv:2504.15851, 2025. → Modern tutorial quoting Fiacco
  Thm 3.2.2 and Robinson Thm 4.1; when strict complementarity fails (under
  LICQ+SSOSC) the solution map is **non-Fréchet but directionally differentiable**,
  directional derivative given by an **auxiliary QP**.

## C. Parametric QP / explicit MPC — piecewise structure

- **[Bemporad2002]** A. Bemporad, M. Morari, V. Dua, E. N. Pistikopoulos. *The
  Explicit Linear Quadratic Regulator for Constrained Systems.* Automatica
  38(1):3–20, 2002. → Optimizer is **continuous piecewise-affine** over polyhedral
  **critical regions**; kinks where the active set changes.

- **[Tondel2003]** P. Tøndel, T. A. Johansen, A. Bemporad. *An Algorithm for
  Multi-parametric QP and Explicit MPC Solutions.* Automatica 39(3):489–497, 2003.
  → Crossing a region facet changes the active set by **exactly one constraint**
  (one-sided gradient); §5 treats degeneracy / LICQ violation.

- **[Haraux1977]** A. Haraux. *How to Differentiate the Projection on a Convex Set
  in Hilbert Space.* J. Math. Soc. Japan 29(4):615–631, 1977. → Metric projection is
  **directionally differentiable**, derivative = projection onto the tangent cone.

## D. Nonsmooth autodiff — conservative Jacobians

- **[BoltePauwels2021]** J. Bolte, E. Pauwels. *Conservative Set Valued Fields,
  Automatic Differentiation, SGD and Deep Learning.* Math. Program. 188(1):19–51,
  2021. arXiv:1909.10300. → Foundational **conservative-Jacobian / path-differentiability**
  theory; backprop yields a conservative field; nonsmooth SGD converges to
  Clarke-critical points.

- **[BoltePauwels2020]** J. Bolte, E. Pauwels. *A Mathematical Model for Automatic
  Differentiation in Machine Learning.* NeurIPS 2020. arXiv:2006.02080. → Canonical
  counterexample where AD returns a value **outside the Clarke subdifferential**;
  SGD avoids such points w.p. 1.

- **[Bolte2021nsift]** J. Bolte, T. Le, E. Pauwels, A. Silveti-Falls. *Nonsmooth
  Implicit Differentiation for Machine Learning and Optimization.* NeurIPS 2021.
  arXiv:2106.04350. → **Nonsmooth implicit function theorem** for definable problems;
  backprop-compatible; warns of "extremely pathological gradient dynamics" without
  the hypotheses.

- **[Bolte2022iter]** J. Bolte, E. Pauwels, S. Vaiter. *Automatic Differentiation of
  Nonsmooth Iterative Algorithms.* NeurIPS 2022. arXiv:2206.00457. → Conservative
  derivatives through fixed points of forward-backward, Douglas-Rachford, **ADMM**,
  Heavy-Ball; **a.e. convergence of computed derivatives**.

- **[Bolte2023onestep]** J. Bolte, E. Pauwels, S. Vaiter. *One-step differentiation
  of iterative algorithms.* NeurIPS 2023. arXiv:2305.13768. → One-step (Jacobian-free)
  derivative error linear in residual **+ floor ∝ ρ** (contraction); asymptotically
  exact for superlinear / Newton solvers. Validated incl. QP interior-point.

- **[KakadeLee2018]** S. M. Kakade, J. D. Lee. *Provably Correct Automatic
  Subdifferentiation for Qualified Programs.* NeurIPS 2018. arXiv:1809.08530.
  → Standard AD is **not generally correct on nonsmooth functions**; a correct
  generalized subderivative is computable at ~6× one function eval.

## E. Inexact / tolerance-dependent gradients & fixed-point differentiation

- **[Blondel2022]** M. Blondel et al. *Efficient and Modular Implicit
  Differentiation (JAXopt).* NeurIPS 2022. arXiv:2105.15183. → **Thm 1:** Jacobian
  error **linear in the solution residual**, constants ∝ inverse conditioning of the
  optimality-Jacobian. The core "tolerance → gradient bias" bound.

- **[Pedregosa2016]** F. Pedregosa. *Hyperparameter optimization with approximate
  gradient (HOAG).* ICML 2016. arXiv:1602.02355. → Hypergradient error **O(ε)** in
  inner-solve tolerance; outer convergence if {εₖ} summable.

- **[Grazzi2020]** R. Grazzi, L. Franceschi, M. Pontil, S. Salzo. *On the Iteration
  Complexity of Hypergradient Computation.* ICML 2020. arXiv:2006.16218. → Under
  contraction, approximate-hypergradient error decays **geometrically (q^t)** in
  inner/linear-solver iterations; AID ≥ ITD; AID-CG fastest.

- **[Scieur2022]** D. Scieur, Q. Bertrand, G. Gidel, F. Pedregosa. *The Curse of
  Unrolling.* NeurIPS 2022. arXiv:2209.13271. → "Better function suboptimality does
  not imply better Jacobian suboptimality"; unrolling has a **condition-number
  burn-in** where derivative error first grows.

- **[Shaban2019]** A. Shaban, C.-A. Cheng, N. Hatch, B. Boots. *Truncated
  Back-propagation for Bilevel Optimization.* AISTATS 2019. arXiv:1810.10667.
  → K-step truncated gradient biased with **exponentially decaying bias**; a
  sufficient **descent direction** under strong convexity.

- **[Bai2019deq]** S. Bai, J. Z. Kolter, V. Koltun. *Deep Equilibrium Models.*
  NeurIPS 2019. arXiv:1909.01377. → Backprop through a fixed point by implicit
  differentiation; exact only if the root-find converges.

- **[Fung2022jfb]** S. W. Fung et al. *JFB: Jacobian-Free Backpropagation for
  Implicit Networks.* AAAI 2022. arXiv:2103.12803. → Identity-Neumann approximation;
  **Cor. 3.1:** descent direction survives an **inexact / loosely-converged** fixed
  point. Direct evidence a crude Jacobian still trains.

- **[Geng2021phantom]** Z. Geng et al. *On Training Implicit Models (Phantom
  Gradients).* NeurIPS 2021. arXiv:2111.05177. → Inexact phantom gradient is an
  **ascent direction** under `error < σ²min/σmax`; matches/beats exact on ImageNet,
  faster.

- **[ArbelMairal2022]** M. Arbel, J. Mairal. *Amortized Implicit Differentiation for
  Stochastic Bilevel Optimization.* ICLR 2022. arXiv:2111.14580. → **Warm-started /
  amortized loose inner solves match exact-gradient-oracle complexity.**

## F. ADMM- / first-order-specific differentiation

- **[Stellato2020osqp]** B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd.
  *OSQP: An Operator Splitting Solver for QPs.* Math. Program. Comput.
  12(4):637–672, 2020. arXiv:1711.08013. → The canonical **ADMM** QP solver (the
  forward-solver model diffmpc2's ADMM inner loop resembles).

- **[ButlerKwon2021]** A. Butler, R. Kwon. *Efficient differentiable QP layers: an
  ADMM approach.* Comput. Optim. Appl. 84(2):449–476, 2021/23. arXiv:2112.07464.
  → Implicit differentiation of the ADMM **fixed-point residual map**; gradient
  **invariant to ADMM iteration count once converged**; ~order-of-magnitude faster
  forward than OptNet.

- **[Butler2023scqpth]** A. Butler. *SCQPTH: A Differentiable Splitting Method for
  Convex QP.* arXiv:2308.08232, 2023. → OSQP-style ADMM, implicit fixed-point
  differentiation (iteration-count invariant once converged).

- **[Sun2023altdiff]** X. Sun et al. *Alternating Differentiation for Optimization
  Layers (Alt-Diff).* ICLR 2023. arXiv:2210.01802. → The one ADMM method with an
  **explicit truncation bound: ‖∂xₖ/∂θ − ∂x⋆/∂θ‖ ≤ C₁‖xₖ − x⋆‖** (Thm 4.1) —
  per-iteration gradient converges to the KKT-derivative gradient as ADMM converges.

- **[Bambade2024qplayer]** A. Bambade, F. Schramm, A. Taylor, J. Carpentier.
  *QPLayer / Leveraging augmented-Lagrangian techniques for differentiating over
  infeasible QPs.* ICLR 2024 (HAL hal-04133055); RSS 2022 control variant.
  → Proximal augmented-Lagrangian (ProxQP); differentiates **feasible and infeasible
  / ill-posed** QPs via an extended conservative Jacobian (removes the LICQ/feasibility
  requirement).

- **[BPQP2024]** *BPQP: Differentiable Convex Optimization Backward.* NeurIPS 2024
  (Spotlight). arXiv:2411.19285. → Backward pass reformulated as a **decoupled QP**
  from the exact KKT matrix, solved with first-order (ADMM).

- **[TracyManchester2024]** R. Tracy, Z. Manchester. *On the Differentiability of the
  Primal-Dual Interior-Point Method.* arXiv:2406.11749, 2024. → The **log-barrier
  smooths the gradient** at active constraints; **decouples solve tolerance from
  gradient smoothing** ("tight solution, smooth gradients at the relaxed point").

## G. Gradient quality for RL / differentiable simulation

- **[Suh2022]** H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake. *Do Differentiable
  Simulators Give Better Policy Gradients?* ICML 2022, PMLR v162:20668–20696.
  arXiv:2202.00817. → **THE anchor.** First-order (FoBG) vs zeroth-order (ZoBG):
  FoBG biased at discontinuities (ZoBG unbiased), higher-variance under
  stiffness/chaos; **"empirical bias" cannot be diagnosed from variance alone**;
  α-order interpolated estimator.

- **[Metz2021]** L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman. *Gradients are
  Not All You Need.* arXiv:2111.05803, 2021. → Chaos-based failure: pathwise-gradient
  **variance explodes exponentially with unroll length** when the per-step Jacobian
  spectral radius > 1; black-box can beat analytic.

- **[Parmas2018]** P. Parmas, C. E. Rasmussen, J. Peters, K. Doya. *PIPPS: Flexible
  Model-Based Policy Search Robust to the Curse of Chaos.* ICML 2018. arXiv:1902.01240.
  → "Curse of chaos": reparameterization gradient direction becomes **random**;
  likelihood-ratio robust; "total propagation" inverse-variance combination.

- **[Pascanu2013]** R. Pascanu, T. Mikolov, Y. Bengio. *On the difficulty of training
  RNNs.* ICML 2013. arXiv:1211.5063. → Spectral conditions for vanishing/exploding
  gradients (product of recurrent Jacobians); the root mechanism.

- **[Xu2022shac]** J. Xu et al. *Accelerated Policy Learning with Parallel
  Differentiable Simulation (SHAC).* ICLR 2022. arXiv:2204.07137. → The canonical fix:
  **short-horizon truncation + learned terminal critic** (TD(λ)); large sample-efficiency
  / wall-clock gains over PPO/SAC on smooth tasks.

- **[Georgiev2024ahac]** I. Georgiev et al. *Adaptive Horizon Actor-Critic (AHAC).*
  ICML 2024. arXiv:2405.17784. → Adapts the horizon to **avoid stiff contact**,
  reducing first-order gradient bias (~40% more reward on locomotion).

- **[Wiedemann2023apg]** N. Wiedemann et al. *Training Efficient Controllers via
  Analytic Policy Gradient (APG).* ICRA 2023. arXiv:2209.13052. → Train a controller by
  gradient descent through differentiable dynamics; matches MPC accuracy at >10× less
  runtime compute; needs **curriculum** for stability.

- **[Suh2022bundled]** H. J. T. Suh, T. Pang, R. Tedrake. *Bundled Gradients through
  Contact via Randomized Smoothing.* IEEE RA-L 7(2):4000–4007, 2022. arXiv:2109.05143.
  → Before contact the control "produces **zero gradients**"; **flatness** gives no
  improvement direction; randomized smoothing recovers anticipatory gradients.

- **[Howell2022dojo]** T. Howell et al. *Dojo: A Differentiable Physics Engine for
  Robotics.* arXiv:2203.00806, 2022. → Contact via interior-point; **central-path κ =
  contact-softness knob**, trades gradient smoothness vs physical bias.

## H. RL paradigms for MPC-as-policy / MPC-as-function-approximator

- **[GrosZanon2020]** S. Gros, M. Zanon. *Data-Driven Economic NMPC Using
  Reinforcement Learning.* IEEE TAC 65(2):636–648, 2020. arXiv:1904.04152. → **A
  parameterized MPC can represent π⋆, V⋆, Q⋆ of the true MDP even with a wrong model**
  (adjust the cost). Q-learning + deterministic-policy-gradient updates use the **NLP/KKT
  solution sensitivities**.

- **[ZanonGros2021]** M. Zanon, S. Gros. *Safe Reinforcement Learning Using Robust
  MPC.* IEEE TAC 66(8):3638–3652, 2021. arXiv:1906.04005. → RL tunes a **robust** MPC;
  constraint satisfaction holds by construction during learning & exploration.

- **[Gros2022learning]** S. Gros, M. Zanon. *Learning for MPC with Stability and Safety
  Guarantees.* Automatica 146:110598, 2022. arXiv:2012.07369. → Online/closed-loop
  constrained Q-learning tunes MPC cost Hessian/gradient/reference/uncertainty set
  (in simulation) with stability/safety theory.

- **[Zanon2020rti]** M. Zanon, V. Kungurtsev, S. Gros. *Reinforcement Learning Based on
  Real-Time Iteration NMPC.* IFAC 2020. arXiv:2005.05225. → Extends RL-MPC theory to
  **inexact RTI** solves (the only work tackling learning on a non-converged NMPC, in
  model-free RL — not end-to-end backprop).

- **[Reiter2025survey]** R. Reiter et al. *Synthesis of MPC and RL: Survey and
  Classification.* arXiv:2502.02133, 2025. → Most recent survey; organizes MPC+RL by the
  **actor-critic** decomposition (MPC as actor vs critic). Names efficient differentiable
  sensitivities as "ongoing work in acados."

- **[Kordabad2023tutorial]** A. B. Kordabad, D. Reinhardt, A. S. Anand, S. Gros.
  *Reinforcement Learning for MPC: Fundamentals and Current Challenges.* IFAC-PapersOnLine
  56(2):5773–5780, 2023. DOI:10.1016/j.ifacol.2023.10.069 (DOI inferred; page paywalled).

- **[Romero2024acmpc]** A. Romero, Y. Song, D. Scaramuzza (w/ E. Aljalbout). *Actor-Critic
  Model Predictive Control.* ICRA 2024 (DOI 10.1109/ICRA57147.2024.10610381); T-RO 2025.
  arXiv:2306.09852. → Differentiable MPC as the **PPO actor head**; vs AC-MLP baseline:
  *slightly worse* sample-efficiency/asymptotic reward but **decisively better robustness/OOD**
  (83.3% vs 6.5% under wind). Only model-free baseline is PPO (no SAC/TD3).

- **[Hansen2022tdmpc]** N. Hansen, X. Wang, H. Su. *Temporal Difference Learning for MPC
  (TD-MPC).* ICML 2022. arXiv:2203.04955. → Sampling-based MPC (MPPI) over a learned latent
  model + TD critic; strong sample efficiency — but **not** a differentiable solver.

- **[Karnchanachari2020]** N. Karnchanachari et al. *Practical RL for MPC: Learning from
  Sparse Objectives in Under an Hour on a Real Robot.* L4DC 2020. arXiv:2003.03200.
  → RL learns the **value function used as MPC terminal cost** (MPC-as-critic-consumer),
  on real hardware in <1 h.

- **[Drgona2022dpc]** J. Drgoňa et al. *Differentiable Predictive Control.* J. Process
  Control 116:80–92, 2022. arXiv:2011.03699. → Offline: learn a neural model, then
  optimize a neural policy by backprop through the differentiable closed-loop (BPTT-style),
  constraints via penalties.

- **[Sun2025gate]** Y. Sun et al. *Learning Agile Gate Traversal via Analytical Optimal
  Policy Gradient.* arXiv:2508.21592, 2025. → NN predicts MPC cost weights + reference;
  gradients via implicit differentiation (Safe-PDP). Reports ~736k steps vs PPO ~200M
  (~271×) on quadrotor gate traversal; **deployed onboard at 100 Hz** (learning offline).
  Preprint.

## I. Model-free RL baselines

- **[Schulman2017ppo]** J. Schulman et al. *Proximal Policy Optimization.*
  arXiv:1707.06347, 2017.
- **[Haarnoja2018sac]** T. Haarnoja et al. *Soft Actor-Critic.* ICML 2018.
  arXiv:1801.01290. → Strong off-policy sample-efficient continuous-control baseline.
- **[Fujimoto2018td3]** S. Fujimoto et al. *Addressing Function Approximation Error in
  Actor-Critic Methods (TD3).* ICML 2018. arXiv:1802.09477.
- **[Janner2019mbpo]** M. Janner et al. *When to Trust Your Model (MBPO).* NeurIPS 2019.
  arXiv:1906.08253. → ~order-of-magnitude fewer samples than SAC at matched asymptotic
  performance — the sample-efficiency anchor for model-based RL.
- **[Silver2014dpg]** D. Silver et al. *Deterministic Policy Gradient Algorithms.*
  ICML 2014, PMLR v32:387–395. → DPG integrates over states only → fewer samples in
  high-dim action spaces; the basis for using DPG with a deterministic MPC policy.

## J. Real-time / deploy-time auto-tuning

- **[Diehl2005rti]** M. Diehl, H. G. Bock, J. P. Schlöder. *A Real-Time Iteration Scheme
  for Nonlinear Optimization in Optimal Feedback Control.* SIAM J. Control Optim.
  43(5):1714–1736, 2005. DOI:10.1137/S0363012902400713. → **One SQP iteration per sample**;
  the deliberately-inexact real-time substrate.

- **[Verschueren2021acados]** R. Verschueren et al. *acados — a modular open-source
  framework for fast embedded optimal control.* Math. Program. Comput. 13:147–183, 2021.
  DOI:10.1007/s12532-021-00208-8. → RTI implementation (preparation + feedback phases).

- **[Cheng2024difftune]** S. Cheng et al. *DiffTune: Auto-Tuning through
  Auto-Differentiation.* IEEE T-RO 2024. arXiv:2209.10021. → Sensitivity propagation
  (forward-mode AD) tunes a controller on **real quadrotor hardware** (~3.5× error
  reduction) — but **gradients computed offline between trials**.

- **[Tao2024difftunempc]** S. Tao et al. *DiffTune-MPC: Closed-Loop Learning for MPC.*
  IEEE RA-L 2024. arXiv:2312.11384. → MPC-policy gradient via an **auxiliary linear-MPC**
  (KKT differentiation); handles open-loop vs closed-loop and horizon mismatch.
  ∂u/∂θ = 0 when inequalities activate. **Simulation.**

- **[Zuliani2023bpmpc]** R. Zuliani, N. A. Balta, J. Lygeros. *BP-MPC: Optimizing
  Closed-Loop Performance of MPC using BackPropagation.* arXiv:2312.15521, 2023.
  → Backprop through closed-loop dynamics to tune MPC; claims convergence guarantees.
  **Simulation.**

- **[Loquercio2022autotune]** A. Loquercio, A. Saviolo, D. Scaramuzza. *AutoTune:
  Controller Tuning for High-Speed Flight.* IEEE RA-L 7(2):4432–4439, 2022.
  arXiv:2103.10698. → **Metropolis-Hastings** (not BO) tunes MPC Q/R/horizon; 50–1000
  samples; validated on a real quadrotor (offline transfer).

- **[Berkenkamp2016safeopt]** F. Berkenkamp, A. Schoellig, A. Krause. *Safe Controller
  Optimization for Quadrotors with Gaussian Processes (SafeOpt).* ICRA 2016.
  arXiv:1509.01066. → Safe BO, **online on real quadrotor** (episode-level).

- **[Frohlich2021crbo]** L. Fröhlich, M. Zeilinger, E. Klenske. *Cautious Bayesian
  Optimization.* L4DC 2021. arXiv:2011.09445.

- **[Nagabandi2019grbal]** A. Nagabandi et al. *Learning to Adapt in Dynamic, Real-World
  Environments.* ICLR 2019. arXiv:1803.11347. → **GrBAL: MAML-style online gradient model
  adaptation on real hardware**, fed to MPC (MPPI) — adapts the *model*, not MPC params.

- **[Finn2017maml]** C. Finn, P. Abbeel, S. Levine. *Model-Agnostic Meta-Learning.*
  ICML 2017. arXiv:1703.03400.

- **[OConnell2022neuralfly]** M. O'Connell et al. *Neural-Fly.* Science Robotics 7(66),
  2022. arXiv:2205.06908. → Online adaptation on a real drone — but online step is a
  composite-adaptive law over an offline-meta-learned basis.

- **[Mei2025]** Q. Mei et al. *Fast Online Adaptive Neural MPC via Meta-Learning.*
  arXiv:2504.16369, 2025. → Online MAML fine-tuning of a neural **residual model** inside
  NMPC (acados/L4CasADi), 45–50 Hz. Adapts the model; **simulation only**.

- **[Hewing2020review]** L. Hewing, K. P. Wabersich, M. Menner, M. N. Zeilinger.
  *Learning-Based MPC: Toward Safe Learning in Control.* Annu. Rev. Control Robot. Auton.
  Syst. 3:269–296, 2020. DOI:10.1146/annurev-control-090419-075625.

- **[Mesbah2022fusion]** A. Mesbah et al. *Fusion of Machine Learning and MPC under
  Uncertainty.* ACC 2022, pp. 835–854. DOI:10.23919/ACC53348.2022.9867643. → "MPC policies
  … are almost everywhere differentiable … policy gradient RL methods can be utilized for
  **offline or online auto-tuning**" — but more sample-hungry / weaker online safety than BO;
  calls BO+gradient hybridization an open opportunity.

- **[Dinev2022ddp]** T. Dinev et al. *Differentiable Optimal Control via Differential
  Dynamic Programming.* arXiv:2209.01117, 2022. → **Must keep second-order dynamics terms**;
  iLQR-level gradients make the outer optimization **diverge**.

- **[Jin2020pdp]** W. Jin, Z. Wang, Z. Yang, S. Mou. *Pontryagin Differentiable
  Programming.* NeurIPS 2020. arXiv:1912.12970.

- **[Jin2021safepdp]** W. Jin, S. Mou, G. J. Pappas. *Safe Pontryagin Differentiable
  Programming (Safe-PDP).* NeurIPS 2021. arXiv:2105.14937. → Log-barrier; **solution AND
  gradient converge to the true constrained quantities as the barrier parameter → 0**, all
  iterates feasible.

---

## Verification caveats (carried from the 2026-06-15 review)

- Bonnans–Shapiro and Dontchev–Rockafellar **book interiors were not read directly**;
  scope verified via listings/secondary sources. Confirm exact theorem numbers against the
  physical books before citing them precisely.
- Holmes (1973) title/venue verified; theorem text not read directly.
- "Safe-PDP raw gradient is wrong on the active set" is **interpretation**, not a verified
  quote (the barrier mechanism *is* verified).
- QPLayer (Bambade et al.) has **no confirmed standalone arXiv ID** (HAL hal-04133055);
  degeneracy claims not verified from primary text.
- AHAC's exact adaptive-truncation criterion is from the abstract + secondary summaries,
  not verbatim equations — treat numeric specifics as lower-confidence.
- Kordabad 2023 tutorial DOI inferred (page returned 403); body not read line-by-line.
- diffcp uses **LSQR** (not LSMR).
- `[Zanon2020rti]` internal mechanism verified only at the abstract level.
- Sun 2025 (gate traversal) and Mei 2025 are **preprints**, not peer-reviewed.
