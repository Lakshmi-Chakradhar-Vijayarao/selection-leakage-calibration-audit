# Correction History — Full Narrative Record

This document is the full, verbatim prose record of every correction and
robustness check behind this paper's Case Study 3 (MultiHaluDet) severity
estimate and every other headline number's correction history (Case
Studies 2 and 4, Mechanism 5, and the Severity Surface).

It exists because a reviewer-suggested compression pass moved this material
out of the compiled PDF. Previously it lived as two appendices in
`draft/latex/main.tex`:

- Appendix A, "Correction History and Robustness Checks for Case Study 3
  (MultiHaluDet)"
- Appendix E, "Correction History for Case Studies 2 and 4, Mechanism 5, and
  the Severity Surface"

Those appendices together ran to roughly 1,944 lines (~25 printed pages) of
dense narrative — over a quarter of the paper's page count — documenting
process (what was found wrong, how it was fixed, and what it moved) rather
than the paper's scientific findings themselves. The paper's main text
(§4.3, §4.4, §5) states the CURRENT numbers directly and needs none of this
to be verified. Reviewers (five independent cold reviews plus three
simulated TMLR reviews, all Accept/Weak Accept) praised this material as
evidence of rigor but agreed it does not need to live as dense prose inside
the submitted PDF.

`draft/latex/main.tex` now carries, in place of the two prose appendices, a
compact table per appendix — one row per correction/finding — with a pointer
to this document for full detail. **Nothing below is required to verify any
number the paper currently reports; everything below is preserved so that a
reader who wants the full reasoning, the intermediate numbers, and the
false starts can see them.**

The text below is reproduced near-verbatim from the paper's LaTeX source
(only page-layout artifacts — `\appendix`, section/subsection commands
retained as headers, table `tabular` environments kept as LaTeX for
fidelity — are otherwise unchanged from what was previously compiled into
the PDF). Cross-references such as "Appendix~\ref{app:corrections-global}"
refer to the OTHER section of this same document (Appendix A refers to what
is now the "Appendix E" section below, and vice versa); references such as
"§4.3" or "Table~\ref{tab:magnitude-triangle}" refer to `draft/latex/main.tex`,
which still carries the main text, tables, and figures unchanged.

---

## Appendix A. Correction History and Robustness Checks for Case Study 3 (MultiHaluDet)

```latex
\section{Correction History and Robustness Checks for Case Study 3 (MultiHaluDet)}
\label{app:corrections}

\textbf{This appendix documents how the Case Study 3 result in §4.3 was
reached, and additional robustness checks omitted from the main text for
length.} §4.3 states the result directly; nothing below is required to
verify it.

\subsection{The sixteen corrections}

\textbf{TL;DR.} Sixteen issues were found and fixed via iterative review while
deriving this case study's severity estimate (full table below). In
order: (1) an uncalibrated
synthetic sweep saturated at AUROC=1.0; (2) the calibration formula
itself was inverted; (3) LEAKY/CLEAN were not budget- or seed-matched;
(4) the real-feature test's architecture did not match the synthetic
sweep's (single-fold MLP vs. the intended 5-fold OOF+meta-learner);
(5) the ``corrected architecture'' version still ran at a saturated
operating point (AUROC$\approx0.985$); (6) the rescaling factor used to
fix (5) was itself computed from a biased, noise-inflated estimator;
(7) a permutation-null number was misattributed to the wrong check;
(8) \textbf{the calibration transform itself leaked label information across
the split it was later evaluated on} (full-sample class-mean centering
before the split) -- this is the load-bearing finding of this appendix,
and the reason this paper's own severity-measurement instrument is used
as a worked example of Mechanism 1 in §4.3; (9) fixing (8) left the
operating-point mismatch from (5) unresolved; (10) the fidelity extension
modelling MultiHaluDet's actual LR-scheduler and early-stopping behaviour
went through two sequential control-construction bugs of its own; (11)
two of this paper's synthetic generators silently realized $4\times$ the
intended Fisher $J$; (12) the fidelity extension's early-stopping
patience was set to the LR scheduler's $3$ rather than the audited
repo's own $15$; and (13) \textbf{the fix for (8) was itself still
label-conditional} --- it fixed which samples the class means were
\emph{estimated} from but not which label decided how each point was
\emph{shifted} --- replaced with a fully label-free calibration that
reaches chance exactly at zero separation, and which, for the first time
in this test's history, reaches its intended $0.80$ operating point
instead of saturating; and (14) \textbf{the ``fidelity extension'' was not
faithful to the audited trainer's optimizer or data pipeline} --- it ran at a
$10\times$ learning rate, full-batch, with plain Adam, the wrong scaler and no
warmup --- which is now ported and, for what remains unported, disclosed
exhaustively; (15) \textbf{that same fidelity extension's control was not
budget-matched to \texttt{code/43}'s convention, and the arm needed to isolate
its headline ``coupling continuity'' claim did not exist} --- fixing the first
moves the reported gap from $+0.0338$ to $+0.0250$, and adding the second
retracts the coupling-continuity attribution outright; and (16) \textbf{the
adaptivity control carved its ``held-out'' selection fold out of the very
training set it then trained on}, so it measured convergence rather than
selection --- rebuilt disjointly, it retains $94.2\%$ of honest selection's
placebo-relative benefit instead of $37.3\%$, \emph{but a later round showed the
rebuilt arm to be bitwise identical to the budget-deficit arm, so the conclusion
it was then read as supporting is withdrawn}
(Appendix~\ref{app:corrections-global}, E27).

\textbf{The numbers that survive all sixteen and are the ones §4.3
actually reports} are stated in §4.3 and in items 9, 12, 13, 14, 15 and 16 of
the table below. Two structural points about them: the two synthetic sweeps
(isotropic, anisotropic) share neither the architecture-mismatch (4) nor the
split-leakage (8) problem the real-feature test needed fixing for, which is why
the isotropic sweep was this paper's primary severity estimate for this
mechanism until the selection-budget factorial of §4.3 removed its standing as a
detectable effect --- after which the corrected real-feature harness and the
corrected fidelity extension carry it instead, at their fold-matched values and
with the capacity dependence §4.3 reports (the un-averaged readout was offered
here too, and is withdrawn: E26); and the fidelity extension's move to $+0.0250$ is \emph{not}
attributed to any one of (12), (13) or (14) alone --- a factorial
$2\times2$ ablation (item 12), rerun under (14)'s fidelity port, shows the
calibration fix rather than the patience fix is what moves it, and retracts
an interaction this appendix previously claimed.

\textbf{Correction table.} Each row is one issue found by iterative
review while deriving §4.3's severity estimate; full derivations live in
the cited script/result files, not below.

\begin{description}
\item[1.] Uncalibrated sweep saturated at AUROC$=1.0$. \textit{Fix:}
  calibrate via $\text{AUROC}=\Phi(\sqrt{J/2})$.
\item[2.] Calibration formula was inverted ($\Phi(\sqrt J/2)$, the
  accuracy identity, not AUROC). \textit{Fix:} corrected inversion,
  $J=2\Phi^{-1}(0.80)^2=1.417$ $\rightarrow$ verified
  $\Phi(\sqrt{J/2})=0.800$.
\item[3.] LEAKY/CLEAN not budget- or seed-matched (CLEAN's ES carve-out
  $\approx$15\% less data; matched retrain used a different seed).
  \textit{Fix:} match both $\rightarrow$ result now stated in §4.3.
\item[4.] Real-feature test's architecture didn't match the synthetic
  sweep's (single-fold MLP vs. intended 5-fold OOF+meta-learner).
  \textit{Fix:} rerun with the actual architecture (\texttt{code/25})
  $\rightarrow$ $+0.0002$ ($p=0.56$, cap. 128), not significant.
\item[5.] ``Corrected architecture'' still ran at a ceiling operating
  point (AUROC$\approx0.985$), making its MDE incomparable to the
  synthetic sweep's. \textit{Fix:} rescale real features to the same
  $0.80$ target (\texttt{code/31}) $\rightarrow$ $+0.0014$/$+0.0010$
  (cap. 128/384), inside the synthetic range but individually
  underpowered.
\item[6.] Rescaling factor $\alpha$ itself came from a biased estimator
  ($J_{\text{real}}$ unstable at $d\approx n$, i.e.\ $414$ against $400$;
  permutation test gives $J_{\text{perm}}=2.31 > J_{\text{target}}=1.42$, so
  pure label noise registers as more separated than the calibration target).
  \textit{Fix:} empirical, CV-calibrated bisection on \emph{achieved} AUROC
  --- bisection over the rescaling factor using 5-fold cross-validated,
  regularized logistic regression to measure rather than analytically predict
  the achieved AUROC at each candidate value (\texttt{code/33})
  $\rightarrow$ converges to $\alpha=0.2031$, hitting AUROC $0.795\pm0.013$
  on a 10-seed re-check by the measure used to calibrate it.
\item[7.] A permutation-null number ($0.512\pm0.044$) was misattributed
  to this calibration check. \textit{Fix:} reconciled -- $0.512\pm0.044$
  is \texttt{calibration\_leakage\_diagnostic.json}'s
  \texttt{check\_A\_plain\_features\_vs\_permuted\_labels}, a different,
  unrelated check; this check's real value is
  \texttt{check\_B\_full\_sample\_calibration\_from\_permuted\_labels},
  $0.120\pm0.010$.
\item[8.] \textbf{The calibration transform itself leaked label
  information across the split it was evaluated on} -- full-sample
  class-mean centering before the split forces
  $\cos(\Delta\mu_{\text{tr}},\Delta\mu_{\text{te}})\approx-1$ by
  construction (verified: $-0.9999\pm0.0000$, train$\to$test
  AUROC$=0.062$). \textit{Fix:} re-center on train indices only
  (\texttt{code/43}, \texttt{apply\_calibration\_train\_only}); re-run
  alpha search per-fold $\rightarrow$ $\alpha=0.1328$;
  $\cos=0.0000\pm0.0007$, AUROC$=0.584\pm0.059$ (no longer inverted).
\item[9.] Operating-point mismatch (issue 5) survives the leakage fix:
  the train-only-calibrated pipeline still ran at
  AUROC$\approx0.96$-$0.97$, not $0.80$ (issue 13's label-free
  recalibration later brought this down to $\approx0.94$-$0.95$, still
  not $0.80$). \textit{Fix:} report at the test's own operating point
  rather than retire it. Under the superseded label-conditional
  calibration this gave LEAKY beating CLEAN\_MATCHED by $+0.0021$
  (cap.\ 128, $p=0.0006$) / $+0.0022$ (cap.\ 384, $p=0.0002$); under the
  label-free calibration of issue 13 it gives $+0.0093$ (cap.\ 128, BCa
  95\% CI $[+0.0060,+0.0134]$, Wilcoxon $p=7.3\times10^{-7}$,
  permutation $p<0.0001$) and $+0.0077$ (cap.\ 384, CI
  $[+0.0050,+0.0109]$, Wilcoxon $p=1.4\times10^{-5}$, permutation
  $p<0.0001$) -- \textbf{the numbers §4.3 reports}. Tie counts on those
  Wilcoxon tests also improved, from $49$/$60$ of $100$ pairs under the
  superseded calibration to $33$/$40$, and the Wilcoxon and permutation
  verdicts now agree where previously they differed by an order of
  magnitude in $p$ (§4.3).
\item[10.] Checkpoint-selection-only harness may understate
  MultiHaluDet's actual leak: the real trainer couples an LR scheduler
  and early stopping to the same validation fold every epoch, not just a
  final checkpoint pick. \textit{Fix attempt 1:} port the
  scheduler+shared-patience mechanic into the harness (\texttt{code/49});
  an early version selected the control's retrain epoch count from the
  leaky run's own training loop, which can never differ from it by
  construction -- caught before reporting. \textit{Fix attempt 2:} select
  the epoch count from a genuinely disjoint carve-out instead, but then
  discard the model that carve-out run itself produced and retrain from
  scratch with plain constant-LR Adam and no scheduler at all --
  reintroducing a new confound (control has no adaptive
  LR/early-stopping mechanism at all, while LEAKY and PLACEBO both do),
  caught by a later independent review before reporting. \textit{Fix
  (final):} keep the model already trained with the real scheduler on
  the disjoint carve-out as the control directly, rather than discarding
  it $\rightarrow$ LEAKY\_PLUS\_LRSCHED beats
  CLEAN\_MATCHED\_PLUS\_LRSCHED by $+0.0111$ ($p=4.3\times10^{-9}$) on the
  real-feature harness at capacity 128, as previously reported. A budget confound
  disclosed here for completeness: \texttt{CLEAN\_MATCHED\_PLUS\_LRSCHED}
  trains on the disjoint carve-out (\texttt{tr2\_idx}, $\approx85\%$ of
  \texttt{tr\_idx}) rather than the full training-fold budget LEAKY and
  PLACEBO use -- the same $15\%$ carve-out confound issue 3 fixed for the
  main checkpoint-selection harness, not re-fixed here for this
  extension. \textbf{A second confound, which this extension had stopped
  tracking:} the arms also differ in training depth --- LEAKY's kept
  checkpoint sits at mean epoch $29.14$ against the control's $19.75$ and
  the placebo's $7.95$ ($+47.6\%$, Wilcoxon $p=7.2\times10^{-13}$;
  \texttt{best\_epoch\_stats} in
  \texttt{results/mechanism3\_fidelity\_extension.json}). §4.3 states why we
  report rather than equalize it. (CLEAN\_MATCHED\_PLUS\_LRSCHED itself
  still beats PLACEBO\_PLUS\_LRSCHED decisively under the ported training
  loop, $+0.0211$, $p=3.6\times10^{-7}$, confirming the honest
  scheduler-bearing control is strongly informative, not
  degenerate. \emph{Correction:} this appendix previously quoted $+0.0498$
  ($p=3.7\times10^{-16}$) here, which was the pre-issue-14 value and did not
  survive the fidelity port --- a stale number found by a further review of
  the shipped JSON against the text, and exactly the failure mode
  \texttt{code/53} exists to catch, which is why that check now covers this
  pair too).
\item[11.] Two of this paper's own synthetic-data generators
  (\texttt{code/46}, Mechanism 5; \texttt{code/47}, the K/selection-set-size
  sweep) independently committed the same calibration-inversion error class
  as issue 2 above, in a new form: the per-class mean offset used the
  \emph{full} separation on each side ($\pm\text{class\_sep}$) rather than
  half ($\pm\text{class\_sep}/2$), silently realizing $4\times$ the
  intended Fisher $J$ and mislabeling every operating point in both
  scripts' output (e.g.\ a labeled $0.985$-AUROC cell actually realized
  Bayes-optimal AUROC $\approx0.99994$). \textit{Fix:} corrected both
  generators to the $\pm\text{class\_sep}/2$ convention already used
  elsewhere in this paper (\texttt{code/02d}), and added a new,
  non-tautological guardrail (\texttt{code/sanity\_checks.py}).

  \emph{Three limits on that guardrail, disclosed rather than left implicit.}
  \emph{(a) Coverage.} Despite its docstring's stated intent it is currently
  called only by the two generators that needed it (\texttt{code/46},
  \texttt{code/47}); the pre-existing generators (\texttt{code/02d},
  \texttt{22}, \texttt{27}, and others) were not retrofitted, so ``every
  synthetic generator in this project must call it'' is aspirational, not yet
  true. \emph{(b) Applicability.} For \texttt{code/27} the guardrail is not
  merely un-retrofitted but not \emph{applicable}: its bias correction
  $\hat J=\lVert\bar x_+-\bar x_-\rVert^2-d(1/n_++1/n_-)$ is exact only under
  identity within-class covariance $\Sigma=I$, since the subtracted term
  equals $d(1/n_++1/n_-)$ only when every dimension has unit variance and
  dimensions are uncorrelated. Under a general $\Sigma$ the correct term is
  $\operatorname{tr}(\Sigma)(1/n_++1/n_-)$ and the target must be the
  Mahalanobis $\Delta\mu^\top\Sigma^{-1}\Delta\mu$, so calling it on
  \texttt{code/27}'s real, correlated covariance could both reject correct data
  and pass buggy data; making it applicable requires modifying the estimator,
  not adding a call site, and the assumption is now stated in
  \texttt{sanity\_checks.py}'s docstring. The generators it does gate are
  isotropic by construction, so the assumption holds where it is used; the
  anisotropic generators (\texttt{code/27}, \texttt{code/56}) verify
  calibration by a different route, and \texttt{code/56} can do so exactly
  because it constructs $\Sigma$ rather than estimating it.
  \emph{(c) Sensitivity.} The guardrail's own design needed two iterations:
  fitting a regularized classifier and checking its CV AUROC against the target
  systematically undershoots by $\sim0.04$-$0.05$ AUROC (a real,
  separately-documented regularization effect, not a bug -- see issue 5), so it
  instead checks the bias-corrected realized Fisher $J$ ratio directly, which
  separates correct generators (ratio $0.33$-$1.83$ across 2000 seeds
  and five operating points) from the $4\times$ bug (ratio never below
  $2.77$). At small $n_{\text{samples}}$ and low target AUROC (verified at
  $n_{\text{samples}}=350$, target $0.70$) the two distributions begin to
  overlap (correct-generator ratios reach $2.32$, buggy ones fall as low as
  $2.03$), so the bounds ($[0.2,2.6]$) were widened to prioritize never
  rejecting correct data, at the cost of sensitivity in exactly that regime ---
  which is the regime of \texttt{code/47}'s Sweep B cell at
  $n_{\text{samples}}=350$. A reader should treat that cell as somewhat less
  protected against this bug class than the others.
\item[12.] \textbf{The fidelity extension used the wrong early-stopping
  patience.} A later independent review found
  \texttt{code/49} used a single constant, $3$, for \emph{both} the LR
  scheduler's reduction threshold and the early-stopping break. The
  audited repo does not: \texttt{trainer.py:76} constructs
  \texttt{ReduceLROnPlateau} with its own \texttt{patience=3}, while the
  early-stopping break at \texttt{trainer.py:149} tests against
  \texttt{config.patience = 15} (\texttt{config.py:19}). Breaking after 3
  non-improving epochs truncates every run drastically. \textit{Fix:}
  separate counters, $3$ for the scheduler and $15$ for early stopping.

  This landed at the same time as issue 13's calibration fix, so both
  were varied factorially rather than reported jointly
  ($n=100$ seeds per cell, \texttt{code/54},
  \texttt{results/fidelity\_extension\_2x2\_ablation.json}). The table below
  is the ablation \emph{rerun under issue 14's training-loop fidelity port},
  so all four cells are on the same footing as the number §4.3 reports:

  \begin{center}\small
  \begin{tabular}{lccc}
  \toprule
  Calibration (LEAKY operating pt.) & ES patience 3 & ES patience 15 \\
  \midrule
  Superseded label-conditional ($0.979$/$0.981$) & $+0.0081$ & $+0.0052$ \\
  Label-free axis-noising ($0.920$/$0.942$) & $+0.0342$ & $\mathbf{+0.0338}$ \\
  \bottomrule
  \end{tabular}
  \end{center}

  \textbf{A previously reported interaction does not survive the fidelity
  port, and we retract it rather than restate it.} The version of this
  ablation run before issue 14 showed the patience correction \emph{shrinking}
  the gap at the superseded calibration ($-0.0046$) and \emph{growing} it at
  the label-free one ($+0.0069$), and this appendix presented that sign flip
  as the operating-point relationship of §5 appearing unbidden inside a
  different harness. Under the ported training loop the sign no longer flips:
  the patience correction shrinks the gap at both calibrations, by $-0.0029$
  at the superseded one and by a negligible $-0.0004$ at the label-free one.
  The earlier sign flip was therefore an artifact of the full-batch,
  $10\times$-too-high-learning-rate training loop, in which a 3-epoch versus
  15-epoch stopping rule meant something very different than it does under
  the repo's actual mini-batch schedule. §5's operating-point relationship
  stands on its own purpose-built sweep; it does not need this cell, and we no
  longer cite this ablation as independent evidence for it.

  What the table does still establish is the decomposition it was built for:
  the calibration fix, not the patience fix, is what moves this number ---
  $+0.0081\to+0.0342$ at patience 3 and $+0.0052\to+0.0338$ at patience 15,
  against a patience effect of at most $-0.0029$ either way. The
  bottom-right cell --- label-free calibration, repo-faithful patience 15,
  $+0.0338$ (BCa 95\% CI $[+0.0279,+0.0405]$, Wilcoxon
  $p=6.2\times10^{-15}$, paired permutation $p<0.0001$, 28 tied absolute
  differences of 100 pairs, no exact zeros) --- is the cell this $2\times2$
  is anchored on; the other three exist to make the decomposition checkable.
  Note that all four cells here score LEAKY against the es-fed control, which
  item 15 supersedes: the number §4.3 reports is $+0.0250$, against the
  budget-matched control. The $2\times2$ is retained on the old control
  because its purpose is to separate the calibration and patience corrections
  from each other, and re-running it against a control that did not exist when
  either correction was made would not make that separation cleaner.

  \textbf{Like-for-like against the checkpoint-selection-only harness.}
  Both harnesses now run on the same real features under the same
  label-free calibration ($\alpha=0.3359$), so at capacity 128 the fidelity
  extension's $+0.0250$ against the checkpoint-selection-only harness's
  $+0.0093$ (issue 9) is a ratio of $\mathbf{2.7\times}$. That replaces the
  $3.6\times$ reported before the issue-15 budget-matching fix, the
  $2.4\times$ reported before the issue-14 fidelity port, the
  $5.2\times$ before that (which compared $+0.0111$ against
  $+0.0021$ under the superseded calibration), and the ``3-4$\times$'' of
  the draft before that, which compared across operating points
  entirely. The residual non-equivalence a previous draft disclosed here
  --- that the two harnesses landed at different operating points (LEAKY
  mean $0.9176$ versus $0.9403$), which, given §5, plausibly inflated the
  ratio --- has largely closed under the fidelity port: the extension's
  LEAKY mean is now $0.9424$ against the checkpoint-only harness's
  $0.9403$, a difference of $0.002$ AUROC. We no longer report the ratio
  as an upper bound on that account. Two non-equivalences do remain: the two
  harnesses still differ in training loop (the checkpoint-only harness is
  full-batch constant-LR Adam, unchanged, since it deliberately models only
  the final-checkpoint argmax), and \texttt{CLEAN\_MATCHED\_PLUS\_LRSCHED}
  still trains on the $\approx85\%$ \texttt{tr2\_idx} carve-out rather than
  the full fold budget (issue 10).
\item[13.] \textbf{The train-only calibration fix from issue 8 was itself
  still label-conditional.} Estimating class means from train indices
  fixes estimation, not application: \texttt{mask = y == cls} ranges over
  the whole array, so each test point's own label still decided which
  class offset was subtracted from it -- a residual instance of
  Mechanism 1, in this paper's own instrument, for the second time in the
  same script. Its signature: at $\alpha=0$, where the classes should be
  exactly indistinguishable, that transform still leaves train-fit
  $\rightarrow$ test-eval AUROC $=0.584\pm0.059$. \textit{Fix:}
  \texttt{apply\_calibration\_label\_free} (\texttt{code/43}) --- axis $u$,
  midpoint $c$ and pooled within-class SD $s_w$ estimated from train
  only, then the identical label-free map
  $p\mapsto\alpha p+\sqrt{1-\alpha^2}s_w\varepsilon$ applied to every
  row. At $\alpha=0$ this gives $0.487\pm0.092$, chance. \textit{Rejected
  alternative, documented because the negative result matters:} pure
  directional shrinkage with no noise re-injection consults no label
  either, but is an invertible linear map for any $\alpha>0$ and so
  cannot reduce separability at all (measured: AUROC $0.993$ at
  $\alpha=0.2031$ against the label-free transform's $0.671$). Re-running
  the alpha search under the label-free transform converges to
  $\alpha=0.3359$ and, for the first time in this test's history, hits
  its intended operating point (CV AUROC $0.7995$ against a $0.80$
  target) rather than saturating far above it.
\item[14.] \textbf{The ``fidelity extension'' was not faithful to the audited
  trainer's optimizer or data pipeline, and the paper did not say so.} A
  further independent review read \texttt{code/49} against the pinned commit
  line by line and found that, while the validation-signal coupling (the
  scheduler, the early-stopping counter, the checkpoint argmax) was ported
  correctly, essentially everything around it was not. \texttt{code/49} used
  Adam at $\text{lr}=2\times10^{-3}$ where \texttt{config.py:17} specifies
  $2\times10^{-4}$ with AdamW --- a $10\times$ mismatch; it was full-batch,
  taking exactly one gradient step per epoch, where \texttt{config.py:15}
  specifies \texttt{batch\_size}$=28$; it had no \texttt{warmup\_epochs}, so
  it also missed that the audited trainer does not step the scheduler during
  warmup (\texttt{trainer.py:134}); it used \texttt{StandardScaler} on the
  input features where \texttt{run\_pipeline.py:81,85} uses
  \texttt{RobustScaler}, and no scaler at all on the deep OOF features where
  \texttt{run\_pipeline.py:130} uses \texttt{StandardScaler}; and it used
  \texttt{min\_lr}$=10^{-5}$ against the repo's $10^{-7}$, no gradient
  clipping, and 60 epochs against the repo's 45. Calling the result a
  ``fidelity extension'' while differing on all of that was an overclaim.
  \textit{Fix:} all of the above are ported, and the module docstring now
  carries an exhaustive ported/not-ported list rather than an implicit claim.
  §4.3 names what remains unported --- the objective, the weight EMA, the
  augmentation, and the model class --- so the honest scope is fidelity of the
  \emph{validation-signal coupling and optimizer schedule}, not of the
  objective or the architecture.

  \textit{What it moved.} The reported gap rose from $+0.0221$ to
  $+0.0338$ (subsequently corrected to $\mathbf{+0.0250}$ by item 15's
  budget-matching fix), the like-for-like ratio against the
  checkpoint-selection-only harness from $2.4\times$ to $3.6\times$ (and then
  to $2.7\times$), and ---
  helpfully --- the extension's LEAKY operating point from $0.9176$ to
  $0.9424$ (see item 12). It also \emph{costs} this appendix a claim: the
  $2\times2$ ablation's sign flip (item 12) does not survive the port and is
  retracted there. A separate,
  smaller bug found in the same pass: \texttt{code/49} hardcoded its output
  JSON's \texttt{calibration\_method} field to
  \texttt{label\_free\_axis\_noising} even in the ablation cells that ran the
  superseded transform, mislabelling two of the four result files; the field
  now reports whichever transform actually ran.

\item[15.] \textbf{The fidelity extension's control was not budget-matched,
  and the arm that would isolate its headline claim was missing.} Two related
  defects, both found by comparing arm means across harnesses rather than
  reading gaps. \emph{(a)} \texttt{code/43}'s CLEAN\_MATCHED retrains on the
  full \texttt{tr\_idx}; \texttt{code/49}'s control trained on
  \texttt{tr2\_idx}, $\approx$$85\%$ of it. Between the two harnesses LEAKY
  moves only $+0.0022$ while the control moves $-0.0224$ and the placebo
  $-0.0284$ --- so most of the extension's larger gap was the controls falling,
  not the leak rising, and reporting the resulting $3.6\times$ as isolating
  ``scheduler coupling continuity'' was unsupported. \textit{Fix:} a
  budget-matched control that retrains on the full \texttt{tr\_idx} while
  replaying the disjoint-carve-out run's own per-epoch learning-rate
  trajectory, so it keeps an adaptive schedule instead of reverting to the
  constant-LR retrain that would re-open item 10's confound. \emph{(b)} The
  claim being made --- that \emph{continuous} coupling is worse than a one-shot
  argmax --- was tested by comparing two harnesses that differ in four other
  respects. \textit{Fix:} an arm with this script's repo-faithful optimizer and
  data pipeline, the same reused val fold, and \emph{only} a final-checkpoint
  argmax.
  \textit{What it moved.} The reported gap falls $+0.0338\to\mathbf{+0.0250}$
  (BCa 95\% CI $[+0.0192,+0.0314]$, Wilcoxon $p=1.5\times10^{-11}$), the
  like-for-like ratio $3.6\times\to2.7\times$, and the
  $(\text{LEAKY}-\text{CM})/(\text{CM}-\text{PLACEBO})$ sanity ratio
  $1.60\to0.83$, back inside the $0.06$--$0.85$ band this paper's other eight
  harness/capacity cells occupy. \textit{What it costs.} The
  coupling-continuity claim, retracted: the isolating arm scores $0.9583$
  against the coupled arm's $0.9424$, i.e.\ adding continuous coupling
  \emph{lowers} the leaky arm by $-0.0159$ (Wilcoxon $p=2.8\times10^{-13}$),
  because early stopping at the repo's \texttt{patience}$=15$ truncates it to a
  mean epoch of $29.14$ where the argmax-only arm reaches $40.73$.

\item[16.] \textbf{The adaptivity control (\texttt{code/22}) selected its
  checkpoint on data inside its own training set.} \texttt{es\_idx} was carved
  out of \texttt{tr\_idx}, but the arm then trained on the full
  \texttt{tr\_idx} --- so the ``held-out'' selection fold was in-sample
  throughout, and the argmax simply ran to the end of the budget. The
  kept-checkpoint epoch, now recorded, shows it directly: $39.4$ of $45$ at
  capacity 128 against $20.8$--$21.4$ for the honest arms. The control
  measured convergence, not adaptive selection, and could not test the
  alternative explanation it existed to test. \textit{Fix:} train on
  \texttt{tr2\_idx}, the complement of \texttt{es\_idx}, mirroring
  \texttt{code/43}. \textit{What it moved.} The control's placebo-relative
  retention rises $37.3\%\to\mathbf{94.2\%}$ (cap.\ 128) and
  $25.7\%\to\mathbf{79.9\%}$ (cap.\ 384), so it holds the adaptive
  mechanism nearly fixed; LEAKY still beats it by $+0.0049$ ($p=0.0008$) and
  $+0.0059$ ($p=0.0012$). \textit{What it changes in the paper --- and what a
  later round took back.} §4.3
  previously called the ``any adaptive selection helps'' alternative
  ``directionally unsupported'' on the strength of an invalid control; with the
  rebuilt one it was then read as positively rejected. \textbf{That reading is
  withdrawn.} A subsequent review proved the rebuilt arm bitwise identical to
  \texttt{code/67}'s budget-deficit arm --- same weights, every (seed, fold)
  pair verified ($15/15$ at capacity 128) --- so the increments above are an arithmetic restatement of a
  training-budget deficit and carry no evidence about adaptivity. The
  alternative is untested, not rejected
  (Appendix~\ref{app:corrections-global}, E27). The broken arm is still run and
  shipped as \texttt{clean\_matched\_adaptive\_contaminated}.
\end{description}

\textbf{Issue 8 is the load-bearing finding of this appendix}, and the
reason this paper's own severity-measurement instrument is used as a
worked example of Mechanism 1 in §4.3: a full-dataset, label-dependent
transform applied before the split it is later evaluated on -- the exact
pattern this paper's checklist tells readers to look for in other
people's code, found instead in ours.

\textbf{What this correction history demonstrates about itself.} A single
confirmatory- or null-looking number was not enough at any step above -- a
correctly-inverted formula, a budget/seed-matched control, adequate power, an
unbiased calibration estimator, and a train-only split all had to hold
\emph{simultaneously}, and each in turn was checked, found wanting, and fixed
by a fresh round of review. The pre-registered decision rule
(\texttt{GENUINE\_LEAK\_CONFIRMED} /
\texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK} / \texttt{MIXED}, from
\texttt{code/02d}) returns \texttt{MIXED} on most tested versions of this
experiment, \texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK} on the one
ceiling-confounded version, and \texttt{GENUINE\_LEAK\_CONFIRMED} on the
current, label-free-calibrated real-feature test -- an earlier claim that the
confound branch was structurally unreachable was itself wrong, and is
corrected here. All three branches are reachable, none is stable across
configurations, and the verdict moved toward our own hypothesis when the
calibration was corrected -- which is exactly when a single pre-registered
label deserves least weight. This is why §4.3's claims rest on effect sizes
with intervals, not on the rule's output.

\subsection{The primary sweep: control design and the randomization fix}

\textbf{A second, smaller asymmetry inside CLEAN\_MATCHED, disclosed at a review's request.} CLEAN\_MATCHED's epoch count is \emph{selected} by a model
trained on only $85\%$ of the fold's training data --- the $15\%$ carve-out it
selects against is taken out of that data (\texttt{tr2\_idx} versus
\texttt{es\_idx} in \texttt{code/47}) --- but the retrain that produces the
reported model then uses $100\%$ of it (\texttt{tr\_idx}), while LEAKY's epoch
count is selected by a model trained on the full $100\%$. So the two arms are
matched on the retrain's \emph{data} budget, which is what this control was
built for, but not on the data budget of the run the epoch count was chosen
from. Its direction is not determined a priori: a model trained on less data
typically peaks later, which would push CLEAN\_MATCHED's chosen epoch up, but
it is also a noisier selection signal. We disclose rather than correct it,
because removing it would require selecting on a carve-out drawn from outside
the fold, which changes what the control is a control for.

\textbf{This table changed under a randomization fix, and the change is
instructive rather than cosmetic.} An independent review found that
\texttt{code/02d} --- the harness behind this table --- passed \emph{the
replicate index $i$ itself} to every randomization point in the replicate: the
generator seed for the synthetic data draw, the \texttt{random\_state} of the
outer \texttt{train\_test\_split}, and the \texttt{random\_state} of the inner
\texttt{StratifiedKFold}. The defect was one integer aliased across three
independent randomization points --- \texttt{seeds\_for(i)} returned
\texttt{(i, i, i, i)} --- not three seed streams that had become correlated;
there was never more than one stream, so across the $100$ replicates those
sources of variation were perfectly confounded. This is exactly the confound
\texttt{code/47}'s explicitly decoupled
\texttt{data\_seed}/\texttt{split\_seed}/\texttt{fold\_seed\_base}/
\texttt{init\_seed\_base} scheme was written to fix, and it had never been
retrofitted to \texttt{code/02d}. It is now (\texttt{seeds\_for()}; the
previous behaviour is reproducible bit-for-bit via \texttt{--coupled-seed} and
its output ships as
\texttt{results/corrected\_capacity\_placebo\_sweep\_coupled\_seed\_legacy.json},
so the comparison below is checkable rather than asserted). §4.3's table is the rerun.

\textbf{The magnitudes are stable; which capacities clear significance is
not.} The band moves from $+0.0009$--$+0.0034$ to $+0.0011$--$+0.0036$ --- a
shift of a few ten-thousandths of an AUROC point, well inside these cells'
seed-draw noise --- but capacity $48$ goes from $p=0.015$ to $p=0.171$ and
capacity $16$ goes the other way, from $p=0.309$ to $p=0.019$. We state this
plainly because it is weaker than the previous, seed-confounded version of this
table, under which two capacities survived Holm-Bonferroni: the randomization
fix cost this result one surviving cell. That the significance pattern is this
movable across a change which leaves the effect sizes essentially fixed is
itself the point, and the reason this paper rests on effect sizes with
intervals rather than on per-cell significance labels. As an internal check
that this is the fix rather than a new bug, the decoupled capacity-128 cell
($+0.0036$, $p=0.0012$) now reproduces \texttt{code/47}'s default cell
($+0.0036$, $p=0.0012$), the identical configuration, which had used decoupled
seeds all along and differed from \texttt{code/02d} under the coupled scheme.
\emph{We previously called that cell ``independently-written'' and withdraw
it:} \texttt{code/47}'s own docstring describes its generative process and
LEAKY/CLEAN/CLEAN\_MATCHED/PLACEBO logic as ``an exact port of
\texttt{code/02d},'' so the two agreeing is a \emph{determinism check}, not
independent confirmation by a second implementation. No such independent
implementation exists in this paper. \texttt{code/55}'s readouts were
retrofitted with the identical seed streams and rerun; the other synthetic
generators (\texttt{code/22}, \texttt{code/27}) still use the coupled scheme
and were not, being robustness checks reported against their own baselines
rather than contributors to this table.


\medskip\noindent\textbf{Intervals, multiplicity and power for the primary table.} BCa bootstrap 95\% CIs on that gap (\texttt{code/44}, $10{,}000$ resamples,
paired permutation $p$ alongside Wilcoxon), at capacities 16/48/128/384:
$[+0.0007,+0.0036]$ ($p=0.009$); $[-0.0004,+0.0028]$ ($p=0.175$);
$[+0.0019,+0.0057]$ ($p=0.0003$); $[-0.0002,+0.0052]$ ($p=0.091$) ---
concordant with the Wilcoxon verdicts above.

CLEAN\_MATCHED beats PLACEBO decisively at every capacity
($p<10^{-6}$ throughout, gaps $+0.012$ to $+0.025$), confirming honest
checkpoint selection is itself strongly informative. The
LEAKY$-$CLEAN\_MATCHED residual -- the actual leakage estimate -- is
individually significant at 2 of the 4 capacities (16 and 128, $p=0.019$ and
$p=0.0012$), but \textbf{only capacity 128 survives Holm-Bonferroni correction
across the four-capacity family} ($0.0012\times4=0.0049$, against
$0.019\times3=0.058$ for the next-smallest). The two non-significant cells are
underpowered rather than null: their observed $+0.0011$ and $+0.0024$ sit below
the exact per-seed MDEs of $0.0023$ and $0.0039$ ($\alpha=0.05$, $80\%$ power,
$n=100$). The correction is applied within this four-capacity family only.

\subsection{Alternative explanation 1: is it adaptivity itself?}

\textbf{Is it ``having adaptive checkpoint selection at all,'' rather than
fold reuse?} CLEAN\_MATCHED's full-budget retrain is a blind, fixed-epoch run
rather than an adaptively-selected one, so an independent review proposed that
LEAKY's advantage might reflect adaptivity itself. We tested this with a fifth
condition, CLEAN\_MATCHED\_ADAPTIVE: \emph{ongoing} adaptive checkpoint
selection during training -- the identical mechanism LEAKY uses -- but driven
by a carve-out disjoint from the reused fold
(\texttt{code/22\_epoch\_forcing\_confound\_control.py}).

\medskip\noindent\textbf{The first version of this control was invalid, and its
numbers are retracted.} A later independent review found that its selection
carve-out \texttt{es\_idx} was split out of \texttt{tr\_idx} but the arm was
then trained on the \emph{full} \texttt{tr\_idx} --- so every point in the
supposedly held-out selection set was inside that arm's own training set
throughout. Selecting a checkpoint on data the model is concurrently fitting is
not selection: in-sample AUROC is near-monotone in training, so the argmax
simply runs to the end of the budget. \texttt{code/22} now records each arm's
mean kept-checkpoint epoch, which shows this directly: the broken arm sits at
$39.4$ of $45$ epochs at capacity 128 (and $19.9$ of $45$ at 384), against
$20.8$ for LEAKY and $21.4$ for the honest arms (and $11.8$/$12.6$ at 384).
It measured convergence, not adaptive selection, and therefore could not test
the alternative it was built to test. Its previously-reported figures ---
LEAKY beating it by $+0.0198$ (cap.\ 128) and $+0.0143$ (cap.\ 384), retaining
$37.3\%$ and $25.7\%$ of honest selection's placebo-relative benefit --- are
withdrawn. The arm is still run and shipped, under the explicit key
\texttt{clean\_matched\_adaptive\_contaminated}, so the size of the artifact is
a reproducible number rather than an assertion; no conclusion rests on it.

\medskip\noindent\textbf{The rebuilt control, and what it now establishes.}
CLEAN\_MATCHED\_ADAPTIVE now trains on \texttt{tr2\_idx}, the complement of
\texttt{es\_idx} within \texttt{tr\_idx}, so training data and selection data
are disjoint by construction. This mirrors \texttt{code/43}, whose
corresponding arm already separated them correctly. Two consequences.

First, the control is now sharp, on the same scale the previous version failed
on: $(\text{CLEAN\_MATCHED\_ADAPTIVE} - \text{PLACEBO}) /
(\text{CLEAN\_MATCHED} - \text{PLACEBO})$ is $94.2\%$ at capacity 128
($+0.0247$ of $+0.0262$) and $79.9\%$ at capacity 384 ($+0.0125$ of $+0.0156$).
The condition retains nearly all of the effect it is supposed to hold constant,
which is what the previous version conspicuously did not do.

Second, LEAKY still beats it at both capacities, and by \emph{more} than it
beats the blind CLEAN\_MATCHED control: $+0.0049$ ($p=0.0008$, 58/100 seeds
positive) at capacity 128 and $+0.0059$ ($p=0.0012$, 64/100) at 384, against
LEAKY$-$CLEAN\_MATCHED of $+0.0034$ and $+0.0027$. Two successive revisions read
that comparison as evidence that the effect is attributable to fold reuse
specifically rather than to adaptive selection in general --- the first calling
it ``direct evidence'', the second softening it to ``evidence consistent with''.

\medskip\noindent\textbf{Both readings are now withdrawn, because the two arms
being compared are one arm.} A later review noticed that
CLEAN\_MATCHED\_ADAPTIVE and \texttt{code/67}'s CLEAN\_MATCHED\_85 report AUROC
means agreeing to the last float digit, and the reason is that they are the same
weights: the adaptive arm keeps its selection run's best checkpoint, that run
trains on \texttt{tr2\_idx} from torch seed $s$ and stops at epoch $e^\ast$, and
CLEAN\_MATCHED\_85 calls \texttt{train\_fixed\_epochs} on \texttt{tr2\_idx} from
the same seed for exactly $e^\ast$ epochs. Full-batch deterministic training from
a fixed seed makes those the identical trajectory, verified bitwise in every
(seed, fold) pair checked. \textbf{So the $+0.0049$/$+0.0059$ increments are not
an adaptivity measurement at all; they are the training-budget deficit,
restated.} Whether adaptivity per se rather than fold reuse drives this
mechanism is a question this paper does not answer
(Appendix~\ref{app:corrections-global}, E27).

\medskip\noindent\textbf{The confounds that made this arm unusable, and the one
that is measured.} Making the selection signal honest costs training data: the
arm trains on $448 \times 0.85 = 381$ samples against LEAKY's $448$, an
$\approx$$15\%$ deficit. And it still selects against a $67$-sample carve-out
where LEAKY selects against its full $112$-sample validation fold ($560/5$), a
$1.67\times$ difference in selection-set size whose extra noise would inflate
the reported gap rather than merely accompany it. Neither can be removed
without giving up the other: an exact budget match and an honest selection
signal cannot both be had inside one fold, because the selection points have to
come from somewhere. The first is now measured directly. \texttt{code/67}'s
CLEAN\_MATCHED\_85 is CLEAN\_MATCHED with the blind retrain run on
\texttt{tr2\_idx} instead of \texttt{tr\_idx} ($380$ samples against $448$);
selection rule, selected epoch, seeds, folds, initialization and the downstream
OOF/test-feature pipeline are identical, so the difference is attributable to
training-set size alone, and the recomputed CLEAN\_MATCHED arm reproduces
\texttt{code/22}'s shipped per-seed values exactly, asserted at runtime. The
deficit is $+0.0015$ at capacity 128
(BCa 95\% CI $[-0.0003,+0.0035]$, Wilcoxon $p=0.29$, $54/100$ seeds positive)
and $+0.0031$ at capacity 384 (CI $[+0.0004,+0.0057]$, $p=0.005$, $65/100$),
which is $31\%$ and $53.5\%$ of the increments above --- and, given that the two
arms are the same weights, is what those increments consist of together with the
selection-set-size difference. At capacity 384 it is larger than the whole
LEAKY-minus-CLEAN\_MATCHED gap of $+0.0027$
(\texttt{results/adaptivity\_control\_budget\_deficit.json}). The
adaptivity question is left to the factorial of §4.3, whose training-depth
factor is null but which varies depth rather than adaptivity, so it constrains
the question without settling it.

\subsection{Alternative explanation 2: downstream feature averaging}

\textbf{Is the measured severity small only because the pipeline averages it
away?} The primary harness (§4.3) --- like MultiHaluDet's own
\texttt{run\_pipeline.py:108} that it mirrors --- accumulates each
fold-model's test-set activations and divides by $K$ before the meta-learner
scores them. A review pointed out that this averages the activations of five
independently-initialized models, so if the leaked signal is fold-specific it
could be largely cancelled before the reported metric ever sees it: the
mechanism would not be benign, the instrument would be destroying the effect.
The obvious version of the test, ``rerun with $K=1$,'' is not runnable, and
not for budget reasons: with one fold there is no out-of-fold partition, so no
OOF features exist to train the meta-learner on. \texttt{code/55} instead
reads out the \emph{same trained fold-models} three ways --- averaged (the
shipped pipeline), un-averaged (the meta-learner scores each fold-model's own
test features and the $K$ AUROCs are averaged after scoring, so no activation
is ever mixed), and the OOF matrix read directly.

\textbf{The alternative explanation is supported, and we report it as such.}
Removing the averaging raises the LEAKY$-$CLEAN\_MATCHED gap at every
capacity, by roughly $2.3$--$5.8\times$: $+0.0073$/$+0.0066$/$+0.0084$/$+0.0084$
un-averaged at capacities 16/48/128/384, against
$+0.0019$/$+0.0011$/$+0.0036$/$+0.0024$ averaged --- and it is significant at
all four capacities un-averaged, against two of four averaged. Reading the OOF
matrix directly agrees ($+0.0058$/$+0.0096$/$+0.0086$/$+0.0042$, significant
at all four). A diagnostic confirms the premise: cross-validating the
meta-learner \emph{across} the OOF matrix, training on rows from some
fold-models and scoring rows from others, sits at chance
($0.484$--$0.522$ for every condition and capacity), i.e.\ the fold-models'
feature spaces carry essentially no information about one another, so
averaging their activations really is mixing incompatible bases. We keep both
readings.

The \emph{averaged} number is the correct estimate of inflation in
what the audited pipeline actually reports, because averaging is what that
pipeline does --- so §4.3's table stands unchanged as the estimate of what the
audited pipeline's own reported metric would carry. The \emph{un-averaged} number is the correct estimate of the
mechanism's severity at its source, and it is larger by
$2.3$--$5.8\times$ (mean $3.9\times$). The
honest summary is therefore narrower than ``this mechanism is small'': it is
small \emph{as reported}, partly because a downstream ensembling step
attenuates it, and a pipeline with the same leak but without that averaging
step should be expected to show more. Full results:
\texttt{results/oof\_averaging\_control.json}.

\subsection{A second generative process: anisotropic covariance}

\textbf{A second, structurally different generative process -- with two
disclosed confounds of its own.} §4.3's primary sweep draws each class from an isotropic Gaussian, so a toy-model artifact is a live alternative explanation.
We built a second generative process that keeps the controlled sweep design
but replaces the covariance structure: fit the real pooled within-class
covariance and mean-difference direction from the real Mistral-7B/HaluEval features used in §4.3, then rescale the mean difference (keeping the real,
anisotropic, correlated covariance shape exactly as observed) to hit the same
AUROC$=0.80$ target via the binormal identity
(\texttt{code/27\_anisotropic\_covariance\_capacity\_sweep.py}, $n=100$ seeds,
same four capacities). Two confounds limit how cleanly this isolates
covariance shape: the feature dimensionality changes from 64 (isotropic) to
414 (real features' native dimension), a $6.5\times$ change in the
dimension-to-sample ratio that itself affects checkpoint-selection dynamics;
and the realized empirical AUROC under the trained MLP is $0.672$-$0.681$
across capacities, not the $0.753$-$0.765$ the isotropic sweep achieves
despite both targeting the same Bayes-optimal $0.80$ --- so some of the gap
difference between them is not attributable to covariance shape alone.

With those disclosed, the pattern: LEAKY vs.\ CLEAN\_MATCHED is $+0.0082$
($p=0.007$, capacity 128, vs.\ $+0.0036$ isotropic) and $+0.0099$ ($p=0.009$,
capacity 384, vs.\ $+0.0024$, not significant, isotropic), while at the two
lower capacities it weakens (capacity 16: $+0.0000$, $p=0.85$; capacity 48:
$+0.0016$, $p=0.22$) and CLEAN\_MATCHED vs.\ PLACEBO becomes significant
instead at those same two ($+0.0084$, $p=0.015$; $+0.0080$, $p=0.010$).
Reporting all four capacities for that second comparison, not only where it is
significant: capacity 128 is $+0.0024$ ($p=0.45$) and capacity 384
\emph{reverses sign} to $-0.0037$ ($p=0.11$), neither significant ---
permuted-label checkpoint selection nominally outperforming honest checkpoint
selection, the same sign anomaly the epoch-count diagnostic (A.10) was built to explain and which does not recur under either corrected real-feature
calibration. Only two of the four capacities (128 significant in both, 48
non-significant in both) are significance-concordant between the sweeps, so
the capacity-by-capacity pattern reshuffles rather than replicating cleanly.

Pooling all eight synthetic LEAKY-vs-CLEAN\_MATCHED tests (both sweeps) into
one Holm-Bonferroni family, rather than treating the anisotropic sweep as a
separate robustness check, leaves \emph{two} of the eight significant at the
family-wise $0.05$ level --- both capacity-128 cells ($0.0012\times8=0.0099$
and $0.0071\times7=0.049$), with the procedure stopping at the anisotropic
capacity-384 cell ($0.0087\times6=0.052$). \emph{This verdict moved in our own
favour} when the seed-decoupling fix landed: under the previous,
seed-confounded isotropic run the same pooled family retained nothing at all
($0.007\times8=0.056$). We flag the direction of that movement explicitly ---
a result that improves when its own instrument is corrected deserves the most
scrutiny, not the least. What survives either way is the capacity-128 cell, in
both generative processes. We report the anisotropic sweep's $p$-values
uncorrected, in a robustness-check role, not as a second independent
confirmation carrying equal weight to the pre-registered isotropic family.
Taken together this is evidence against the isotropic-covariance assumption
alone being responsible for the severity estimate, and does not shrink or
eliminate the gap -- but it is weaker, less clean evidence than a simple
``replicates and is larger'' summary would suggest.

\subsection{The discriminating experiment: a fixed-$d$ eigenspectrum sweep}

\textbf{The discriminating experiment: covariance shape or dimensionality?}
The comparison in A.5 cannot say, because the two sweeps differ in both. An
independent review named the experiment that separates them, and it is cheap:
hold $d=64$ --- the isotropic sweep's dimension --- and vary \emph{only} the
within-class eigenvalue profile (\texttt{code/56}). We fix a single random
orthonormal eigenbasis shared by every cell, so the eigen\emph{basis} does not
move either; renormalize every spectrum to $\operatorname{tr}(\Sigma)=d$, so
total within-class variance is identical everywhere and only its distribution
across directions differs; use one fixed mean-difference direction with equal
components in that eigenbasis, the neutral choice in that it does not
preferentially align $\Delta\mu$ with the high- or low-variance directions of
any spectrum; and set its length per cell so the Mahalanobis
$J=\Delta\mu^\top\Sigma^{-1}\Delta\mu$ hits the same $\text{AUROC}_0=0.80$
target exactly (verified to $10^{-9}$; checked directly rather than through
\texttt{sanity\_checks.py}, whose bias correction assumes $\Sigma=I$ --- see A.1, issue 11). Cells: power-law spectra $\lambda_i\propto i^{-\beta}$
for $\beta\in\{0,0.5,1,2\}$, where $\beta=0$ is an exact isotropic control,
plus \texttt{real\_top64}, the top-64 eigenvalues of the same real within-class
covariance the $d=414$ sweep uses.

\textbf{Covariance shape does move severity at fixed $d$ --- but the
experiment buys one separation at the cost of another.} At capacity 128 the
gap rises monotonically with anisotropy: $+0.0015$ ($\beta=0$, not
significant), $+0.0038$, $+0.0044$, $+0.0104$ ($\beta=2$), with
\texttt{real\_top64} at $+0.0048$ --- a roughly $7\times$ span across
nothing but the eigenvalue profile. So the isotropic-vs-anisotropic difference
above is \emph{not} purely a dimensionality artifact, which is what this
experiment was built to establish. \textbf{But anisotropy and the achieved
operating point are perfectly rank-confounded in this design}, and unavoidably
so: concentrating the within-class variance into fewer directions makes the
task harder for the MLP even at a fixed Bayes-optimal $J$, so LEAKY falls from
$0.7523$ ($\beta=0$) to $0.5826$ ($\beta=2$) across exactly these cells. At
capacity 128 the gap is perfectly rank-inversely ordered by that operating
point (Spearman $-1.00$ across all five cells) --- which is §5's relationship
again, and means these data cannot distinguish ``anisotropy raises severity''
from ``anisotropy lowers the operating point, and a lower operating point
raises severity.'' We do not claim to have distinguished them. \textbf{The
ordering is also not stable across capacity}: at capacity 384 the same five
cells give $+0.0043$/$+0.0036$/$+0.0062$/$+0.0039$ (n.s.)/$+0.0056$, a Spearman
of $-0.10$ --- no monotone pattern at all. Pooling all ten cells gives Spearman
$-0.73$ ($p=0.016$). One calibration note that should temper how finely any of
these are read: the $\beta=0$ control is nominally the same configuration as
\texttt{code/02d}'s capacity-128 cell but draws different samples, and gives
$+0.0015$ ($p=0.60$) against \texttt{code/02d}'s $+0.0036$ ($p=0.0012$) --- seed-draw
noise at an effect size of a few thousandths of an AUROC point, and the reason
these cells should be compared \emph{within} \texttt{code/56} (against its own
$\beta=0$ control) rather than against \texttt{code/02d}. Full results:
\texttt{results/eigenspectrum\_sweep\_fixed\_dim.json}.

\subsection{Real-feature calibration, in full}

Feeding real Mistral-7B/HaluEval features ($n=400$, MultiHaluDet's own
unmodified feature-extraction code, a 4-bit AWQ checkpoint pinned to a fixed
revision) through the same 5-fold out-of-fold-plus-meta-learner
LEAKY/CLEAN\_MATCHED/PLACEBO architecture used in §4.3 gives LEAKY beating
CLEAN\_MATCHED by only $+0.0002$ AUROC ($p=0.56$, $n=100$ seeds) at capacity
128 and $+0.0001$ ($p=0.96$) at capacity 384. \textbf{That comparison is not
usable as reported: every condition operates at AUROC $\approx0.985$ on these
raw real features} -- a task with no room to fail leaves no room for a leakage
bug to show inflation either, and an absolute-AUROC-point MDE computed at a
$0.80$ operating point is not comparable to an effect measured at $0.985$.

Calibrating the real features down to the synthetic sweep's operating point
took four rounds of correction, two of which found this paper's own
measurement instrument committing the very pattern it audits for. Issues 5, 6, 8, 9 and 13 of A.1 record the full progression; two findings matter to a
reader of this section. \emph{First}, the transform used to rescale the
features centered class means on the \emph{full} labeled sample before any
train/test split existed. Because each class's per-sample deviations sum to
exactly zero over the sample used to estimate the class means, any subsequent
split mechanically forces the two halves' leftover class-mean directions into
an algebraic identity ($n_{\text{tr}}\cdot\overline{\Delta}_{\text{tr}} =
-n_{\text{te}}\cdot\overline{\Delta}_{\text{te}}$), verified directly at
$\alpha=0$: $\cos(\Delta\mu_{\text{train}},\Delta\mu_{\text{test}})=-0.9999$,
and a linear model fit on train and scored on test gives AUROC $=0.06$ --- not
``zero separation,'' but near-perfect, mechanically-induced anti-correlation.
That is exactly the pattern this paper's own checklist names (§6). \emph{Second},
the obvious fix --- re-centering on train indices only (\texttt{code/43}) ---
removes the algebraic artifact ($\cos = 0.0000\pm0.0007$,
train-fit$\to$test-eval AUROC $=0.58\pm0.06$) but is incomplete in precisely
the way that makes it interesting: estimating the class means from train only
fixes \emph{estimation}, not \emph{application}. The transform still decided
which class offset to subtract from each point --- test points included --- by
reading that point's own label (\texttt{mask = y == cls} ranges over the whole
array). Its signature is visible: at $\alpha=0$, where classes should be
exactly indistinguishable, that transform still leaves a
train-fit$\to$test-eval AUROC of $0.584\pm0.059$, not chance.

\textbf{The replacement: a fully label-free calibration.} It consults no
per-sample label at all. From the train indices only we estimate the class-mean
axis $u$, the midpoint $c$, and the pooled within-class SD along $u$, $s_w$;
then every row of the matrix --- train and test alike --- is passed through the
identical map $p\mapsto\alpha p+\sqrt{1-\alpha^{2}}\,s_w\varepsilon$ on its
projection $p=\langle x-c,u\rangle$, with $\varepsilon\sim\mathcal{N}(0,1)$
drawn independently. This shrinks the between-class separation along $u$ by
exactly $\alpha$ while preserving the marginal spread along $u$, so difficulty
is monotone in $\alpha$ and reaches chance exactly at $\alpha=0$ --- which it
does: $0.487\pm0.092$, statistically indistinguishable from $0.5$, versus the
label-conditional version's residual $0.584$. \emph{A simpler label-free
candidate does not work, and the negative result is load-bearing:} pure
directional shrinkage toward the midpoint,
$x\mapsto x-(1-\alpha)\langle x-c,u\rangle u$, with no noise re-injection,
consults no label --- but for any $\alpha>0$ it is an \emph{invertible linear
map}, so it cannot reduce linear separability at all: it rescales the
discriminative axis and the within-class spread along that axis by the
identical factor, leaving $d'$ unchanged. Measured behaviour confirms this:
AUROC $0.993$ at $\alpha=0.2031$ where the label-free axis-noising transform
gives $0.671$. Both transforms are retained in \texttt{code/43}'s diagnostic so
this rejection is reproducible rather than asserted.

\textbf{Tie disclosure for these Wilcoxon tests.} The paired differences
contain many tied absolute values --- $33$ of $100$ at capacity 128 and $40$ of
$100$ at capacity 384 (plus $1$ and $2$ exact zeros respectively), because
AUROC on an $80$-sample test split takes discrete values. The Wilcoxon
signed-rank statistic degrades under ties, so we report the paired sign-flip
permutation test alongside it throughout (\texttt{code/44}); here both give the
same verdict. This matters because under the \emph{superseded}
label-conditional calibration the tie counts were substantially worse ($49$ and
$60$ of $100$) and the two tests disagreed by an order of magnitude in $p$
(Wilcoxon $p=0.00064$ versus permutation $p=0.0025$ at capacity 128) --- a fact
the previous version of this section quoted the Wilcoxon $p$ without
disclosing.

\subsection{The fidelity extension: training depth and the scope of ``fidelity''}

\textbf{The two arms differ in training depth as well as in validation signal.}
That depth is chosen by the same signal, so it is not an independent nuisance:
LEAKY's kept checkpoint sits at a mean epoch of $\mathbf{29.14}$ against
\texttt{CLEAN\_MATCHED\_PLUS\_LRSCHED}'s $\mathbf{19.75}$, a gap of $+9.40$
epochs or $+47.6\%$ ($n=100$ seeds $\times$ $5$ folds, Wilcoxon
$p=7.2\times10^{-13}$); the placebo arm, whose selection signal carries no
information, stops earliest of all at $7.95$. (An earlier version of this
harness tracked this and lost the tracking when the training loop was
re-ported; it is restored in \texttt{code/49}'s \texttt{best\_epoch\_stats},
and the difference under the fully-ported configuration is \emph{larger}, not
smaller, than the version that went missing.) So the reported gap is measured
across arms that differ in training depth by roughly half. The
$\approx15\%$ data-budget difference that used to sit on top of this
(\texttt{tr2\_idx} versus \texttt{tr\_idx}) is now removed for the reported
number, which is computed against the budget-matched control
(\texttt{clean\_matched\_budget\_matched}, A.1 item 15); the es-fed arm and its
$+0.0338$ are retained for comparison only. \textbf{We do not treat the depth
difference as a nuisance to be subtracted}: choosing when to stop \emph{is} the
leaked decision this extension exists to model, and an arm forced to LEAKY's
epoch count would no longer be reacting to its own honest signal --- which is
exactly the degenerate control A.1 item 10's second fix attempt had to back out of. But a reader should know the two arms are not matched on it, and
that the depth ordering (LEAKY deepest, placebo shallowest) is itself the
signature of a more informative selection signal producing later stopping
points, not an artifact independent of the mechanism.

\textbf{Exactly what ``fidelity'' means here, and exactly what it does not.} A
review found that this extension, while porting the validation-signal coupling
faithfully, silently differed from the audited trainer on most of its optimizer
and data-pipeline settings, which made ``fidelity extension'' an overclaim as
written. Those gaps are now closed and verified against the pinned commit ---
learning rate $2\times10^{-4}$ (\texttt{config.py:17}; the harness had used
$2\times10^{-3}$, a $10\times$ mismatch), AdamW rather than plain Adam
(\texttt{trainer.py:75}), \texttt{batch\_size}$=28$ mini-batching
(\texttt{config.py:15}; the harness had been full-batch, i.e.\ exactly one
gradient step per epoch), a 5-epoch linear LR warmup during which the scheduler
is \emph{not} stepped (\texttt{config.py:21},
\texttt{trainer.py:88-91,134}), gradient clipping at $0.5$,
\texttt{min\_lr}$=10^{-7}$, \texttt{epochs}$=45$, \texttt{RobustScaler} on the
input features and \texttt{StandardScaler} on the deep OOF features
(\texttt{run\_pipeline.py:81,85,130}) --- alongside the scheduler,
early-stopping and checkpoint mechanics already ported (A.1, issue 14).
\textbf{Not ported, and this is the substantive limit on the word
``fidelity'':} the audited trainer's EMA of the weights (which it also
evaluates and checkpoints under), its composite objective ($0.45$ BCE ${+}$
$0.35$ focal ${+}$ $0.20$ asymmetric, plus a $0.20$-weighted contrastive term
on the embedding), BCE \texttt{pos\_weight} class rebalancing, label smoothing,
mixup and cutmix, and the model class itself (their 6-layer, 8-head,
384-hidden multi-scale transformer, against this harness's 3-layer MLP, which
is kept so the extension stays comparable to the checkpoint-selection-only
harness it is meant to be contrasted with). \texttt{use\_swa} appears in their
\texttt{config.py} but is not referenced anywhere in their sources, so there is
nothing to port. So: this measures fidelity of the \emph{validation-signal
coupling and the optimizer schedule}, not of the loss or the model. It is an
estimate of how much continuous scheduler and early-stopping coupling adds over
a final-checkpoint argmax within this paper's own harness --- not an estimate
of MultiHaluDet's own reported number's inflation, which would need their
objective and architecture too. The full ported/not-ported list is in
\texttt{code/49}'s module docstring; A.1 (issue 12) reports the factorial $2\times2$ ablation separating this extension's two most recent
corrections.

\subsection{The pre-registered decision rule}

Before running the sweep, we pre-registered a rule classifying each capacity as
\texttt{GENUINE\_LEAK\_CONFIRMED}, \texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK},
or \texttt{MIXED} from the placebo-relative gaps. We initially reported the
confound branch as structurally unreachable; that was wrong, and applying the
rule to every version actually run: the isotropic synthetic sweep returns
\texttt{MIXED} at all four capacities; the anisotropic sweep returns
\texttt{MIXED} at three and \texttt{GENUINE\_LEAK\_CONFIRMED} at one (capacity
128); the ceiling-confounded real-feature test returns
\texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK} at both tested capacities; the
biased-calibration, the leak-contaminated bias-corrected, and the (superseded)
label-conditional train-only-centered real-feature tests all return
\texttt{MIXED} at both capacities; and the current, label-free-calibrated
real-feature test returns \texttt{GENUINE\_LEAK\_CONFIRMED} at both. The rule
is therefore reachable in every direction --- what it is not, is \emph{stable}:
the same underlying mechanism produces all three verdicts depending on
generative process, calibration, and capacity. We flag this bluntly, including
that the verdict moved in the \emph{favourable} direction for our own
hypothesis when the calibration was corrected: that is precisely the
circumstance in which a single pre-registered rule's output should be trusted
least, and it is why this paper's severity claims rest on effect sizes with
intervals rather than on the rule's label.

\subsection{Smaller checks, retained for completeness}

\emph{Power.} Exact
per-seed gap SDs (not back-of-envelope estimates) confirm the real-feature
test is adequately powered for the gaps it reports: under the superseded
label-conditional calibration the MDEs were $0.00198$/$0.00154$ at capacities
128/384 against observed $+0.0021$/$+0.0022$ (adequate but marginal), and
under the current label-free calibration the observed $+0.0093$/$+0.0077$ sit
far above that scale, with BCa intervals excluding zero by a wide margin;
earlier, architecturally-mismatched versions were not adequately powered.
\emph{Quantization.} A single-point-estimate FP16-vs-AWQ control was re-run
with 100 seed-resampled splits through this paper's own severity
architecture: gap $+0.0068$, 95\% CI $[-0.061,+0.080]$ (includes zero) --
consistent with no confound, not a demonstration ruling one out. \emph{Epoch
count.} An anomaly specific to the now-superseded single-fold architecture
(CLEAN\_MATCHED underperforming PLACEBO by $-0.0025$) reverses sign under
every corrected version (CLEAN\_MATCHED outperforms PLACEBO by $+0.0157$
under the final, train-only-calibrated test) and does not recur; full detail
in \texttt{results/real\_feature\_leakage\_diagnostics.json}.
\emph{Cross-model feature averaging.} An independent review proposed that the
small measured severity could be an artifact of the harness's own averaging
of five fold-models' test activations rather than a property of the
mechanism; §4.3 reports the result in full, and briefly, the alternative is
\emph{supported} (\texttt{code/55},
\texttt{results/oof\_averaging\_control.json}). \emph{Covariance shape versus
dimensionality.} §4.3's isotropic ($d=64$) and anisotropic ($d=414$) sweeps
differ in two things at once; \texttt{code/56} runs the discriminating
experiment, holding $d=64$ and the eigenbasis, the total within-class
variance, the mean-difference direction and the calibrated Mahalanobis $J$ all
fixed, and varying only the within-class eigenvalue profile. Results and
reading are in §4.3.

\subsection{Does severity scale the way extreme-value theory predicts?}

\textbf{(It does scale with $K$ --- but whether it does so in the functional
form the theory specifies is a question this harness turns out not to be
instrumented to answer.)} The reasoning behind Mechanism 3 is a winner's-curse
argument: selecting the best of $K$ noisy validation estimates inflates the
reported score by $\text{gap}\approx c\cdot\sigma_{\text{val}}\sqrt{2\ln K}$,
shrinking with validation-set size $n_{\text{val}}$.
\texttt{code/47\_selection\_multiplicity\_sweep.py} tests this directly
(decoupled \texttt{data\_seed}/\texttt{split\_seed}/\texttt{fold\_seed\_base}/
\texttt{init\_seed\_base}, so no single draw can produce all three patterns by
coincidence). \textbf{Correction 1:} an earlier version of this sweep's
synthetic generator had the same calibration bug as Mechanism 5 (issue 11) --
corrected using the identical fix and the same non-tautological
\texttt{sanity\_checks.py} guardrail. Sweeping
$K\in\{1,3,5,10,15,25,45,75,135,225\}$ at fixed capacity gives a gap that rises
essentially monotonically with $K$ ($+0.0000$ at $K{=}1$ to $+0.0052$ at
$K{=}225$, with only a brief plateau between $K{=}10$ and $K{=}15$), and every
cell from $K{=}3$ upward is individually significant ($p<0.0002$ throughout).

\textbf{Correction 2: the fit as originally reported was not the model the
theory states.} \texttt{code/47} fits the EVT form with $\sigma_{\text{val}}$
replaced by a single \emph{constant} --- the mean of the per-cell
$\sigma_{\text{val}}$ across all ten $K$ values (\texttt{code/47}, lines
230-231) --- and reports $c=0.0417$, $R^2=0.756$. The theory's $\sigma$ is the
noise SD of the validation estimates \emph{in the cell being predicted}, and
the sweep measures exactly that per cell. Substituting the constant silently
converts a one-parameter EVT model into ``gap is proportional to
$\sqrt{2\ln K}$,'' a materially different and much more flattering claim,
because the measured per-cell $\sigma_{\text{val}}$ varies by $2.66\times$
across the sweep ($0.0186$ to $0.0494$) \emph{and moves in the opposite
direction from the gap}
($\mathrm{corr}(\sigma_{\text{val}},\text{gap})=-0.84$, where the theory
requires this correlation to be positive). Refit correctly, with each cell's
own measured $\sigma_K$ (\texttt{code/50\_evt\_scaling\_refit.py},
\texttt{results/evt\_scaling\_refit.json}), the EVT form gives $c=0.0355$,
$R^2=\mathbf{-0.399}$ --- worse than predicting the mean of the gaps.
Two-parameter alternatives fit far better on the same nine cells: $a+b\ln K$
with $a=-0.00012$, $b=0.00104$ gives $R^2=\mathbf{0.970}$, and $a\cdot K^{b}$
with $a=0.00083$, $b=0.378$ gives $R^2=0.866$. \emph{Correction 4 below shows
those nine cells include four that are largely degenerate, and refits the
logarithmic law on the six that are not.}

\textbf{Correction 3, which retracts Correction 2's verdict: this is not a
falsification, because $\sigma$ is the wrong quantity.} A third independent
review found that neither the constant-$\sigma$ fit nor the per-cell-$\sigma$
refit tests extreme-value theory at all. In \texttt{code/47},
\texttt{mean\_val\_auc\_std} is \texttt{np.std(val\_aucs)} where
\texttt{val\_aucs} holds one validation AUROC \emph{per epoch of a single
training trajectory}. The extreme-value derivation's $\sigma$ is the
sampling-noise SD of $K$ \emph{exchangeable} estimates scattered around a
\emph{common} mean. Successive epochs of one optimizer run are neither: they
are a learning curve, and their dispersion is dominated by the systematic
climb from initialization to convergence. Both anomalies Correction 2 cited as
evidence against the theory follow mechanically from that misspecification,
with no need for the theory to be wrong. $\sigma$ falls $2.66\times$ as $K$
rises because a longer trajectory spends proportionally more of itself
converged and flat, so the SD of the whole curve shrinks --- a fact about
learning curves, whereas genuine selection noise should be roughly flat in
$K$. And $\mathrm{corr}(\sigma,\text{gap})=-0.84$ because the gap rises with
$K$ while this $\sigma$ falls with $K$; two quantities driven oppositely by the
same variable anticorrelate regardless of whether the theory holds.

\textbf{The verdict is therefore downgraded from ``falsified'' to
``untestable in this harness as instrumented.''} $R^2=-0.399$ cannot
distinguish ``extreme value theory is wrong here'' from ``$\sigma$ is the
wrong quantity to test it with,'' and we no longer assert the former. A real
test needs $\sigma$ estimated across independent, exchangeable candidate
evaluations at a fixed point in training --- $K$ independently-seeded models,
or repeated resampled validation splits at a fixed epoch --- which this
harness does not produce. That is a measurement gap rather than a theoretical
obstacle, and Limitations (2a) no longer lists it as an obstacle to deriving
a formal bound. \textbf{What is unaffected by \emph{this} correction:} the
empirical law $\text{gap}=a+b\ln K$ never references $\sigma$ and is untouched
by the misspecification. It is, however, affected by a separate defect found
later; see Correction 4. We report all of the above as a correction of the
previous version's reported fit, which was an artifact of the
constant-$\sigma$ substitution, not a finding.

\textbf{Correction 4: the degeneracy gate had been applied to the joint grid
but never to this sweep.} An earlier round added a bitwise
\texttt{state\_dict}-identity check and wired it into \texttt{code/57}'s joint
$K\times\text{AUROC}_0$ grid, where it caused the $K=3$ column to be dropped and
a headlined interaction term to be retracted (C.1). A later independent review
observed that \textbf{the same gate was never enabled at \texttt{code/47}'s own
call sites}, even though Sweep A is what produces the $a+b\ln K$ law this paper
quotes in its abstract. \texttt{code/47} now passes
\texttt{degeneracy\_check=True} at every call site --- the check consumes no
RNG and does not touch the training path, so every previously reported gap, arm
mean, Wilcoxon $p$ and bootstrap CI reproduces bit for bit and only new fields
are added. By the same criterion \texttt{code/57} uses (bitwise-identical
fraction $\leq0.50$, read at runtime from that script rather than re-chosen),
Sweep A's small-$K$ cells are largely degenerate:

\begin{center}\small
\begin{tabular}{lcccccccccc}
\toprule
$K$ & $1$ & $3$ & $5$ & $10$ & $15$ & $25$ & $45$ & $75$ & $135$ & $225$ \\
\midrule
identical frac. & $1.000$ & $0.973$ & $0.900$ & $0.645$ & $0.355$ & $0.060$ &
$0.031$ & $0.026$ & $0.024$ & $0.020$ \\
in the fit & --- & no & no & no & yes & yes & yes & yes & yes & yes \\
\bottomrule
\end{tabular}
\end{center}

\noindent No cell sits near the threshold: the retained cells are at $0.36$ and
below, the excluded ones at $0.645$ and above, so the partition does not depend
on the exact cutoff. Refitting on the six non-degenerate cells gives
$a=-0.00042$, $b=0.00110$, $R^2=\mathbf{0.939}$, against the contaminated
nine-cell fit's $a=-0.00012$, $b=0.00104$, $R^2=0.970$. \textbf{What survives:}
the slope, which moves by $6\%$, so the monotone increasing regularity itself is
not an artifact of the degenerate cells. (That the regularity is
\emph{logarithmic} is a separate claim, and one this paper no longer makes:
Appendix~\ref{app:k-form} shows four alternative two-parameter concave forms fit
these same six cells better, and that these $R^2$ values are computed on points
whose Monte-Carlo error exceeds the fit residuals by $2.9\times$. Both $R^2$
figures are retained here because the comparison between them --- contaminated
against gated, same form --- is what this subsection is for.)
\textbf{What does not:} the apparent dynamic
range. Over the fitted cells the gap now spans $2.31\times$ ($+0.00226$ to
$+0.00522$) rather than $6.10\times$ ($+0.00086$ to $+0.00522$) --- so roughly
two-thirds of the range the original headline displayed came from cells in
which the two arms were the same model. Both fits ship
(\texttt{fits.log\_two\_parameter}, labelled contaminated, and
\texttt{fits.log\_two\_parameter\_non\_degenerate}) with the per-cell
degeneracy record beside them.

\textbf{A dynamic-range observation, reported as suggestive rather than
decisive.} Across $K=3\to225$ the measured gap grows by a factor of
$6.10\times$ ($+0.00086\to+0.00522$), while the $\sqrt{2\ln K}$ factor grows by
only $3.29/1.48=2.22\times$. (Correction 4 supplies a third reason this
observation is not evidence: the $K=3$ endpoint anchoring the $6.10\times$ is
$97\%$ degenerate, and across the non-degenerate cells the measured span is
$2.31\times$ against the $\sqrt{2\ln K}$ factor's $1.41\times$ over the same
range --- no longer a mismatch worth remarking on.) If the true sampling-noise $\sigma$ were
approximately constant across the sweep --- which it arguably should be, since
$n_{\text{val}}$ is held at $112$ throughout --- then even with the fitted $c$
free to absorb any overall scale, the functional form would be off by nearly a
factor of three in \emph{span}, not merely in fit quality. \emph{We previously
presented this as a third independent confirmation of falsification. It is
not, for two reasons.} First, its premise is exactly the quantity this harness
fails to measure: we cannot verify that the true $\sigma$ is constant, because
the only $\sigma$ available is the trajectory dispersion above. Second, the
span comparison is itself confounded by the operating point, which moves from
$0.6872$ to $0.7574$ across those same cells and which §5 shows is a strong
severity modifier. So the dynamic-range mismatch is consistent with the
extreme-value form being wrong here, and equally consistent with the span being
inflated by the operating-point drift. It is listed as an observation, not as
evidence.

The scientifically interesting content survives, and is arguably more
informative than a clean confirmation would have been: severity does rise with
the number of candidates selected among, smoothly and monotonically, and a
two-parameter logarithmic law describes it well ($R^2=0.94$ on the cells where
the two arms are not the same model); whether it
does so \emph{through} the $\sigma_{\text{val}}\sqrt{2\ln K}$ channel the
winner's-curse heuristic posits, we neither confirm nor exclude.

\textbf{Correction 5: Sweep B was mislabelled, and the isolated sweep it should
have been reverses the reading.} (An earlier revision numbered this
``Correction 4'' as well, so this appendix carried two distinct corrections
under one number; they are now distinct.) This appendix and \texttt{code/47} both
described Sweep B as an ``$n_{\text{val}}$ sweep (via \texttt{N\_SAMPLES})''.
It is not one. Varying $N_{\text{SAMPLES}}\in\{350,700,2800\}$ moves the
training set ($280/560/2240$), the test set ($70/140/560$) and the per-fold
validation set ($56/112/448$) simultaneously and proportionally, so nothing it
shows can be attributed to $n_{\text{val}}$. It is relabelled a
\emph{sample-size} sweep (\texttt{sweep\_B\_sample\_size} in the result JSON,
with the confound recorded alongside it), and its numbers are unchanged:
$+0.0026$ ($p=0.33$), $+0.0036$ ($p=0.0012$), $+0.0005$ ($p=0.047$). The
isolated sweep it should have been costs nothing extra: hold
$N_{\text{SAMPLES}}=700$ fixed --- and with it the outer train/test split and
the test set the metric is reported on --- and vary only the inner CV fold
count $K_{\text{CV}}\in\{2,3,5,10\}$, which is precisely what sets
$n_{\text{val}}=n_{\text{train}}/K_{\text{CV}}$ (\texttt{code/47}, Sweep D,
$n=100$ seeds/cell):

\begin{center}\small
\begin{tabular}{lccccc}
\toprule
$K_{\text{CV}}$ & $n_{\text{val}}$ & $n_{\text{tr/fold}}$ & gap & $p$ & LEAKY mean \\
\midrule
2  & 280 & 280 & $+0.0087$ & $1.4\times10^{-7}$ & 0.7474 \\
3  & 186 & 374 & $+0.0036$ & $0.0019$ & 0.7523 \\
5  & 112 & 448 & $+0.0036$ & $0.0012$ & 0.7586 \\
10 & 56  & 504 & $+0.0005$ & $0.22$ (n.s.) & 0.7599 \\
\bottomrule
\end{tabular}
\end{center}

§5.3 reads this sweep; two details belong here rather than there. \emph{(i)}
This sweep's $\sigma$ carries the same trajectory-dispersion misspecification
Correction 3 identifies, so its agreement with the theory's direction
($\sigma_{\text{val}}$ rising from $0.0331$ at $n_{\text{val}}=280$ to
$0.0375$ at $56$) is as uninformative as the $K$ sweep's disagreement was;
what carries weight is the gap's own direction, which does not depend on
$\sigma$ at all. \emph{(ii)} As an internal consistency check, Sweep D's
$K_{\text{CV}}=5$ cell is the same configuration as Sweep B's
$N_{\text{SAMPLES}}=700$ cell and reproduces it exactly, $+0.0036$. The
previous draft's summary --- that the $n_{\text{val}}$ pattern was
``directionally consistent with the winner's-curse account'' --- does not
survive this and is withdrawn.

Finally, the operating-point sweep
($\text{TARGET\_AUROC}\in\{0.70,\ldots,0.985\}$), whose monotone
decline §5.3 reports: that decline is cleanly monotone, but
significance is not. $0.70$, $0.80$, and $0.95$ are each significant
($p=0.0001$, $p=0.0012$, $p=0.037$), while $0.90$ and $0.985$ -- MultiHaluDet's
own operating point -- are not ($p=0.067$, $p=0.219$), the latter consistent
with the ceiling-effect ranking noise issue 5 documents rather than with the
mechanism vanishing at high AUROC specifically. This relationship is the
strongest and cleanest of the three sweeps, and is promoted to a headline
finding on that basis \emph{as a within-harness modifier}, \emph{with the scope limit §5.4 establishes}: it is
measured inside this harness, it does not transport to the real-feature
harnesses at a matched operating point (they exceed it by $14.3\times$ and
$38.3\times$ on \texttt{code/58}'s shipped numerators, $9.2\times$ and
$24.3\times$ on this paper's corrected fold-matched-carveout severities;
§5.2, §5.4), and it licenses only the requirement that
operating points be matched before severities are compared. Full per-cell
results: \texttt{results/selection\_multiplicity\_sweep.json}; refits:
\texttt{results/evt\_scaling\_refit.json}.

\subsection{Every constant in the harness, and whether it was derived}
\label{sec:constants}

A review asked, reasonably, why each hardcoded value is what it is. For several
the answer is a mechanical one we can point at; for several others the honest
answer is that the value was not derived, and saying so is more useful than
constructing a rationale after the fact. Both kinds are listed. Where a value
is undisclosed-arbitrary we also state what depends on it, so a reader can
judge the exposure rather than take our word that it is small.

\medskip\noindent\textbf{Derived, with the derivation stated.}
\begin{itemize}\itemsep2pt
\item \textbf{Early-stopping patience $15$; LR-scheduler patience $3$.} Both
  read off the audited repository (\texttt{src/config.py:19} and
  \texttt{src/training/trainer.py:76}); they are deliberately different because
  MultiHaluDet uses two distinct thresholds against the same signal. Getting
  this wrong was correction 12.
\item \textbf{The other ported optimizer settings} --- learning rate $2\times
  10^{-4}$, batch size $28$, warmup $5$ epochs, gradient clip $0.5$, minimum LR
  $10^{-7}$, $45$ epochs, AdamW, \texttt{RobustScaler} --- each cite a specific
  line of the pinned commit (correction 14, and the list in §4.3).
\item \textbf{$n_{\text{val}}=112$}: derived as $560/5$ from the five-fold
  structure, not chosen.
\item \textbf{The $81$-point threshold grid} (Mechanism 5): ported verbatim
  from the audited pipeline.
\item \textbf{$n=100$ seeds}: chosen as $10\times$ the original capacity
  sweep's $10$ specifically to resolve significance, which the smaller run
  could not (\texttt{code/02c}'s own comment records this). The MDEs in A.2 are
  computed \emph{at} $n=100$ and are therefore a description of what this
  choice bought, not a justification of it.
\item \textbf{Degeneracy gate at $0.50$ bitwise-identical folds}: set in
  \texttt{code/57} against the observed distribution --- the dropped $K=3$
  column ran at $0.87$--$1.00$ and every $K\geq15$ cell far below. It is now
  also reused, unchanged and read at runtime from \texttt{code/57}, by
  \texttt{code/50}'s corrected $K$-law fit, precisely so that a threshold is
  not re-picked after seeing which cells it removes. On Sweep A no cell sits
  near it: retained cells are at $\leq0.36$, excluded cells at $\geq0.65$.
\item \textbf{Zero-snap at $10^{-12}$} (Case Study 4): the residues being
  snapped are $\pm3.7\times10^{-17}$ and the smallest genuine estimate is
  $+0.0002$, so any threshold across roughly thirteen orders of magnitude gives
  the same partition.
\item \textbf{$50$ reps} for $\Delta_{\text{sel}}$: bounded by the shipped
  replay artifact's size ($\approx$$100$\,KB/rep, $4.0$\,MB at $50$), stated in
  B.3. This is a preference about archive size, not a constraint, and we say so
  there.
\end{itemize}

\medskip\noindent\textbf{Not derived. What each one exposes.}
\begin{itemize}\itemsep2pt
\item \textbf{$\text{AUROC}_0 = 0.80$}, the synthetic harness's calibration
  target. This is the most consequential undisclosed choice in the paper, and
  we flag it rather than defend it: §5 establishes that the operating point is
  the strongest severity modifier we measure \emph{within} a harness, so
  Mechanism 3's synthetic $+0.0011$ to $+0.0036$ is \emph{conditional on} $0.80$ and would be roughly
  $2.6\times$ larger at $0.70$ and $3.3\times$ smaller at $0.90$ (§5.3's own
  numbers). The mitigation is that the sensitivity analysis a reviewer would
  ask for is already the paper's Sweep C, run across
  $\{0.70,0.80,0.90,0.95,0.985\}$ and reported in full; and that §4.3 declines
  to compare the synthetic estimate's magnitude against the real-feature
  harness's for exactly this reason. Had $0.80$ been chosen after seeing the
  severities it produces, the whole estimate would be selection-optimistic in
  the paper's own sense; it was not, but nothing in the artifact proves that,
  and a reader should treat the operating-point-conditional reading as the
  binding one.
\item \textbf{Capacity grid $\{16,48,128,384\}$.} Roughly $3\times$ spacing
  spanning under- to over-parameterized relative to $d=64$; the spacing is
  conventional, not derived. What depends on it: the Holm--Bonferroni family in
  A.2 is defined over exactly these four cells, so the multiplicity verdict
  (one of four surviving) is a function of a grid size nobody derived. We
  therefore report effect sizes with intervals as primary and the corrected
  $p$-values as secondary, which is stated in §4.3 and is the right response to
  this exposure rather than a coincidence.
\item \textbf{$d=64$} for the isotropic synthetic features. Not derived. The
  anisotropic sweep at $d=414$ exists partly to check that conclusions are not
  a property of this value, and the fixed-$d$ eigenspectrum sweep (A.7)
  separates dimension from covariance shape.
\item \textbf{$\texttt{ES\_HOLD\_FRACTION}=0.15$.} Not derived, and it is the
  single constant generating the most disclosed confounds in this paper: the
  $\approx15\%$ budget deficit in every honest-selection arm, and the
  $1.67\times$ selection-set size mismatch against LEAKY's $112$-sample fold.
  $0.25$ would match the fold exactly at the cost of a larger budget deficit;
  we did not run it, and A.3 flags that as an open robustness gap.
  \textbf{Updated, Round 5 (E38 below):} this gap is now closed as far as the
  decisive $2\times2\times2$ factorial's shared out-of-fold pool allows it to
  be. $0.25$ itself turns out to be infeasible under that pool (a $1$-sample
  stratified complement), a sharper ceiling than the ``we did not run it''
  framing above implied; the factorial is instead re-run at
  $\{0.05,0.10,0.15,0.20\}$. The qualitative NOT-established verdict is
  unchanged at every value tested, but the specific $75.5\%$/$28.1\%$/$-6.2\%$
  decomposition is not: the selection-set-size share ranges from $38.5\%$ (at
  $0.20$) to $75.5\%$ (the shipped value, at $0.15$), with the
  selection-run-budget share correspondingly ranging from $16.6\%$ to
  $62.5\%$. Full detail in E38.
\item \textbf{Saturation threshold $0.975$} (Case Study 4). Not derived. Its
  exposure is the $12$/$12$ split it induces and the ceiling contrast §5.6
  draws from it. Mitigation: a second, strictly stronger criterion (\emph{every}
  layer above $0.975$, not just the selected one) is computed and reported
  alongside it in B.7, and the qualitative conclusion is the same under both.
\item \textbf{$K$ grid $\{1,3,5,10,15,25,45,75,135,225\}$ and
  $\{15,45,135,405\}$; $\text{AUROC}_0$ grid
  $\{0.70,0.80,0.90,0.95,0.985\}$.} Geometric-ish spacing, not derived, except
  that $0.985$ matches MultiHaluDet's own reported operating point and that
  $K=3$'s removal and $K=405$'s addition are justified in C.1. Note $0.70$ and
  $0.985$ were the numerator and denominator of a $48.6\times$ ratio an earlier
  revision headlined; that ratio was partly a statement about where we put the
  endpoints, and §5.3 now withdraws it entirely --- its denominator is
  indistinguishable from zero, so by Fieller's theorem it has no finite
  confidence interval (\texttt{code/60}). The endpoint gaps themselves are
  reported with their own $p$-values.
\item \textbf{$200$ seeds} for Sweep A and the Mechanism 5 $N$-sweep against
  $100$ elsewhere. Not derived, and the mismatch is consequential: C.3 shows it
  is why the joint fit's out-of-sample check compares two legitimately
  different values of the same cell.
\item \textbf{$10{,}000$ bootstrap resamples, $2{,}000$ permutations,
  $2{,}000$ guardrail seeds, $20$ probe-configuration reps.} Conventional. The
  first three are large enough that Monte Carlo error is well below the
  reported precision; the fourth is small and its result is reported as a
  robustness check rather than an estimate.
\item \textbf{The AUROC $\geq0.90$ case-study inclusion criterion.} Circular,
  and we say so in §3: it was derived from the observation that motivated the
  paper, so it selects for the regime the paper is about and is not a random
  sample of the literature. §6 lists the resulting non-representativeness as a
  limitation.
\end{itemize}

```

---

## Appendix E. Correction History for Case Studies 2 and 4, Mechanism 5, and the Severity Surface

```latex
\section{Correction History for Case Studies 2 and 4, Mechanism 5, and the Severity Surface}
\label{app:corrections-global}

\textbf{Why this appendix exists.} Appendix~A records how §4.3's Case Study 3
estimate was reached, issue by issue. Every other headline number in this paper
has a comparable history, and until now it was told inline: the main text
carried, beside several current figures, the superseded figure and the reason it
was superseded. That is the right information to ship and the wrong place to
ship it --- it makes the abstract, §4, §5 and the Conclusion read as a changelog
rather than as a statement of what this paper currently claims. Those
comparisons are collected here instead, so the main text can state the current
number and nothing else. Nothing below is required to verify a current claim;
each entry names what an earlier revision of this work asserted, what replaced
it, and which script and result file the replacement comes from. Entries are in
the order the corrections were made, across three rounds of remediation
following independent adversarial review.

\subsection{Round 1: estimators, contradictions, and withdrawn ratios}

\begin{description}
\item[E1. Case Study 4's severity estimator was non-negative by construction.] \emph{Previously:} §4.4's headline was
  $\Delta_{\text{wc}}=+0.0076$ (BCa 95\% CI $[+0.0048,+0.0134]$) over $17$
  non-degenerate cells, offered together with the observation that all $17$ are
  positive and a Wilcoxon $p=1.5\times10^{-5}$. \emph{What was wrong:} the
  leave-one-seed-out rotation estimator is non-negative for \emph{any} real
  matrix, not merely in the equal-argmax special case an earlier revision
  identified (Theorem~\ref{thm:wc-nonneg}, Appendix~B.6; verified over
  $300{,}000$ random matrices). Uniform positivity across $17$ files is
  therefore a restatement of the algebra, and that Wilcoxon $p$ is exactly
  $2^{-16}$, the arithmetic floor of the signed-rank test when every
  observation shares a sign. \emph{Now:} the reported estimate is the bootstrap
  max-bias estimator over \emph{all $24$} published cells, $+0.0021$. For
  comparability with the retired figure, restricted to the same $17$ cells it
  is $+0.0028$; on the $7$ rotation-degenerate cells it is $+0.0005$ rather
  than the exact zero the rotation rule assigns them. $\Delta_{\text{wc}}$
  survives as a non-negative diagnostic and an upper reference point, reported
  as such (\texttt{code/45}, \texttt{code/59},
  \texttt{results/case\_study\_4\_estimator\_audit.json}).
  \textbf{This entry's own justification was itself wrong and is corrected in
  E15:} the replacement was promoted on the ground that it ``is not
  sign-constrained and needs no degeneracy exclusion'' with ``$2$ of $24$ cells
  negative'' and a Wilcoxon $p=6.0\times10^{-7}$. All four of those claims are
  withdrawn.
\item[E2. The severity band's upper endpoint moved with E1.] \emph{Previously:}
  the abstract and Conclusion described the code-verified AUROC-scale estimates
  as landing ``between roughly $0.000$ and $0.034$ AUROC'' --- the top of the
  retired diagnostic's per-cell range. \emph{Now:} roughly $0.000$ to $0.026$,
  the range spanned by the current per-mechanism estimates; the primary Case
  Study 4 estimator's largest single cell is $+0.0122$ (§4.4). (That cell,
  \texttt{qwen2.5-7b/fever/linear}, is $0.0121527$; this entry rounded it up to
  $+0.0123$ where §4.4 and Table~\ref{tab:cs4-cells} round it to $+0.0122$, and
  a third review caught the inconsistency. $+0.0122$ everywhere.)
\item[E3. The ``$48.6\times$'' operating-point ratio is withdrawn.]
  \emph{Previously:} §5 and the abstract quoted this decline as a multiplier:
  ``severity falls by $48.6\times$'' between AUROC$_0=0.70$ and $0.985$. \emph{What was wrong:} the
  denominator cell is indistinguishable from zero, so the ratio of means has no
  finite confidence set. Fieller's condition
  $g=(t_{.975}\,\mathrm{SE}_{\text{denom}}/\text{denom})^2\geq1$ is satisfied at
  $g=2.03$ from regenerated per-seed data and $g=1.93$ from the shipped BCa
  halfwidth (an earlier revision said $1.98$, mixing a normal and a $t$
  quantile in one expression; both exceed $1$); a $200{,}000$-draw paired bootstrap places $8.1\%$ of resamples at
  or below zero in the denominator and returns a percentile interval of
  $[-340,+448]$, spanning zero, both signs and infinity. \emph{Now:} the decline
  is reported as monotone and reaching a level indistinguishable from zero
  ($+0.0002$, $p=0.219$, $n=100$), with no multiplier anywhere in the paper. The
  scientific content is unchanged; only the ratio is withdrawn (\texttt{code/60},
  \texttt{results/operating\_point\_ratio\_fieller\_check.json}).
\item[E4. The transport reference cell was called zero \emph{and} used as a denominator.]
  \emph{Previously:} §5.4 divided two real-feature severities by a synthetic
  reference cell of $+0.00065$ and, in the same breath, called that cell
  indistinguishable from zero --- which, if true, would make the $14\times$ and
  $38\times$ ratios divisions by noise. \emph{Now:} the cell's own gap does
  exclude zero (Wilcoxon $p=0.037$, BCa 95\% CI $[+0.00007,+0.00121]$), and
  §5.4 says so. The contrast with E3 is the point: the $0.985$ endpoint
  genuinely is at the floor and gets no ratio; the $\approx0.94$ reference cell
  is not, and does (\texttt{code/58},
  \texttt{results/operating\_point\_transport\_check.json}).
\item[E5. Contradiction: the FP16-vs-AWQ control was quoted at unshipped numbers.] \emph{Previously:} §4.3 reported the
  quantization control as ``identical AUROC ($0.9600$ both).'' \emph{What was
  wrong:} that matched neither the shipped result file nor Appendix~A.10.
  \emph{Now:} mean AUROC $0.9460$ (FP16) against $0.9392$ (AWQ) over $100$
  seed-resampled splits, a gap of $+0.0068$ with a resampled 95\% CI of
  $[-0.061,+0.080]$ and Wilcoxon $p=0.148$ --- consistent with no quantization
  confound rather than a demonstration ruling one out at this resolution.
\item[E6. Case Study 2's separability claim was asserted, not computed.] \emph{Previously:} §4.2 stated that
  GUARDIAN's two sequential halves are ``separable by a hidden-state probe at
  AUROC $0.734$--$0.776$,'' and that an all-data argmax selects L19. \emph{What
  was wrong:} neither number came from any computation in this repository; both
  existed only as hardcoded prose. \emph{Now:} \texttt{code/61} fits the
  half-membership probe, which reaches out-of-fold AUROC $0.7885$ at L25, mean
  $0.7259$, above $0.70$ at $25$ of $32$ layers; the all-data argmax is L21
  ($0.7829$). The same pass found a shorter route to the same conclusion that
  had been missed entirely --- the two halves differ in \emph{label base rate}
  by $8.0$ percentage points ($0.7300$ against $0.8100$), which is on its own
  enough to make their AUROCs non-comparable --- and a base-rate-matched control
  showing that matching it does not rescue the sequential split
  ($\Delta_{\text{sel}}=-0.0050$, SD $0.0058$, over $20$ draws). The direction
  and the conclusion of the original retraction are unchanged; its evidence is
  now computed (Appendix~\ref{sec:cs2-provenance}).
\end{description}

\subsection{Round 2: measurement rules, null references, and functional form}

\begin{description}
\item[E7. Mechanism 5's F1 gap used a threshold the pipeline discards.] \emph{Previously:} the F1 gaps were reported as
  $+0.0212$ to $+0.0338$, measured at the argmax over
  \texttt{find\_best\_thresholds}' $81$-point F1 grid. \emph{What was wrong:}
  \texttt{run\_pipeline.py:139-140} passes only \texttt{thresholds['youden']} to
  \texttt{evaluate\_all}, so every threshold-dependent metric MultiHaluDet
  reports --- F1 included --- is scored at the Youden threshold. \emph{Now:}
  \texttt{code/62} re-measures at the repo-faithful rule with everything else
  held fixed, and re-emits the F1-argmax column as a check that it reproduces
  \texttt{code/46}'s shipped cells to float64. The correction moves the finding
  in this paper's favour twice over: the gaps rise to $+0.0225$ to $+0.0520$,
  larger at every operating point and by up to $2.28\times$ at the low-AUROC
  end; and the measurement becomes a measurement, since the F1-argmax gap was
  non-negative by algebra ($0/1000$ negative reps) while the Youden gap is
  genuinely two-sided ($4$ to $67$ negative reps of $200$). §4.5 keeps that
  comparison in the main text rather than here, because which threshold rule a
  severity is measured at is a scientific choice a reader has to be able to
  audit (\texttt{results/mechanism5\_youden\_threshold.json}).
\item[E8. $\Delta_{\text{sel}}$ was tested against a null we call false.] \emph{Previously:} §4.2 noted that $\Delta_{\text{sel}}$ is
  positive in expectation under a pure-noise null and then reported a Wilcoxon
  test of $H_0:\Delta_{\text{sel}}=0$. \emph{Now:} \texttt{code/64} computes the
  mechanical null by permuting which layer's held-out AUROC is paired with which
  layer's CV AUROC, leaving both marginals, the argmax rule and the layer count
  untouched: the null mean is $+0.0429$ (SD $0.0043$), and the observed
  $+0.0255$ sits below that entire interval. The statistic then decomposes
  exactly into the winner's curse on the selection criterion ($+0.042865$,
  matching the simulation to $7\times10^{-5}$) minus what the selected layer is
  genuinely worth out of sample ($B=+0.0173$, $t=9.07$). The reading changes
  from ``significantly positive'' to ``$40.4\%$ of the winner's curse is bought
  back by the selected layer being genuinely better, and $+0.0255$ is the
  residual'' --- a stronger result than the one it replaces
  (Appendix~\ref{app:delta-sel-null}).
\item[E9. The $K$-law's functional form was asserted, not established.]
  \emph{Previously:} §5.3 reported the candidate-count relationship as
  $\text{gap}=a+b\ln K$ with $R^2=0.939$. \emph{What was wrong:} with per-seed
  data now shipped (\texttt{code/63}), the Monte-Carlo standard error of the six
  fitted cell means averages $7.72\times10^{-4}$ against an RMS fit residual of
  $2.63\times10^{-4}$ --- the noise on the points is $2.9\times$ the misfit, so
  $R^2=0.939$ measures how monotone the cells are, not how logarithmic. Four
  other two-parameter concave forms beat $\ln K$ on $R^2$, AICc and
  leave-one-out simultaneously, and a parametric bootstrap propagating each
  cell's own error puts $P(\ln K$ best$)$ at $0.063$. \emph{Now:} the
  relationship is stated as monotone and concave in $K$, with the slope
  $b=+0.00110$ ($95\%$ CI $[+0.00040,+0.00180]$) reported and the functional
  form explicitly not identified. This matters operationally, not only
  statistically: $\ln K$ implies unbounded growth in candidate count while the
  two best-fitting alternatives imply a ceiling (\texttt{code/68},
  Appendix~\ref{app:k-form}).
\item[E10. The operating-point axis had no control for variance compression.]
  \emph{Previously:} no revision of this paper addressed the strongest
  alternative account of its largest effect --- that as AUROC approaches $1$ the
  sampling variance of any AUROC estimate collapses against the bound, so every
  difference of AUROCs must shrink near the ceiling whether or not the leakage
  changes. \emph{Now:} \texttt{code/69} re-expresses the same $100$ paired
  per-seed contrasts per cell in three stabilized coordinates chosen to be
  different in kind, and the verdict splits. The decline is real --- all three
  fall several-fold, and a pure compression account, which predicts the mean and
  its seed-to-seed SD fall at the same rate, fails, the mean's decline
  outrunning the SD's by a factor of $2.7$. But most of the raw decline's
  \emph{size} is coordinate effect: on a log scale $0.51$ of it survives in
  probit units, $0.70$ on the rank measure and only $0.25$ standardized, and
  strict monotonicity survives in none of them. This is the second, independent
  reason --- alongside E3 --- that the paper quotes no multiplier for this axis
  (Appendix~\ref{app:variance-control}).
\end{description}

\subsection{Smaller items from the same two rounds}

\begin{description}
\item[E11. The candidate-count exponent is disclosed as unstable.] It spans $b=0.09$ to $0.21$ under single-column deletion and moves
  further, to $b=0.29$, under ordinary log-space least squares instead of
  Gauss-Newton --- a larger move than deletion produces, which is why it is
  reported as unstable rather than as a measured exponent. The operating-point
  coefficient, which is what the paper's claims rest on, is stable under both
  (\texttt{code/65}, Appendix~C.2).
\item[E12. Case Study 4's permutation null is now reported in both tails.] The
  null had been computed only for the retired diagnostic and quoted only where
  it was favourable. \texttt{code/66} runs it for the estimator that is now
  primary, enumerating all $3^3=27$ resamples exactly rather than sampling:
  $0$ of $24$ cells sit significantly above their own null and $20$ of $24$ sit
  significantly below, which is what a null built to maximize the curse does to
  cells whose argmax is stable (§4.4).
\item[E13. The adaptivity control's budget deficit is an upper bound.] Its selection carve-out is smaller than LEAKY's
  fold and costs it $\approx15\%$ of LEAKY's training data; both differences
  bias against the control, so §4.3's $+0.0049$/$+0.0059$ are upper bounds on
  the fold-reuse effect rather than point estimates of it (Limitations (4)).
\item[E14. Two literatures the paper should have cited are now cited.] The
  adaptive data analysis and post-selection-inference results (Dwork et
  al.\ 2015a, 2015b; Blum \& Hardt 2015; Berk et al.\ 2013; Taylor \&
  Tibshirani 2015) are the natural home of the bound Limitations (2a) names as
  open, and §2 states both what they supply and what they do not. The
  hidden-state hallucination-probing subfield whose evaluation protocol this
  paper audits is likewise named there, including HaloScope (Du et al.\ 2024),
  whose layer and threshold selection §6.2's scanner flags and whose code, read
  directly, turns out to be correct --- a positive data point reported as such.
\end{description}

\subsection{Round 3: the replacement estimator, calibration, control asymmetries, and the scope of the title}

\begin{description}
\item[E15. The estimator promoted in E1 is non-negative too.]
  \emph{Previously:} E1 retired $\Delta_{\text{wc}}$ because
  Theorem~\ref{thm:wc-nonneg} shows it cannot be negative, and promoted the
  bootstrap max-bias estimator $\Delta_{\text{boot}}$ on the stated ground that
  ``its two sides are computed on different seed multisets, so nothing
  telescopes, and it is not sign-constrained --- $2$ of the $24$ cells return
  negative values, which is the property $\Delta_{\text{wc}}$ structurally
  cannot have.'' \emph{What was wrong:} nothing telescopes, and the estimator is
  sign-constrained anyway. A bootstrap resample of seeds is mean-preserving at
  each fixed layer, the maximum is convex, and Jensen's inequality gives
  $\Delta_{\text{boot}}\ge0$ for any data whatsoever
  (Theorem~\ref{thm:boot-nonneg}, Appendix~B.6), with equality exactly when the
  grand-mean argmax is bootstrap-stable --- the same species of degeneracy that
  retired $\Delta_{\text{wc}}$. The two ``negative'' cells were Monte-Carlo
  noise from a $2{,}000$-draw sampler; enumerating all $3^3=27$ resamples
  exactly, they are $\pm10^{-16}$, and they are precisely the two
  bootstrap-stable cells. \emph{Now:} \texttt{code/45} computes
  $\Delta_{\text{boot}}$ by exact enumeration; the Wilcoxon
  $p=6.0\times10^{-7}$ is withdrawn, as is every claim resting on the
  estimator's sign, on ``$2$ of $24$ negative'', or on ``needs no degeneracy
  exclusion'' (true only of \emph{equal-argmax} degeneracy, and only on $22$ of
  the $24$ cells). The point estimate is retained, unchanged at $+0.0021$, as a
  \emph{magnitude} (\texttt{code/70},
  \texttt{results/case\_study\_4\_exact\_enumeration.json}).
\item[E16. Case Study 4's estimators are now calibrated.]
  \emph{Previously:} no Case Study 4 estimator had ever been run on data whose
  true selection bias is known, so nothing established that any of them
  recovers the right magnitude, or what any of them reports when the true bias
  is zero. \emph{Now:} \texttt{code/71} generates data at the audited design
  from a two-way model in which the true bias is known by construction, and
  sweeps it from its maximum down to exactly zero with seed noise still
  present. At the zero condition all three estimators return $0$ and none is
  ever negative. Elsewhere the ratios are stable: $\Delta_{\text{boot}}$
  recovers $0.50$--$0.55\times$ of the truth, its textbook variant
  $0.28$--$0.30\times$, and the retired $\Delta_{\text{wc}}$
  $1.27$--$1.34\times$ --- so the estimator this paper promoted understates by
  about a factor of two, the one it retired is the best calibrated of the three
  for magnitude, and the calibration-corrected severity is $+0.0029$ to
  $+0.0040$ rather than $+0.0021$. (E31 records the marginal sweeps' subsequent
  disagreement and the joint sweep that resolves it in favour of this range.)
  The $24$-cell BCa interval is no longer
  offered as a confidence interval for the severity: in simulation it covers
  the true bias in $0\%$ of runs for both bootstrap estimators
  (\S\ref{sec:cs4-calibration}).
\item[E17. The ceiling decomposition used the retired estimator.]
  \emph{Previously:} §4.4 and §5.5 reported ceiling-saturated $+0.0042$ against
  non-saturated $+0.0157$, ``a $3.7\times$ difference'', unlabelled as to
  estimator, and with a numerator drawn from a $5$-cell sub-selection its
  denominator did not undergo. \emph{Now:} under $\Delta_{\text{boot}}$ the
  split is $+0.0016$ against $+0.0026$, a $1.6\times$ difference on all $24$
  cells; across the three estimators the multiplier ranges over
  $1.5$--$5.5\times$ and the direction is robust. Every number in the passage
  now names its estimator, as does the associated power statement (SD $0.0027$
  and MDE $+0.0015$ under $\Delta_{\text{boot}}$, against SD $0.0079$ and
  $+0.0045$ under $\Delta_{\text{wc}}$).
\item[E18. The title's ``rather than mechanism'' claim is withdrawn.]
  \emph{Previously:} the title asserted that severity tracks candidate count
  and operating point ``rather than mechanism'', and the abstract that severity
  is ``governed not by \emph{which} mechanism is responsible but by two
  continuous quantities''. \emph{What was wrong:} §5.4 already withdraws exactly
  that comparison --- a transport check finds mechanism-and-harness differences
  an order of magnitude above prediction at matched operating points --- and the
  paper's own
  three ranges, placed on one scale, put the operating-point axis at
  $48.6$--$68.3\times$, mechanism-and-harness at up to $38.3\times$, and
  candidate count at $1.6$--$7.1\times$ --- and, restricted to sound
  denominators, $2.2$--$14.3\times$, $9.2$--$24.3\times$ and $1.8$--$4.2\times$
  (the mechanism-and-harness figure using this paper's corrected
  fold-matched-carveout severities as numerators; the superseded shipped
  numerators gave $14.3$--$38.3\times$),
  with mechanism-and-harness at least comparable to the operating point (E30
  records why the stronger ``largest of the three'' reading was later dropped,
  and E36 records the numerator correction).
  The claim was not merely
  unproven; the paper's own numbers contradict it. \emph{Now:} the title claims only the
  within-harness result, the abstract states the scope and names the transport
  failure, and §5.2 tabulates the three ranges side by side
  (Table~\ref{tab:magnitude-triangle}).
\item[E19. ``24 independent result files'' is corrected.] These are a
  fully crossed $3\times4\times2$ design from one artifact sharing three seeds,
  not $24$ independent files; dataset alone explains $32\%$ of the between-cell
  variance. A dataset-clustered interval, $20\%$ wider than the i.i.d.\ one, is
  now reported alongside it (§4.4, Appendix B.4).
\item[E20. Mechanism 3's control carried a second, unsized budget asymmetry.]
  \emph{Previously:} §4.3 disclosed that CLEAN\_MATCHED retrains on the full
  fold after choosing its epoch count on $85\%$ of it, and \texttt{code/67}
  sized that. It never noted that the two arms' \emph{selection runs} are
  trained on different amounts of data at all --- LEAKY's on $100\%$ of
  \texttt{tr\_idx}, the control's on $85\%$ --- and that asymmetry was never
  sized. \emph{Now:} \texttt{code/75} sizes it two ways. Resizing the in-fold
  carve-out to the exactly fold-matched $0.25$ (which is $1/(K_{\text{CV}}-1)$,
  not $1/K_{\text{CV}}$) leaves no capacity significant, moving the
  Holm-surviving capacity-128 cell from $+0.0036$ to $+0.0016$. Moving the
  carve-out out of the fold entirely, so the honest selection run also sees
  $100\%$ of \texttt{tr\_idx}, gives a mean gap of $+0.0007$ against the
  primary $+0.0023$, with none of the four BCa intervals excluding zero.
  \textbf{The primary synthetic band $+0.0011$ to $+0.0036$ therefore does not
  survive as a detectable effect once this second asymmetry is removed}, and
  §4.3 now reports the attenuation next to the band. Honest checkpoint
  selection's value relative to selecting on noise is unaffected (it still
  beats the placebo by $+0.0124$ to $+0.0361$); what goes away is the
  fold-reuse residual on top of it. Relative to a reference arm that performs
  no selection at all (fixed depth $20$, \texttt{code/77}), however, checkpoint
  selection of either kind costs $-0.0026$ AUROC in this harness ($p=0.003$,
  $34/100$ seeds positive) --- so ``honest selection's value'' should be read
  as its value against a noise-selecting baseline, not against not selecting
  (E29).
\item[E21. Mechanism 5's two arms were not size-matched, and that was never disclosed.]
  \emph{Previously:} §4.5 reported its F1 and accuracy gaps with
  LEAKY selecting a threshold on the same $n_{\text{test}}=140$ points it is
  scored on while HONEST selects on $n_{\text{val}}=112$ --- a $1.25\times$
  asymmetry in selection-set size, which inflates the measured gap on top of the
  leakage. §4.3 discloses the exactly analogous asymmetry for Mechanism 3; this
  one was not disclosed at all. \emph{Now:} \texttt{code/74} runs a size-matched
  third arm ($n_{\text{val}}=n_{\text{test}}=140$, $200$ seeds, paired on seed)
  plus a budget control isolating the $28$ training samples that size-matching
  costs. At MultiHaluDet's own $0.985$ operating point the size-matched F1 gap is
  $+0.0214$ against the shipped $+0.0225$ (paired difference $-0.0012$,
  $p=0.67$) and accuracy $+0.0204$ against $+0.0215$; isolating size alone leaves
  $-0.0004$ ($p=0.51$). The asymmetry accounts for about $2\%$ of the reported
  gap and at most $8\%$ at any of the five operating points, so it is real, it
  should have been disclosed, and Mechanism 5's finding stands. The same script
  established that the accuracy gap at the Youden rule is near-structurally
  non-negative ($0$ of the $105$ exactly balanced replicates is negative), which
  §4.5 now reports in place of treating accuracy as independent confirmation of
  the F1 result (\texttt{results/mechanism5\_size\_matched.json}).
\item[E22. The $K$-severity law was measured on epochs and used as a law about candidates.]
  \emph{Previously:} the slope $b=+0.00110$ was quoted wherever the
  paper reasoned about selection among $32$ layers, $33$ layers or $81$
  thresholds, although \texttt{code/47}'s $K$ is literally \texttt{EPOCHS} --- a
  single optimizer trajectory, which this paper's own Appendix A.11 argues is
  \emph{not} exchangeable, and uses that non-exchangeability to retract an
  extreme-value claim. \emph{Now:} \texttt{code/73} re-runs Sweep A with $K$
  indexing independently seeded probes trained to a fixed epoch count. The
  downstream relationship does not reproduce: gaps of $-0.0022$, $-0.0009$,
  $-0.0029$ and $-0.0021$ at $K=2,5,15,45$, every interval containing zero,
  slope $b=-0.00019$. It is underpowered --- $30$ seeds against Sweep A's $200$,
  a minimum detectable gap of $+0.0099$ against the $+0.0042$ it is testing for,
  and a slope interval overlapping \texttt{code/47}'s --- so §5.3 reports it as
  a failure to reproduce at inadequate power and not as a demonstrated
  contradiction. Two things are gained rather than lost: with the correct
  $\sigma$, the winner's curse is present at the \emph{selection stage} and
  matches the \emph{exact} $E[\max$ of $K$ standard normals$]$ to within $2\%$
  at every $K$ (not the $\sqrt{2\ln K}$ closed-form approximation to it, which
  is itself $20$--$40\%$ off at these $K$), so Appendix
  A.11's ``untestable in this harness'' verdict on the extreme-value form
  becomes ``supported where it applies, and not a predictor of reported-metric
  severity here''; and the $K$-law now carries a candidate-type qualifier
  wherever it appears (\texttt{results/exchangeable\_candidate\_sweep.json}).
\item[E23. The magnitude triangle's raw ratios carried the same defect as the withdrawn ``$48.6\times$''.]
  \emph{Previously:} the paper's three severity
  axes had never been placed on one scale; when they first were, the
  operating-point axis was quoted at $48.6$--$68.3\times$ and candidate count at
  $1.6$--$7.1\times$. \emph{What was wrong:} every $\text{AUROC}_0=0.985$ cell
  and the low-contrast $K=15$ cells have intervals reaching zero, so both of
  those raw spans have no finite upper confidence limit --- precisely the defect
  for which E3 withdraws the ``$48.6\times$''. \emph{Now:} \texttt{code/72}
  restricts each axis to denominator-sound cells, giving $2.2$--$14.3\times$ for
  the operating point, $14.3$--$38.3\times$ for mechanism-and-harness and
  $1.8$--$4.2\times$ for candidate count, with propagated intervals reported
  alongside them so no ordering is read as an established separation. The
  ordering reverses far enough for E18's retitle to follow: the
  mechanism-and-harness axis is at least comparable to the operating-point axis,
  and candidate count is smallest (§5.2,
  \texttt{results/magnitude\_triangle.json}). \emph{Qualified by E30:} the
  stronger reading, that mechanism-and-harness is the largest of the three, is
  withdrawn. \emph{Further corrected by E36:} the $14.3$--$38.3\times$
  mechanism-and-harness figure here used numerators later superseded by §4.3's
  fold-matched-carveout correction; the current figure is $9.2$--$24.3\times$.
\item[E24. The definition of leakage was not control-relative.]
  \emph{Previously:} §2 treated severity as a property of a (mechanism, metric)
  \emph{pair}. \emph{What was wrong:} a severity number is a difference against
  \emph{some} counterfactual, and defensible counterfactuals disagree --- on one
  Mechanism-3 cell by $1.45\times$ across three admissible controls, and by
  $5.9\times$ if the fourth that §4.3 disqualifies is counted.
  (Count as of this entry; E27 below later established that two of those
  three arms are bitwise identical, so the admissible family has \emph{two}
  members and the disqualified control is the third --- which is the count
  the submitted paper states.) \emph{Now:} the
  definition is stated over the (mechanism, metric, \emph{control}) triple,
  every severity number in the paper names the control it is measured against,
  and §4.3 reports the spread across the control family rather than one member
  of it.
\item[E25. §4.3 led with a synthetic estimate its own controls dissolve.]
  \emph{Previously:} Case Study 3 was titled and organized around
  $\Delta_{\text{synth}}=+0.0011$ to $+0.0036$, with the real-feature harness,
  the fidelity extension and the un-averaged readout following as corroboration.
  \emph{What was wrong:} E20's controls remove that band's standing as a
  detectable effect, so the section led with its weakest evidence and buried its
  strongest. \emph{Now:} §4.3 is retitled around $\Delta_{\text{real}}=+0.0077$
  to $+0.0093$ and the fidelity extension's $+0.0250$, leads with those three
  measurements, and presents the primary sweep's non-survival as a finding in
  its own right. \emph{Superseded by E26:} the premise that those three
  measurements escape the asymmetry was false.
\item[E26. The three measurements §4.3 fell back on carry the same asymmetry.]
  \emph{Previously:} the abstract, §4.3, §5 and E25 above stated that the
  real-feature harness (\texttt{code/43}), the fidelity extension
  (\texttt{code/49}) and the un-averaged readout (\texttt{code/55}) are
  untouched by the control asymmetry that dissolved the primary synthetic
  estimate. \emph{What was wrong:} all three set
  \texttt{ES\_HOLD\_FRACTION = 0.15} and use it in the identical in-fold
  carve-out construction as \texttt{code/02d}; \texttt{code/55} additionally
  imports the constant from \texttt{code/02d} and re-reads \texttt{code/02d}'s
  own trained models, so its control arm \emph{is} the asymmetric arm. The claim
  was checkable by grep and survived two review rounds. \emph{Now:}
  \texttt{code/78}, \texttt{code/79} and \texttt{code/83} re-run all three under
  the corrections at both capacities, reporting corrected and original side by
  side. The real-feature estimate moves $+0.0093\to+0.0060$ (capacity 128) and
  $+0.0077\to+0.0067$ (384) under a fold-matched carve-out, both intervals still
  clear of zero, and to $+0.0006$ and $+0.0030$ under the fully corrected
  out-of-fold control. The fidelity extension moves $+0.0250\to+0.0159$, clear
  of zero. The un-averaged readout's gap does not survive correction at the
  per-fold-model readout ($+0.0015$, interval covering zero) and does at the
  direct OOF readout ($+0.0059$); that line is withdrawn from the evidence
  carrying the mechanism. So the confound generalizes beyond the synthetic
  harness --- the more important finding, and the one this round reports as
  such --- while a real-feature severity of $+0.0060$ to $+0.0067$ and a
  full-schedule severity of $+0.0159$ survive it. \emph{Extended by E29:} the
  fully corrected verdicts are capacity-dependent, and the fidelity extension
  now exists at both capacities.
\item[E27. The adaptivity control and the budget-deficit control were one arm.]
  \emph{Previously:} §4.3's control-family table listed CLEAN\_MATCHED\_ADAPTIVE
  and CLEAN\_MATCHED\_85 as two of three admissible controls, and argued that
  LEAKY beats the adaptive control by more than it beats the blind one.
  \emph{What was wrong:} the two arms are bitwise identical by construction ---
  \texttt{code/22}'s adaptive arm keeps its selection run's checkpoint, trained
  on \texttt{tr2\_idx} from torch seed $s$ to epoch $e^\ast$, and
  \texttt{code/67}'s trains on \texttt{tr2\_idx} from the same seed for exactly
  $e^\ast$ epochs. Verified in $15/15$ (seed, fold) pairs; their shipped means
  agree to the last float digit. The argument was an arithmetic restatement of
  the budget deficit. \emph{Now:} withdrawn wherever it appeared --- §4.3's
  control-family table (which marks the duplication), §4.3's
  alternative-explanations subsubsection, §7's limitation (4) and Appendix A.3
  --- and the ``is it adaptivity itself?'' alternative is reported as
  \emph{untested} rather than rejected. The nearest available evidence is
  \texttt{code/77}'s factorial, whose training-depth factor is worth $-0.0002$,
  i.e.\ nothing; but depth is not adaptivity, so it constrains the question
  without settling it, and §7 names the design that would.
\item[E28. \texttt{code/22} and \texttt{code/27} numbers sat beside decoupled ones unmarked.]
  \emph{Previously:} §4.3's control-family table (\texttt{code/22},
  \texttt{code/67}) appeared six lines after the primary table
  (\texttt{code/02d}) with no indication that the two use different
  randomization schemes, which is why the same capacity-128 cell reads
  $+0.0034$ in one and $+0.0036$ in the other. \emph{What was wrong:}
  Appendix A.2 disclosed the scheme difference but defended it on the grounds
  that these scripts are reported ``against their own baselines,'' which is not
  what placing them in the same subsection does. We verified the consequence
  rather than assuming it: \texttt{code/22}'s per-seed arrays are bit-identical
  to the coupled-seed legacy JSON and differ from the shipped decoupled arrays
  by up to $0.16$ per seed. \emph{Now:} the caveat is attached to the table.
\end{description}

\subsection{Round 4: the factorial, the softened magnitude comparison, the resolved calibration, and the generated checklist}

\begin{description}
\item[E29. The primary control's asymmetries are decomposed, not merely removed, and the corrected verdicts are capacity-dependent.]
  \emph{Previously:} E20 removed two asymmetries one at a time
  (\texttt{code/75}), each arm confounding the factors it did not vary, and
  neither touching training depth; E26 then re-ran the real-feature and
  fidelity harnesses at a fold-matched carve-out only, and the fidelity
  extension existed at one capacity. \emph{What was missing:} nothing said
  \emph{which} asymmetry bought the gap, so the result was a null without a
  cause, and a reader could not tell whether the corrected design had simply
  lost power. \emph{Now:} \texttt{code/77} runs the $2\times2\times2$ factorial
  --- selection-set size ($56$ vs.\ $94$) $\times$ selection-run budget (in-fold
  $75$--$85\%$ vs.\ out-of-fold $100\%$) $\times$ training depth (free vs.\ mean
  matched to LEAKY's) --- with all eight arms sharing one pool, one fold
  assignment and one LEAKY reference. The shipped cell gives $+0.0030$ and the
  fully corrected cell $-0.0001$ (BCa $[-0.0021,+0.0021]$, Wilcoxon $p=0.657$);
  averaged over the other factors, selection-set size accounts for $75.5\%$ of
  the movement, selection-run budget $28.1\%$ and training depth $-6.2\%$, with
  an additivity residual of $+0.00008$. Under a rule fixed before the run the
  verdict is \texttt{NOT\_ESTABLISHED}. \texttt{code/78} and \texttt{code/79}
  apply the same factorial to the real-feature harness and the fidelity
  extension, at both capacities --- the fidelity extension's capacity-384 run is
  new, which also retires §7's ``only single-capacity estimate'' limitation. The
  fully corrected verdicts are \texttt{ESTABLISHED} (\texttt{is\_established},
  BCa-interval-only: gap\_mean $>0$ and \texttt{bca\_ci\_95[0]} $>0$) at
  capacity $384$ in both
  ($+0.0030$, $[+0.00003,+0.0071]$; and $+0.0044$, $[+0.0002,+0.0112]$) and
  \texttt{NOT\_ESTABLISHED} at capacity $128$ in both ($+0.0006$,
  $[-0.0037,+0.0062]$; and $+0.0045$, $[-0.0030,+0.0127]$). Wilcoxon disagrees
  with the fidelity extension's capacity-$384$ verdict ($p=0.9155$,
  $44/100$ seeds positive); §4.3 and the abstract state both readings and rest
  the verdict on the interval criterion alone. §4.3 reports the
  fold-matched figures as those harnesses' severities and states the
  capacity dependence rather than averaging it away
  (\texttt{results/mechanism3\_factorial\_selection\_controls.json},
  \texttt{results/real\_feature\_corrected\_selection\_controls.json},
  \texttt{results/fidelity\_extension\_corrected\_selection\_controls.json}).
\item[E30. ``Mechanism-and-harness is the largest of the three axes'' is withdrawn.]
  \emph{Previously:} E18 and E23 concluded, and §5.2, §5.4, the
  abstract, the introduction and the Conclusion asserted, that on
  denominator-sound spans the mechanism-and-harness axis is the largest of the
  three. \emph{What was wrong:} three things, all found by the same review.
  (i)~The operating-point axis's sound maximum and the mechanism axis's minimum
  divide the \emph{same} denominator cell, bit-identical at
  $0.0006530612244898$, so their abutment at $14.266\times$ and $14.278\times$
  is arithmetic rather than an empirical tie. (ii)~\texttt{code/58} matched the
  operating point on the LEAKY arm in all three rows, which is the convention
  that maximizes the ratio; \texttt{code/82} recomputes under each convention
  and control-arm matching gives $13.1$--$28.7\times$ instead of
  $15.1$--$42.0\times$, overlapping the operating-point axis. (iii)~The
  $38.3\times$ upper end rested on the fidelity extension at capacity $128$
  alone; at capacity $384$ (E29) that ratio is $8.0\times$, \emph{below} the
  operating-point axis's sound maximum. \emph{Now:} every occurrence reads ``at
  least comparable to, and plausibly larger than'' the operating-point axis, with
  the shared-denominator, $n=2$ and capacity qualifications stated where the
  comparison is made. The only unqualified ordering fact retained is that
  candidate count is the smallest axis on every convention, which is all the
  retitle of E18 requires
  (\texttt{results/transport\_check\_arm\_matching\_sensitivity.json}).
  \emph{Numerator staleness in (i)--(iii) corrected by E36:} the $14.278\times$,
  $13.1$--$28.7\times$ and $38.3\times$/$8.0\times$ figures reasoned about here
  used shipped numerators later superseded by §4.3's fold-matched-carveout
  correction; (ii) and (iii)'s underlying sensitivity checks have not
  themselves been re-run against the corrected numerators, but the qualitative
  conclusion of this entry is unaffected and, per E36, better supported by the
  correction.
\item[E31. Case Study 4's two calibration sweeps disagreed, and the joint sweep resolves them.]
  \emph{Previously:} the correlation sweep put
  $\Delta_{\text{boot}}$ at $0.50$--$0.55\times$ of the truth and the separation
  sweep's nearest condition at $1.25\times$ --- understating by half versus
  overstating by a quarter --- and §4.4 stated that the high-$\rho$,
  high-separation corner where the real data sits is simulated by neither, so
  the calibration factor is not identified and the joint sweep ``has not been
  run''. \emph{Now:} \texttt{code/81} runs it, crossing both grids under a flat
  layer-mean profile and under the $24$ real cells' own plug-in profiles, and
  reads the ratio along the manifold of $(\rho,\gamma)$ cells reproducing the
  observed $\Delta_{\text{boot}}=+0.0021$ to within $10\%$. Under the real
  profiles the manifold is well identified --- qualifying ratios span
  $0.545$--$0.574\times$, a factor of $1.05$ --- and gives the understating
  reading; under a flat profile it is degenerate over a factor of $3.7$, which
  is why the marginal separation sweep disagreed. \emph{The conclusion is
  favourable rather than adverse}: the calibration-corrected range is confirmed
  at $+0.0029$ to $+0.0040$, now resting on an identified joint manifold rather
  than on one marginal slice quoted as though it were both. What is withdrawn is
  only the earlier claim that the ratios are ``strikingly stable'' across both
  sweeps (\texttt{results/cs4\_joint\_calibration\_sweep.json}).
\item[E32. §4.4 asserted both ``$20$ of $24$ significant below'' and ``no cell significant in either direction''.]
  \emph{Previously:} the two sentences sat
  twenty lines apart in the same subsection. \emph{What was wrong:} a tail
  confusion, not an estimator confusion --- the upper-tail result ($0$ of $24$
  above, true of both estimators) had been restated as though it were the
  two-sided one, and the fields \texttt{n\_two\_sided\_below\_05} and
  \texttt{min\_p\_two\_sided} that \texttt{code/66} computes had never been
  quoted. \emph{Now:} §4.4 states the counts once, labelled by tail and by
  estimator: upper tail $0$ of $24$ for both; two-sided $16$ of $24$ for the
  primary $\Delta_{\text{boot}}$ (minimum $p=0.0010$) and $11$ of $24$ for the
  retired $\Delta_{\text{wc}}$, every one of them below its null. The reading is
  unchanged --- being below a null the shuffle makes maximally inflated is
  evidence about the null, not about the effect --- and what the null
  establishes is still only that no cell is significant \emph{above} it
  (\texttt{results/case\_study\_4\_two\_sided\_null.json}).
\item[E33. Table 8 printed the retired estimator's null statistics under the primary estimator's headings.]
  \emph{Previously:} all $24$ rows showed
  $\Delta_{\text{wc}}$'s null mean and $p$ in columns captioned as
  $\Delta_{\text{boot}}$'s. \emph{Now:} both estimators' null means and
  $p$-values are shown in separate, correctly labelled column groups, and
  \texttt{code/53} asserts the $\Delta_{\text{boot}}$ null column against the
  shipped per-cell JSON on the row where the two differ most visibly.
\item[E34. The control-health ratio was applied only where it disqualified.]
  \emph{Previously:} the
  $(\text{LEAKY}-\text{CM})/(\text{CM}-\text{PLACEBO})$ band of $0.06$--$0.85$
  was used as a pass/fail criterion to disqualify \texttt{code/49}'s
  non-budget-matched control at $1.60$, and was never run on cells the paper
  cites approvingly. \emph{What was wrong:} applied consistently it flags three
  of \texttt{code/47}'s five Sweep C cells --- and four of five against the
  band's measured floor of $0.0631$ --- including the AUROC$_0=0.95$ cell at
  $0.047$ that is the \emph{denominator} of both §5.4 transport ratios; and the
  ratio falls monotonically with the operating point ($0.910$ at $0.70$ to
  $0.029$ at $0.985$), so it is confounded with the operating point and cannot
  discriminate a degraded control from a well-behaved one measured elsewhere.
  \emph{Now:} the diagnostic is retired as a general control-health criterion
  and reported as a descriptive, operating-point-dependent within-harness
  quantity, computed and shown wherever quoted. \texttt{code/49}'s control is
  still disqualified, on the unconfounded ground that it trains on $85\%$ of
  what its comparator trains on.
\item[E35. \texttt{draft/leakage\_checklist.md} had gone stale three rounds running, and is now generated rather than maintained.]
  \emph{Previously:} the
  checklist was hand-edited, and each of the last three rounds left it asserting
  at least one number or claim the main text had already corrected.
  \emph{Now:} \texttt{code/80} generates it from a template
  (\texttt{leakage\_checklist.md.in}) plus facts read out of the shipped result
  JSONs, and \texttt{code/53} runs \texttt{code/80 --check} as a sync assertion,
  so a stale checklist fails the build in the same way a stale
  \texttt{paper\_draft.md} already did. Two related repairs shipped with it:
  \texttt{code/58}'s docstring still quoted ``$+0.0338$'', whose arithmetic
  reproduces the long-retracted ``$52\times$''; and the retracted-string guard
  was made insensitive to whitespace and LaTeX/markdown emphasis delimiters,
  which immediately caught a live bug --- \texttt{README.md} still asserted the
  retracted ``governed not by \emph{which} mechanism'' claim in markdown
  emphasis, a form the old exact-string guard had missed for two rounds.
\item[E36. The magnitude triangle's mechanism-and-harness numerators were stale, sourced from superseded shipped gaps rather than this paper's own currently-reported corrected severities.]
  \emph{Previously:} \texttt{code/72} read the real-feature and fidelity-extension
  numerators for this axis from \texttt{statistical\_rigor\_retrofit.json} and
  \texttt{mechanism3\_fidelity\_extension.json} directly --- the shipped,
  uncorrected gaps ($+0.0093$ and $+0.0250$) --- even after §4.3's fold-matched-carveout
  correction (E26, E29) had superseded them with $+0.0060$ and $+0.0159$. This
  put the axis's denominator-sound range at $14.3$--$38.3\times$ in Table
  \ref{tab:magnitude-triangle}, §5.2, the abstract, the introduction, §5.4,
  the Conclusion and §6.1's checklist, all quoting a number this paper's own
  §4.3 no longer asserted as the severity of either harness. \emph{Now:}
  \texttt{code/72} sources both numerators from
  \texttt{results/real\_feature\_corrected\_selection\_controls.json} and
  \texttt{results/fidelity\_extension\_corrected\_selection\_controls.json}'s
  \texttt{headline\_comparison["128"]["fold\_matched\_in\_fold\_gap"]} fields,
  giving a corrected range of $9.2$--$24.3\times$ (real-feature $9.241\times$,
  fidelity extension $24.276\times$, against the same bit-identical reference
  denominator $0.0006530612244898$). One consequence disclosed rather than
  hidden: under the superseded numerator the mechanism axis's minimum
  ($14.278\times$) abutted the operating-point axis's sound maximum
  ($14.266\times$) almost exactly, which §5.2 already flagged as arithmetic
  coincidence rather than evidence (both divide the same denominator cell);
  under the corrected numerator that near-tie is gone and the two axes overlap
  substantially instead ($9.2\times$ sits clearly below $14.266\times$). The
  qualitative conclusion --- mechanism-and-harness is at least comparable to,
  and plausibly larger than, the operating-point axis, and is not claimed to be
  strictly the largest of the three --- is unchanged by this correction and, if
  anything, better supported by it, since the corrected ranges now overlap
  directly rather than only abutting at a shared denominator. Two subsidiary
  sensitivity checks that also source these harnesses' gaps
  (\texttt{code/82}'s control/placebo-arm-matching table, $13.1$--$28.7\times$
  and $13.0$--$24.0\times$; and the probit-space interpolation reported
  alongside the primary $15.1$--$42.0\times$) have not themselves been re-run
  against the corrected numerators in this round and are flagged as stale
  wherever they are quoted, rather than silently left looking current; the
  primary corrected comparison does not depend on them. \texttt{code/53} is
  updated to check comparability-with-overlap rather than strict-largest
  dominance, and \texttt{results/magnitude\_triangle.json}'s stored
  \texttt{denominator\_sound\_comparison.statement} and \texttt{verdict.statement}
  fields, which previously asserted the mechanism-and-harness axis is the
  ``LARGEST'' of the three verbatim, are regenerated to match.
\end{description}

\subsection{Round 5: the follow-up items from the third Frontier AI Scientific Review Board pass --- Mechanism 2's second construction, and the ES\_HOLD\_FRACTION sensitivity sweep}

\begin{description}
\item[E37. Mechanism 2's harness (§5.9, \texttt{code/97}) had never been independently reconstructed a second way; three reviewers across two rounds named this the single most important open question in the head-to-head comparison.]
  \emph{Previously:} §5.9 fit a per-mechanism-intercept model across all five
  leakage mechanisms and reported a wide intercept spread ($304\times$
  multiplicative range), with Mechanism 2's own intercept ($a_2=-3.389$)
  resting on one synthetic harness --- $K$ independent isotropic-Gaussian
  candidate ``layers,'' built solely for that section --- never checked
  against an alternative construction, unlike Mechanisms 1, 3, 4 and 5's
  harnesses, each of which reuses or adapts machinery already through this
  paper's own multi-round adversarial correction process. The concern: the
  wide intercept spread could reflect real mechanism-identity differences, or
  it could simply reflect that Mechanism 2's (and Mechanism 3's) harnesses are
  less mature than the other three. \emph{What was done:} \texttt{code/99}
  builds a genuinely different second construction (``construction B''),
  changing exactly one thing --- independence across the $K$ candidates is
  replaced with a one-factor shared-latent-factor model,
  $X_k=\text{class\_sep}/2\cdot\text{sign}+\sqrt{\rho}\,L+\sqrt{1-\rho}\,E_k$,
  $\rho=0.5$ disclosed and fixed before the run, so adjacent candidates carry
  correlated signal (the property real transformer layers have) while each
  candidate's own marginal calibration to the target AUROC$_0$ is unchanged
  from construction A. Run at the identical $(K,\text{AUROC}_0)$ grid and
  $N_{\text{SEEDS}}=100$, with a stability rule pre-registered in
  \texttt{code/99}'s own module docstring before the run: STABLE requires both
  an own-fit intercept within one order of magnitude of construction A's, and
  a positive, significant per-cell correlation across the $15$ matched cells.
  \emph{Result:} construction B's own fitted intercept is $a_B=-3.349$ against
  construction A's $a_2=-3.389$ ($\Delta=0.040$, well inside the
  $\ln10\approx2.303$ threshold); the two constructions' per-cell gaps
  correlate at Pearson $r=0.982$ ($p=9.3\times10^{-11}$); and substituting
  construction B for construction A inside the full four-mechanism pooled
  comparison shifts Mechanism 2's fitted intercept by only $+0.035$ (to
  $-3.353$) while leaving the qualitative result intact ($R^2$
  $0.851\to0.997$, $F(3,42)=658.7$, $p=1.1\times10^{-16}$, intercept spread
  $271\times$ against the original $304\times$). By the pre-registered rule,
  the intercept is \textbf{STABLE}. \emph{What this does and does not settle:}
  this is one robustness check on one of the two least-mature harnesses, run
  in good faith without a preferred outcome --- an unstable intercept would
  have been reported as directly confirming the harness-maturity confound,
  which is why this is being reported at all regardless of which way it came
  out. It leaves Mechanism 3's own underpowering ($N_{\text{SEEDS}}=15$)
  untouched, tests one alternative construction (correlated candidates) rather
  than a sweep over the correlation parameter or a change of estimator family,
  and a stable intercept under this specific perturbation is evidence against,
  not proof against, the harness-maturity confound the reviewers named. §5.9,
  §7 and the Conclusion are updated to state this finding and its limits.

\item[E38. \texttt{ES\_HOLD\_FRACTION}=0.15 (flagged above as the single
  constant generating the most disclosed confounds in this paper) had never
  been swept inside the decisive $2\times2\times2$ Mechanism-3 factorial
  (\texttt{code/77}) itself; \texttt{code/75}'s own earlier
  $\{0.15,0.20,0.25\}$ sweep predates the factorial and does not report how
  its three-way decomposition moves.]
  \emph{Previously:} the note above (``Not derived. What each one exposes'')
  flagged that $0.25$, the exact fold match, ``would match the fold exactly at
  the cost of a larger budget deficit; we did not run it,'' naming this an
  open robustness gap. \emph{What was done:} \texttt{code/100} re-runs
  \texttt{code/77}'s identical factorial (same pool/fold/init-seed scheme,
  capacity $128$, AUROC$_0=0.80$, $N_{\text{SEEDS}}=100$) at
  $\texttt{ES\_HOLD\_FRACTION}\in\{0.05,0.10,0.15,0.20,0.25\}$ as originally
  planned, holding every other factor fixed. $F_{\text{SMALL}}=0.30$ was
  expected to be infeasible (the small/out-of-fold arm's required selection-set
  size, $\approx112$, exceeds the shared out-of-fold pool's $\approx94$-point
  size, fixed by \texttt{OOF\_HOLD\_FRACTION} independent of
  $F_{\text{SMALL}}$) and was excluded before the run on that basis.
  \emph{What was NOT expected:} $F_{\text{SMALL}}=0.25$ --- the exact value
  Appendix A's note above says was never run --- \emph{also} turns out to be
  infeasible under this same shared pool: its required size ($\approx93$)
  very nearly exhausts the $\approx94$-point pool, leaving a $1$-sample
  stratified complement that \texttt{scikit-learn}'s \texttt{train\_test\_split}
  refuses (\texttt{ValueError: test\_size=1 < number of classes=2}), confirmed
  to fail identically at every seed attempted, not a seed-specific fluke. The
  sweep grid actually run is therefore $\{0.05,0.10,0.15,0.20\}$, with both
  $0.25$ and $0.30$ reported as infeasible rather than silently dropped.
  \emph{Result:} the fully corrected cell does not move ($-0.0001$ at every
  value tested, exactly as it must, since it never depends on this constant).
  The shipped gap and total movement shrink together as
  \texttt{ES\_HOLD\_FRACTION} grows ($+0.0104,+0.0057,+0.0030,+0.0024$ for
  $0.05,0.10,0.15,0.20$) --- mechanical, since a larger
  \texttt{ES\_HOLD\_FRACTION} makes the ``small'' selection set less different
  from the fold-matched one. The selection-set-size share is
  \textbf{not stable} across this range: $74.7\%,69.0\%,75.5\%$ (the shipped
  value) and $38.5\%$ at $0.05,0.10,0.15,0.20$ respectively, with the
  selection-run-budget share rising correspondingly
  ($16.6\%,31.4\%,28.1\%,62.5\%$) to fill the difference; this is consistent
  with the ``small'' and ``fold\_matched'' levels converging as
  \texttt{ES\_HOLD\_FRACTION} approaches the fold match at $0.25$, which makes
  the selection-set-size main effect's own share of an also-shrinking total
  movement the least numerically stable of the three quantities in this
  decomposition. \emph{What this does and does not settle:} the qualitative
  NOT-established verdict for Mechanism 3's fold-reuse-specific severity is
  unchanged at every value tested, closing the open robustness gap this
  appendix flagged in that respect. The specific $75.5\%$/$28.1\%$/$-6.2\%$
  decomposition, however, is \emph{not} shown to be robust to this
  undisclosed choice --- it holds up over $0.05$--$0.15$ but the
  selection-set-size share very nearly halves by $0.20$, and neither $0.25$
  nor $0.30$ could be tested at all under this factorial's current shared-pool
  construction, which a wider sweep would require enlarging the pool to
  address. §4.3 and this appendix are updated to state this finding plainly
  rather than let the shipped $75.5\%$ read as more invariant than it is.
\end{description}

```

---

---

## Addendum: table rows removed from the compiled paper in the 34-page compression pass

A second, later compression pass (79 printed pages to 34) further reduced the
two compact correction tables that had replaced these appendices. `main.tex`'s
Appendix A now carries **six** of the sixteen Case Study 3 corrections --- those
that changed a number the paper still reports (issues 3, 8, 9, 11, 13, 15) ---
and its correction log for Case Studies 2/4, Mechanism 5 and the severity
surface carries **fourteen** of the thirty-six rows, under the same E-numbering
used here. The secondary-corrections table (robustness checks beyond the
sixteen) was dropped from the paper entirely. Every removed row is reproduced
verbatim below, and the full narrative behind each is in the corresponding
section above.

### Appendix A, Table 10 — the ten rows removed

```latex
1 & Uncalibrated sweep saturated at AUROC$=1.0$. & Calibrate via $\text{AUROC}=\Phi(\sqrt{J/2})$. & Usable operating point. \\

2 & Calibration formula was inverted. & Corrected inversion, $J=2\Phi^{-1}(0.80)^2=1.417$. & Verified $\Phi(\sqrt{J/2})=0.800$. \\

4 & Real-feature test's architecture didn't match the synthetic sweep's (single-fold MLP vs.\ intended 5-fold OOF+meta-learner). & Rerun with the actual architecture (code/25). & $+0.0002$ ($p=0.56$, cap.\ 128), n.s. \\

5 & ``Corrected architecture'' still ran at a ceiling operating point (AUROC$\approx0.985$). & Rescale real features to the $0.80$ target (code/31). & $+0.0014$/$+0.0010$ (cap.\ 128/384), underpowered. \\

6 & Rescaling factor $\alpha$ came from a biased estimator ($J_{\text{real}}$ unstable at $d\approx n$; $J_{\text{perm}}=2.31>J_{\text{target}}=1.42$). & Empirical, CV-calibrated bisection on achieved AUROC (code/33). & Converges to $\alpha=0.2031$, hits AUROC $0.795\pm0.013$. \\

7 & A permutation-null number ($0.512\pm0.044$) was misattributed to this check. & Reconciled: that number belongs to a different check. & This check's real value: $0.120\pm0.010$. \\

10 & Checkpoint-selection-only harness may understate the real trainer's coupled LR-scheduler/early-stopping mechanic. Two invalid fix attempts caught before reporting (one let the control mirror LEAKY by construction; one dropped the scheduler entirely). & Keep the model trained with the real scheduler on a disjoint carve-out as the control directly. & LEAKY\_PLUS\_LRSCHED beats the (later superseded) es-fed control by $+0.0111$ ($p=4.3\times10^{-9}$). Training-depth confound disclosed: mean epoch $29.14$ (LEAKY) vs.\ $19.75$ (control) vs.\ $7.95$ (placebo), $+47.6\%$, $p=7.2\times10^{-13}$. \\

12 & The fidelity extension (code/49) used one constant, $3$, for both the LR-scheduler patience and the early-stopping break; the audited repo uses $3$ and $15$ respectively. & Separate counters. Factorial $2\times2$ ablation (calibration $\times$ patience), rerun under issue 14's fidelity port. & Cells: $+0.0081/+0.0052$ (superseded calib., patience 3/15), $+0.0342/\mathbf{+0.0338}$ (label-free calib., patience 3/15). Calibration, not patience, moves the number; a previously-reported sign-flip interaction did not survive the fidelity port and is retracted. \\

14 & The ``fidelity extension'' silently differed from the audited trainer on optimizer/data-pipeline settings ($10\times$ LR mismatch, full-batch vs.\ mini-batch, no warmup, wrong scalers, wrong min-LR/clip/epoch-count). & All ported and verified against the pinned commit (LR $2\times10^{-4}$, AdamW, batch $28$, warmup $5$, clip $0.5$, min-LR $10^{-7}$, $45$ epochs, RobustScaler/StandardScaler). & Gap moved $+0.0221\to+0.0338$ (later $+0.0250$, item 15); ratio $2.4\times\to3.6\times$ (later $2.7\times$); LEAKY operating point $0.9176\to0.9424$. Cost: item 12's sign-flip interaction did not survive and is retracted. \\

16 & The adaptivity control (code/22) selected its checkpoint on data inside its own training set (\texttt{es\_idx} carved from, but trained within, \texttt{tr\_idx}); kept-checkpoint epoch $39.4$/$45$ vs.\ $20.8$--$21.4$ for honest arms. & Train on \texttt{tr2\_idx}, the complement of \texttt{es\_idx}. & Placebo-relative retention $37.3\%\to\mathbf{94.2\%}$ (cap.\ 128), $25.7\%\to\mathbf{79.9\%}$ (cap.\ 384); LEAKY still beats it, $+0.0049$/$+0.0059$. \textbf{Later withdrawn (E27):} the rebuilt arm is bitwise identical ($15/15$ seed/fold pairs) to code/67's budget-deficit arm --- an arithmetic restatement of the deficit, not adaptivity evidence. The ``is it adaptivity itself'' question is untested, not rejected. \\
```

### Appendix A, Table 11 — secondary corrections and robustness checks (removed in full)

```latex

CLEAN\_MATCHED's epoch count is selected on $85\%$ of the fold but retrained on $100\%$. & Direction not determined a priori (disclosed, not fixed; removing it would change what the control controls for). & Retained and disclosed rather than corrected. \\

\texttt{code/02d}'s randomization: the replicate index $i$ was passed to all three randomization streams (generator seed, split seed, fold seed) --- one stream aliased as three. & Same confound \texttt{code/47}'s decoupled scheme was written to fix, never retrofitted to \texttt{code/02d}. & Fixed (\texttt{seeds\_for()}). Band moves $+0.0009$--$+0.0034\to+0.0011$--$+0.0036$ (stable); but capacity 48 loses significance ($p=0.015\to0.171$) and capacity 16 gains it ($p=0.309\to0.019$) --- only capacity 128 survives Holm--Bonferroni either way. \texttt{code/47}'s cap-128 cell was called ``independently-written''; withdrawn --- it is a determinism check (shared implementation), not independent confirmation. \\

Is the effect adaptive selection itself, not fold reuse? First adaptivity control (\texttt{CLEAN\_MATCHED\_ADAPTIVE}) trained on the FULL fold while selecting on a carved-out subset of it --- in-sample selection, not held-out. & Kept-checkpoint epoch $39.4$/$45$ (cap.\ 128) vs.\ $20.8$--$21.4$ honest arms: it measured convergence. Retracted figures: LEAKY beating it by $+0.0198$/$+0.0143$, retention $37.3\%$/$25.7\%$. & Rebuilt on \texttt{tr2\_idx} (disjoint): retention $94.2\%$/$79.9\%$; LEAKY beats it by $+0.0049$ ($p=0.0008$)/$+0.0059$ ($p=0.0012$). \textbf{Both readings then withdrawn}: bitwise identical to \texttt{code/67}'s budget-deficit arm. Deficit measured directly: $+0.0015$ (cap.\ 128, $p=0.29$)/$+0.0031$ (cap.\ 384, $p=0.005$) --- at cap.\ 384 this exceeds the whole LEAKY$-$CLEAN\_MATCHED gap ($+0.0027$). \\

Does downstream fold-model activation averaging (5-way, before scoring) attenuate the measured gap? & Averaged gaps ($+0.0019$/$+0.0011$/$+0.0036$/$+0.0024$, caps 16/48/128/384) are $2.3$--$5.8\times$ smaller than un-averaged ($+0.0073$/$+0.0066$/$+0.0084$/$+0.0084$, code/55); significant at all 4 capacities un-averaged vs.\ 2/4 averaged. Cross-fold-model scoring sits at chance ($0.484$--$0.522$), confirming fold-model feature spaces share no information. & Alternative explanation SUPPORTED. §4.3's headline keeps the averaged number (what the audited pipeline actually reports); the un-averaged number is reported as the mechanism's severity at its source. \\

Second generative process: real Mistral-7B/HaluEval covariance shape (code/27, $d=414$) vs.\ isotropic ($d=64$) --- two confounds (dimension, achieved AUROC $0.672$--$0.681$ vs.\ $0.753$--$0.765$). & LEAKY vs.\ CLEAN\_MATCHED $+0.0082$/$+0.0099$ (caps 128/384) vs.\ isotropic's $+0.0036$/$+0.0024$; weakens at caps 16/48. Pooled 8-test Holm--Bonferroni family: 2/8 survive (both cap-128 cells) --- up from 0/8 under the pre-seed-decoupling isotropic run. & Evidence against isotropic-covariance-alone explaining the estimate, but weaker/less clean than a simple ``replicates and is larger'' summary; verdict improving when its own instrument was corrected is flagged, not treated as vindication. \\

Discriminating experiment: fixed $d=64$, vary only the within-class eigenvalue profile ($\beta\in\{0,0.5,1,2\}$, plus \texttt{real\_top64}), code/56. & Cap.\ 128: gap rises monotonically with anisotropy, $+0.0015\to+0.0104$ ($\approx7\times$ span). But anisotropy and achieved operating point are perfectly rank-confounded (Spearman $-1.00$); LEAKY falls $0.7523\to0.5826$ across the same cells. Ordering not stable across capacity (cap.\ 384 Spearman $-0.10$). & Cannot distinguish ``anisotropy raises severity'' from ``anisotropy lowers the operating point, which raises severity'' (§5's relationship again). Pooled 10-cell Spearman $-0.73$ ($p=0.016$). \\

Pre-registered decision rule (\texttt{GENUINE\_LEAK\_CONFIRMED}/\texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK}/\texttt{MIXED}) applied to every version run; an earlier claim called the confound branch structurally unreachable. & Wrong: all three verdicts are reachable, and none is stable --- the same mechanism yields all three depending on generative process, calibration, and capacity. The verdict moved toward this paper's own hypothesis exactly when the calibration was corrected. & Reported as unstable; this is why §4.3 rests on effect sizes with intervals, not the rule's label. \\

Does severity scale as extreme-value theory (EVT) predicts? $K$-sweep (code/47), gap rises $+0.0000\to+0.0052$ ($K{=}1\to225$), every $K\ge3$ cell significant. & \emph{Corr.\ 1:} an earlier version had the same $4\times$-$J$ bug as issue 11; fixed identically. \emph{Corr.\ 2:} the original fit substituted a CONSTANT $\sigma$ (not per-cell), converting a 1-parameter EVT model into ``gap $\propto\sqrt{2\ln K}$''; per-cell $\sigma$ actually varies $2.66\times$ and anti-correlates with the gap ($-0.84$, wrong sign). Refit per-cell: $R^2=-0.399$. \emph{Corr.\ 3 (retracts Corr.\ 2's ``falsified'' verdict):} $\sigma$ in code/47 is one training trajectory's per-epoch dispersion, not the sampling-noise SD of $K$ exchangeable estimates EVT requires --- wrong quantity, not a falsification. & Verdict downgraded to \textbf{``untestable in this harness as instrumented,''} not falsified. The empirical law $a+b\ln K$ never references $\sigma$ and is unaffected. \\

EVT, continued: degeneracy gate (bitwise \texttt{state\_dict} identity) was wired into code/57's joint grid but never enabled at code/47's own call sites (\emph{Corr.\ 4}). & Enabled everywhere (no RNG/training-path effect; all prior numbers reproduce bit-for-bit). $K{=}3/5/10$ cells are $97.3\%/90.0\%/64.5\%$ bitwise-identical LEAKY/CLEAN\_MATCHED (degenerate); $K\ge15$ are not ($\le35.5\%$). & Refit on 6 non-degenerate cells: $R^2=0.939$ (slope moves only $6\%$ and survives), but apparent dynamic range shrinks $6.10\times\to2.31\times$ --- $\approx2/3$ of the original headline's range came from cells where the two arms were literally the same model. Both fits shipped side by side. \\

EVT, continued: Sweep B was labeled an ``$n_{\text{val}}$ sweep'' (\emph{Corr.\ 5}). & It varies \texttt{N\_SAMPLES}, moving train/test/val proportionally --- not an $n_{\text{val}}$ isolation at all. & Relabeled a sample-size sweep (numbers unchanged: $+0.0026$/$+0.0036$/$+0.0005$). True isolated $n_{\text{val}}$ sweep is Sweep D (fixed \texttt{N\_SAMPLES}, vary $K_{\text{CV}}$): gaps $+0.0087\to+0.0005$ as $n_{\text{val}}$ falls $280\to56$; $K_{\text{CV}}{=}5$ cell reproduces Sweep B's $N{=}700$ cell exactly. Earlier ``directionally consistent with winner's-curse'' framing of the $\sigma$ pattern withdrawn (same misspecification as Corr.\ 3). \\

Operating-point sweep ($\text{AUROC}_0\in\{0.70,\ldots,0.985\}$): monotone decline, but significance is not (0.70/0.80/0.95 significant; 0.90/0.985 not). & 0.985's non-significance is consistent with the ceiling-effect ranking noise issue 5 documents, not the mechanism vanishing. & Promoted to a headline finding as a within-harness modifier only; does not transport to the real-feature harnesses at matched operating points ($9.2$--$24.3\times$ larger there per §5.4); licenses only that operating points be matched before severities are compared. \\
```

### Appendix E — the twenty-two rows removed

```latex
E2 & Severity band quoted as ``$0.000$ to $0.034$ AUROC.'' &
Was the retired diagnostic's per-cell range. &
Now: $0.000$ to $0.026$; largest current cell $+0.0122$ (qwen2.5-7b/fever/linear). \\

E4 & §5.4 called the transport reference cell ($+0.00065$) ``indistinguishable from zero'' while also dividing by it. &
Contradiction. &
Cell's own gap excludes zero (Wilcoxon $p=0.037$, CI $[+0.00007,+0.00121]$); §5.4 now says so; contrasted with E3's genuinely-zero denominator. \\

E5 & §4.3 quoted the FP16-vs-AWQ control as ``identical AUROC ($0.9600$ both).'' &
Matched no shipped file. &
Now: $0.9460$ (FP16) vs $0.9392$ (AWQ), gap $+0.0068$, CI $[-0.061,+0.080]$, $p=0.148$ --- consistent with no confound, not a demonstration ruling one out. \\

E6 & §4.2 asserted GUARDIAN halves ``separable at AUROC $0.734$--$0.776$,'' all-data argmax L19. &
Neither number came from any computation; hardcoded prose. &
Now computed (code/61): OOF AUROC max $0.7885$ at L25, mean $0.7259$; all-data argmax L21 ($0.7829$). Also found: an $8.0$pp base-rate difference between halves, on its own enough to make AUROCs non-comparable; a base-rate-matched control does not rescue the split ($\Delta_{\text{sel}}=-0.0050$, SD $0.0058$). \\

E10 & No revision addressed variance compression as an alternative account of the operating-point decline (the paper's largest effect). &
Untested alternative. &
code/69: 3 stabilized coordinates (log, probit, rank). Decline is real (mean falls $2.7\times$ faster than SD) but most of the raw decline's SIZE is coordinate effect ($0.51$/$0.70$/$0.25$ of it survives per coordinate); strict monotonicity survives in none. Second reason (with E3) the paper quotes no multiplier here. \\

E11 & --- &
Candidate-count exponent disclosed unstable: $b=0.09$ to $0.21$ under column deletion, $b=0.29$ under OLS vs.\ Gauss-Newton. &
Operating-point coefficient (what claims rest on) stable under both. \\

E12 & CS4 permutation null quoted only where favourable. &
--- &
code/66, exact enumeration ($3^3=27$): $0$/$24$ cells sig.\ above own null, $20$/$24$ sig.\ below --- expected of a null built to maximize the curse for stable-argmax cells. \\

E13 & --- &
Adaptivity control's selection carve-out is smaller than LEAKY's fold and costs $\approx15\%$ training data; both bias against the control. &
§4.3's $+0.0049$/$+0.0059$ restated as upper bounds, not point estimates, on the fold-reuse effect. \\

E14 & Two literatures were missing: adaptive data analysis / post-selection inference, and the hidden-state hallucination-probing subfield audited. &
--- &
Both now cited in §2, incl.\ HaloScope (Du et al.\ 2024), whose code, read directly, is correct --- a positive data point reported as such. \\

E15 & E1 promoted $\Delta_{\text{boot}}$ on the ground it is ``not sign-constrained ($2$/$24$ cells negative).'' &
Wrong: Jensen's inequality gives $\Delta_{\text{boot}}\ge0$ for ANY data (Thm.~\ref{thm:boot-nonneg}); the ``$2$ negative'' cells were Monte-Carlo noise --- exact enumeration ($3^3=27$) gives $\pm10^{-16}$. &
Point estimate unchanged, $+0.0021$, retained as a magnitude only; Wilcoxon $p=6.0\times10^{-7}$ and all sign-based claims withdrawn. \\

E18 & Title/abstract: severity governed by two axes ``rather than mechanism.'' &
§5.4 already withdraws that comparison; the paper's own three ranges put mechanism-and-harness up to $38.3\times$, at least comparable to operating-point's $48.6$--$68.3\times$. &
Title/abstract now claim only the within-harness result; §5.2 tabulates all three ranges (Table~\ref{tab:magnitude-triangle}). \\

E19 & ``24 independent result files.'' &
A crossed $3\times4\times2$ design sharing three seeds, not 24 independent files; dataset alone explains $32\%$ of between-cell variance. &
A dataset-clustered interval ($20\%$ wider) now reported alongside the i.i.d.\ one. \\

E21 & Mechanism 5's arms were never disclosed as size-mismatched (LEAKY selects on $n=140$, HONEST on $n=112$). &
$1.25\times$ asymmetry, undisclosed. &
code/74: size-matched arm gives F1 $+0.0214$ vs.\ shipped $+0.0225$ ($p=0.67$); asymmetry alone accounts for $\approx2\%$ of the gap (max $8\%$ across operating points). Finding stands. \\

E23 & Magnitude triangle's raw ratios: op.-point $48.6$--$68.3\times$, candidate-count $1.6$--$7.1\times$. &
Same defect as the withdrawn ``$48.6\times$'': zero-reaching denominators. &
code/72 restricts to denominator-sound cells: $2.2$--$14.3\times$ (op.\ point), $14.3$--$38.3\times$ (mechanism), $1.8$--$4.2\times$ (candidates). Superseded by E36. \\

E24 & §2 treated severity as a property of a (mechanism, metric) pair. &
A severity number is relative to SOME counterfactual; one Mechanism-3 cell varies $1.45\times$ (or $5.9\times$ incl.\ a disqualified control) across admissible controls. &
Definition restated over (mechanism, metric, \emph{control}); every number now names its control. \\

E25 & §4.3 led with the synthetic estimate $+0.0011$ to $+0.0036$, real-feature/fidelity/un-averaged as corroboration. &
E20's controls dissolve that band --- section led with its weakest evidence. &
Retitled around $\Delta_{\text{real}}=+0.0077$--$+0.0093$ and fidelity's $+0.0250$. \emph{Superseded by E26}: those three shared the same asymmetry. \\

E28 & §4.3's control-family table (code/22, code/67) sat beside the primary table (code/02d) with no note that they use different randomization schemes. &
Same cap-128 cell reads $+0.0034$ vs.\ $+0.0036$ across the two. &
Verified (not merely disclosed): code/22 per-seed arrays are bit-identical to the coupled-seed legacy JSON, differing from decoupled arrays by up to $0.16$/seed; caveat now attached directly to the table. \\

E30 & E18/E23 concluded mechanism-and-harness is the LARGEST of the three axes. &
(i) op.-point sound max.\ and mechanism-axis min.\ share the same denominator cell ($0.0006530612244898$) --- arithmetic abutment, not an empirical tie. (ii) code/58 matched op.\ point on the LEAKY arm only, maximizing the ratio; control-arm matching gives $13.1$--$28.7\times$, overlapping op.\ point. (iii) The $38.3\times$ upper end was cap.\,128 only; at cap.\,384 (E29) it is $8.0\times$, below op.-point's sound max. &
Now: ``at least comparable to, and plausibly larger than'' everywhere. Numerator staleness in (i)--(iii) corrected by E36. \\

E32 & §4.4 stated both ``$20$/$24$ significant below'' and ``no cell significant in either direction.'' &
Tail confusion --- the upper-tail result ($0$/$24$ above) was restated as the two-sided one. &
Now stated once, by tail and estimator: two-sided $16$/$24$ ($\Delta_{\text{boot}}$, min $p=0.0010$) and $11$/$24$ ($\Delta_{\text{wc}}$); upper-tail $0$/$24$ for both. \\

E33 & Table 8 printed $\Delta_{\text{wc}}$'s null stats in columns captioned as $\Delta_{\text{boot}}$'s. &
Mislabelled columns, all 24 rows. &
Both estimators' nulls now shown in separately-labelled column groups; code/53 asserts the row where they differ most. \\

E34 & The $(\text{LEAKY}-\text{CM})/(\text{CM}-\text{PLACEBO})=0.06$--$0.85$ band was used only to disqualify code/49's control (at $1.60$). &
Applied consistently it flags $3$/$5$ (or $4$/$5$ vs.\ the measured floor $0.0631$) of code/47's own Sweep C cells, incl.\ the $\text{AUROC}_0=0.95$ denominator cell (ratio $0.047$); the ratio is confounded with operating point ($0.910\to0.029$). &
Retired as a general health criterion; reported descriptively wherever quoted. code/49's control is still disqualified, on the unconfounded $85\%$-vs-$100\%$ budget ground. \\

E35 & \texttt{leakage\_checklist.md} was hand-edited and went stale three rounds running. &
--- &
code/80 now generates it from a template + shipped JSONs; code/53 runs \texttt{code/80 --check} as a build gate. Caught live bugs: a stale ``$+0.0338$'' in code/58's docstring, and README.md still asserting a retracted claim in markdown emphasis (missed twice by the old exact-string guard; guard is now formatting-insensitive). \\
```

---

## Addendum: a presentation-only rewrite pass separating narration from result

This entry documents a pass over `main.tex` that changed **presentation only**:
no number, statistical test result, or conclusion's truth value was touched.
The motivation was three independent adversarial reviews that each scored the
paper's Clarity at 4/10, specifically flagging the risk of a reviewer
disengaging before reaching the material that substantiates the paper's own
hedges, because the main-flow prose narrated the paper's own editorial/
discovery process inline (e.g. "an earlier round found X," "a reviewer-driven
pass discovered Y and fixed it") in addition to the dedicated Appendix A and
Appendix D, which already exist to hold exactly that history.

**What changed.** Roughly a dozen main-flow passages across the Abstract,
Introduction Contributions list, §4.3, §4.5, §5.3, §5.5, §6.3, §7 Limitations
and the Conclusion were rewritten to state the current, final result directly,
with a short pointer to the Appendix A/D item (or this file) that already
carries the "how we got here" narrative, instead of re-narrating it inline.
Examples: the abstract's "further round of checks, run against a worry this
paper's own correction record raises" paragraph now opens "Additional
robustness checks" and states the calibrated-estimator result directly; the
§5.3 exchangeable-candidate $K$-sweep now leads with the current,
properly-powered ($N_{\text{SEEDS}}=200$) null and points to item E38 above
for the earlier underpowered pass, rather than narrating the escalation from
$N=30$ to $N=200$ inline; the §4.5 calibration-guardrail sentence and the
Introduction's Contributions item for it now point to Appendix A issue 11
instead of narrating "built after two of this paper's own generators were
found to have silently mislabelled their operating points."

**What did not change.** Every legitimate scientific hedge was left untouched,
including the ones this pass could most easily have been mistaken for
narration: §5.9's harness-maturity discussion (Mechanism 2 newly built,
Mechanism 3 underpowered at $N_{\text{SEEDS}}=15$), Limitation (6)'s
researcher-degrees-of-freedom admission, Limitation (8)'s residual-asymmetry
and fidelity-gradient discussions, and the Broader Impact section's argument
that the checklist is not a guarantee, evidenced by the paper's own imperfect
self-audit. These are honest statements about the *current* evidentiary
status of a claim, not narration of drafting history, and none of their
language was softened.

**Verification.** `code/53_verify_paper_numbers.py` ran after each block of
edits and stayed at 0 failures throughout except for the expected, transient
`paper_draft.md` drift between edits and the final sync (558/558 before this
pass; 558/558 after `code/52_sync_paper_draft_md.py` was re-run). Two-pass
`pdflatex` compiled clean (0 `^! ` errors, 0 undefined references) both before
and after, and the page count is unchanged at 53 pages. An identity-leak grep for this
paper's author-name variants over every file touched found nothing, as
before this pass.

*End of full correction-history record. See `draft/latex/main.tex` Appendix A
and Appendix D for the compact tables that replaced this material in the
compiled PDF, `EXTENDED_TECHNICAL_DETAIL.md` for technical detail relocated
out of the main text in the same pass, and `code/53_verify_paper_numbers.py`
for the machine-checked provenance of every number named above.*

---

## Addendum 2: an editorial compression pass (merge and de-duplicate, not narration removal)

This entry documents a second pass over `main.tex`, run after the
presentation-only pass above had already removed the in-text process
narration it targeted and found little more of it left. This pass had a
different, explicit mandate: achieve real page compression toward a
30–35-page target via merging, tightening and de-duplication — not by
finding additional narration (there was very little left) and not by
deleting any finding, statistic, hedge, or the distinct severity vocabulary
($\Delta_{\text{sel}}$, $\Delta_{\text{real}}$, $\Delta_{\text{synth}}$,
$\Delta_{\text{boot}}$, $\Delta_{\text{wc}}$, $\Delta_{\text{boot}}^{\text{std}}$,
kept fully distinct throughout).

**What changed.** The paper's own structure meant most of its remaining bulk
was *vertical* redundancy — the same 6–8 headline numbers and hedges restated
at three or four levels of the document (Abstract → Introduction →
§4/§5 body → Limitations/Conclusion) in near-identical wording — rather than
off-topic content. Edits targeted exactly that:

- **§9 Conclusion** and **§1 Introduction**'s narrative paragraphs were
  tightened to point at the Abstract/§4/§5 numbers rather than restate them
  in full precision a second or third time; no number or conclusion in either
  section was changed, only shortened where it duplicated a fuller statement
  elsewhere.
- **§7 Limitations**: each of the ~14 numbered items was kept as a distinct
  hedge (none deleted, none softened), but numeric restatements already given
  in full in §4.3/§4.4/§5 were replaced with a pointer to where the number
  already lives; three closely related fidelity-extension sub-items
  ((4c)/(4c-i)/(4c-ii)) were merged into one paragraph without losing any of
  the three claims.
- **§8 Broader Impact**: tightened; no claim removed.
- **§4.3 (Case Study 3)**: the real-feature-harness, fidelity-extension,
  second-model-family, capacity-sweep, alternative-explanations and
  oracle-baseline passages were tightened throughout (shorter transitions,
  fewer restated caveats, dash-asides converted to direct clauses), with
  every BCa interval, $p$-value, and percentage retained verbatim. The
  ES\_HOLD\_FRACTION sensitivity paragraph and the "bottom line" paragraph
  were shortened by pointing at Table~\ref{tab:m3-factorial}/\ref{tab:m3-primary}/\ref{tab:m3-real}
  instead of re-quoting their contents. (A true table-merge of
  `tab:m3-primary` into `tab:m3-real`, considered as the single highest-value
  cut per the compression brief, was attempted only on paper: the two tables
  are not the same shape — one is a 4-capacity single-harness sweep with
  absolute LEAKY/CLEAN\_MATCHED/PLACEBO values, the other a 2-harness
  ×2-capacity ×3-correction-level grid — and the fully-corrected per-capacity
  values for the primary sweep are not carried in the main text at all
  (already relocated to `EXTENDED_TECHNICAL_DETAIL.md`), so merging them
  would have required either expanding the appendix-relocated content back
  into the main text or fabricating a table shape neither original table
  has. We left the two tables separate rather than force a merge that risked
  numeric fidelity for uncertain page gain.)
- **§4.4 (Case Study 4)**, **§4.1 (Case Study 1)**, **§5.2** (the
  magnitude-triangle's three-qualification paragraph), **§5.6** (the ADA-bound
  numeric check), **§5.7** (the non-Gaussian check), **§5.9**'s closing
  paragraph, and **§6.2** (the automation-attempt discussion) were each
  tightened by 10–30%: shorter transitions, one fewer restatement of an
  already-stated hedge, dash-asides converted to direct sentences, without
  touching a single reported number, interval, or conclusion.
- **Related Work (§2), Method (§3)**, every table (severity vocabulary, Holm
  summary, five-mechanism map, factorial, primary sweep, real-feature,
  magnitude-triangle, ADA-bound, severity-surface, checklist-scope), both
  figures, the References, and Appendices A–D were left untouched, per the
  compression brief's explicit instruction not to expand or further compress
  the appendices.

**Result.** 53 pages before this pass (46 main + 7 appendix) → **51 pages**
after (44 main + 7 appendix). This is real, non-trivial compression achieved
purely through merging and de-duplication, but it falls short of the
30–35-page target stated in the compression brief. The shortfall is not for
lack of effort: this paper's remaining bulk, once the vertical redundancy
identified above was removed, is almost entirely (a) statistical detail that
is load-bearing for the paper's central claim that severity measurement is
harder than severity detection (the very corrections, factorials, and
calibration checks that are this paper's most distinctive contribution), or
(b) content already relocated to the appendices/supplementary markdown files
in the prior session's pass, leaving nothing further to relocate without
re-inflating those already-dense documents. Reaching 30–35 pages would have
required cutting a genuine finding, hedge, or robustness check rather than a
duplicate restatement — out of scope per this pass's explicit instructions —
so the page count was allowed to land at its honest value (51) rather than
being forced down further.

**Verification.** `code/53_verify_paper_numbers.py` ran after every section
edit and stayed at 558/558 checks passing throughout (one transient failure
— `paper_draft.md` drift — appeared between edits and the final
`code/52_sync_paper_draft_md.py` re-sync, exactly as expected, and cleared
once that script ran). Two-pass `pdflatex` compiled clean (0 `^! ` errors, 0
undefined references) both before and after. An identity-leak grep for this
paper's author-name variants over every file touched (`main.tex`,
`paper_draft.md`, this file) found nothing, as before this pass.
`supplementary_material.zip` was rebuilt via `code/91_build_supplementary_zip.py`
after this pass completed.

*End of Addendum 2.*

## Addendum 3: a narrative-restructuring pass (lead with one claim, not eleven)

Independent human review (not another automated pass) identified a
different problem from Addenda 1-2: the abstract and introduction opened by
enumerating essentially every result in the paper -- five mechanisms, the
severity vocabulary, the K/operating-point surface, §5.9's matched
head-to-head, and every robustness check -- before establishing one central
claim, which risks overloading a reviewer's working memory before the
paper's actual contribution is legible.

Both rewritten to lead with one bolded central claim -- that measuring
selection-induced optimism's size, not detecting its existence, is this
paper's real contribution, and that severity is a property of a (mechanism,
metric, control) triple -- before any per-mechanism numeric detail. The
abstract's exhaustive walkthrough of all five mechanisms' individual point
estimates was compressed to a summary paragraph pointing into §4/§5, where
every number already appeared in full; the introduction's two-finding
preview was tightened the same way. Related Work's "pipelines this paper
audits" paragraph was also lightly compressed (citation list combined into
fewer, denser clauses; same citations, same claims). No finding, hedge, or
number was removed from the paper -- only reordered, and where a number was
already stated in the body, not repeated a second time in the abstract.

Verification: `code/53_verify_paper_numbers.py` at 558/558 throughout (one
transient failure was the expected `paper_draft.md`-out-of-sync warning
after each edit, resolved by re-running `code/52_sync_paper_draft_md.py`
before the next check); 2-pass `pdflatex` clean, 0 errors, 0 undefined
references. Page count: **49 pages**, down from 51 (a 2-page reduction from
restructuring the abstract and introduction alone, on top of the 2-page
reduction Addendum 2's compression pass already achieved -- confirming, as
that addendum's blueprint predicted, that this paper's remaining
compressible bulk included real *vertical* redundancy the same information
restated at the abstract/intro/body/conclusion level, which restructuring
addresses directly). Identity-leak grep on every touched file
(`main.tex`, `paper_draft.md`, this file): clean. `supplementary_material.zip`
rebuilt via `code/91_build_supplementary_zip.py` after this pass completed.

*End of Addendum 3.*

## Addendum 4: a second independent review, a further abstract cut, a new figure, and a title change

A fresh, independent read of the Addendum-3 rewrite (fed the actual
compiled PDF, not the LaTeX source) found the abstract, even after
Addendum 3's restructuring, still carried upwards of twenty distinct
technical ideas before the reader reaches the first section. Three
changes followed:

1. **A second, more aggressive abstract cut.** Every remaining
   per-mechanism numeric walkthrough (the $\Delta$-vocabulary's exact
   span, the $2\times2\times2$ factorial's explicit mention, the exact
   $R^2$/$F$-statistic values for the severity surface and the matched
   head-to-head) was removed from the abstract, since every one of these
   numbers already appears in full in §3-§5.9 and was independently
   re-confirmed present there (`code/53_verify_paper_numbers.py`, 558/558)
   before and after the cut. No hedge was softened: ``we do not rank the
   three axes,'' the mechanism-identity/harness-maturity ambiguity in
   §5.9, and every ``untested''/``not a bound'' qualifier are all still
   there, just without the specific statistics repeated inline.
2. **A new conceptual figure** (Figure~\ref{fig:central-claim}, start of
   §4): a simple two-row diagram -- selection $\to$ reuse $\to$ optimistic
   metric on top, Mechanism/Metric/Control $\to$ observed severity on the
   bottom -- stating the paper's central claim visually before the
   mechanism-by-mechanism detail begins. This is the paper's second real
   figure (the first is Figure~\ref{fig:severity-surface}); everything
   else in the document is a table, not a figure, a distinction the
   review that requested this addendum did not have available to it
   before checking, since the same session found there was almost
   nothing to consolidate at the true-figure level.
3. **The title was broadened.** ``Selection-Induced Optimism in
   Hidden-State Hallucination Detection: A Code-Verified Taxonomy, and
   What Governs Severity Within a Controlled Harness'' became ``Severity
   Is a Property of Mechanism, Metric, and Control, Not of Mechanism
   Alone: A Code-Verified Taxonomy of Selection-Induced Optimism in
   Hidden-State Hallucination Detection'' -- foregrounding the general
   methodological claim (already the abstract's own lead sentence) rather
   than only the specific domain it is demonstrated in, per explicit
   author confirmation before the change was made (a title change this
   late touches citations and running headers, so it was not done
   unilaterally).

Verification: `code/53_verify_paper_numbers.py` at 558/558 after each
step; 2-pass `pdflatex` clean, 0 errors, 0 undefined references, including
the new `tikzpicture` figure compiling without error and rendered/visually
inspected (not just checked for compile success) before being accepted.
Page count: **50 pages**, up from 49 -- the one page reduction this
addendum could have claimed from the second abstract cut was spent
instead on the new figure, which was judged worth that one page given
what it was specifically requested to fix. Identity-leak grep on every
touched file: clean. `supplementary_material.zip` rebuilt via
`code/91_build_supplementary_zip.py` after this pass completed.

*End of Addendum 4.*

**Follow-up to Addendum 4**: the same reviewer who requested the title
change noted the result was itself slightly long. ``, Not of Mechanism
Alone'' was trimmed from the main clause -- redundant given ``a property
of mechanism, metric, and control'' already implies severity is not the
mechanism alone -- keeping the domain reference (``Hidden-State
Hallucination Detection'') in the subtitle clause rather than dropping it,
for discoverability. 558/558 checks, 0 errors, identity-clean, zip
rebuilt (572 entries, unchanged).

## Addendum 5: a full nine-phase adversarial review's fixes

A full nine-phase review (three simulated independent reviewers plus an
Action Editor, reading the compiled PDF directly, including rendering and
visually inspecting Figure 1 rather than only its caption) landed at
Accept (~70%) with five concrete, high-impact-low-risk fixes identified:

1. **Figure 1 annotated** with the paper's own headline control-relativity
   number ($1.45\times$--$5.9\times$, §4.3) directly on the "Observed
   severity" node, so the figure carries quantitative content independent
   of its caption -- the review's specific critique was that the diagram,
   while legible and correctly captioned, "carries no number at all."
2. **§5.9's nested F-test statistic and its harness-maturity caveat fused
   into one sentence** rather than the caveat following in a separate
   sentence/paragraph, per the review's specific finding that a reader
   could quote $F(3,42)=799.7$ without the qualification that this
   comparison cannot yet separate mechanism identity from unequal harness
   maturity.
3. **An independence-assumption footnote added to the same F-test**: the
   correctness-focused simulated reviewer (Reviewer #3 analog) noted the
   test's degrees-of-freedom accounting assumes independence across the
   48 cells, but cells within a mechanism share generator code and, for
   Mechanism 3, a pooled data structure -- flagged as an open technical
   question, not resolved (no clustered-residual re-analysis was run;
   the paper states plainly that the $304\times$ intercept spread makes
   it very unlikely such a correction would overturn the qualitative
   conclusion, but this is judgment, not verification).
4. **The Introduction's eight-item contributions list trimmed** to one or
   two clauses per item, moving inline statistics (the $R^2=0.976$
   severity-surface fit; the $75.5\%$/$28.1\%$/$-6.2\%$ CS3 factorial
   attribution) to their home sections (§4.2-§4.5, §5, §6), where they
   already appear in full -- the same "too many technical ideas
   competing for primacy" pattern the abstract was restructured to avoid
   in Addenda 3-4, recurring one level below it.
5. **Mechanism 1's "plausibly the most severe" claim fused with its own
   countervailing real-feature evidence** (an order-of-magnitude-smaller
   gap at a higher operating point, §4.1) in the same sentence, at both
   of its two occurrences in the paper (§4.1 itself and the Discussion
   summary), rather than the countervailing evidence appearing only in
   the next paragraph or a different section.

No number, hedge, or finding was changed; every fix is reordering,
annotation, or clause-level fusion. Verification: `code/53_verify_paper_numbers.py`
558/558 throughout; 2-pass `pdflatex` clean, 0 errors, 0 undefined
references; new TikZ annotation compiled and the resulting figure
re-rendered and visually re-inspected (not just checked for compile
success). Page count: **49 pages**, down from 50 -- the contributions-list
trim (fix 4) recovered the one page the earlier figure addition had cost,
net effect of this addendum is page-neutral relative to Addendum 4's end
state while fixing five real issues. Identity-leak grep on every touched
file: clean. `supplementary_material.zip` rebuilt via
`code/91_build_supplementary_zip.py` after this pass completed.

*End of Addendum 5.*
