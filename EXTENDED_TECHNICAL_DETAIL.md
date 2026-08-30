# Extended Technical Detail — Supplementary Material

This document holds technical detail relocated out of `draft/latex/main.tex`
during a length-compression pass. It is a companion to
`PROVENANCE_LOG.md`, which holds the same paper's full
correction-history record.

**Nothing below is required to verify any number the paper currently reports.**
Every headline figure is stated in the main text, and each block below is
reproduced near-verbatim from the LaTeX source that previously compiled into
the PDF, so that a reader who wants the derivation, the per-cell values or the
argument behind a one-sentence summary in the paper can find it.

Blocks are organized by the paper section they were relocated from, and each
carries a note saying what the main text retains in its place. LaTeX markup is
preserved as-is for fidelity; cross-references such as
"Appendix~\ref{app:k-form}" or "§4.3" refer to `draft/latex/main.tex`, and
references to a former "Appendix B.3" or "Appendix D.2" refer to the
correspondingly-headed section of this document.

---

## §2 (Related Work) — the definition's worked example

Relocated from §2. §2 now states the (mechanism, metric, control) triple and the $1.45\times$/$5.9\times$ spread in one sentence, with the worked numbers where they belong, in §4.3. The full original passage:

```latex
disagree. Our own data make the point. On a single Mechanism-3 cell (capacity
128), the admissible control family we consider defensible has \emph{two}
distinct members, not three --- a blind retrain on the full fold, and the same
blind retrain on the $85\%$ subset its epoch count was chosen on; an adaptive
control selecting on a disjoint carve-out reproduces the second of these
bitwise ($15/15$ seed-fold pairs at capacity 128, §4.3) and so does not add a
third --- giving
$+0.0034$ and $+0.0049$, a $1.45\times$ range on a quantity we report
to two significant figures. A contaminated adaptive control §4.3
disqualifies on independent grounds gives $+0.0198$, and the range becomes
$5.9\times$ if it is counted. The disqualification is the point rather than a rescue: it is a
judgement about which counterfactual is admissible, and that judgement, not the
mechanism or the metric, is what fixes the number. Every severity
number in this paper is therefore reported as a signed change in a named
metric \emph{against a named control}, and §4.3 reports the spread across the
control family rather than only one member of it.
```

---

## §2 (Related Work) — the Pineau / ML Reproducibility Checklist paragraph

Relocated from §2's ``Existing leakage taxonomies and checklists'' paragraph.

```latex
it for fifteen years. The
NeurIPS ML Reproducibility Checklist (Pineau et al.\ 2021) is
reporting-practice-oriented rather than leakage-specific and would not, on its
own, have caught any of the sixteen issues Appendix A documents in this
paper's own instrument: the two kinds of checklist are complementary, not
interchangeable, and a paper can satisfy the former while still committing the
latter. Appendix D.1 gives the item-by-item comparison.
```

---

## §4.3 (Case Study 3) — the retired within-harness ratio diagnostic, and its Sweep C tabular

Relocated in full from §4.3. The diagnostic $(\text{LEAKY}-\text{CLEAN\_MATCHED})/(\text{CLEAN\_MATCHED}-\text{PLACEBO})$ is retired as a pass/fail criterion (it is confounded with the operating point); §4.3 states the retirement and the unconfounded ground on which \texttt{code/49}'s original control is still disqualified. The full argument, including the eight-cell band, the four anisotropic cells, and the five Sweep C cells:

```latex
\textbf{A within-harness ratio diagnostic, now retired.}
$(\text{LEAKY}-\text{CLEAN\_MATCHED})/(\text{CLEAN\_MATCHED}-\text{PLACEBO})$
should be a fraction: the placebo has adaptive selection driven by pure noise,
so the denominator is what an honest signal is worth and the numerator is what
reusing the fold adds on top. Across \emph{eight} of this paper's harness and
capacity cells that ratio spans $0.06$--$0.85$. The
non-budget-matched control puts this harness at $\mathbf{1.60}$ --- outside that
band, read as the signature of a degraded control --- against the budget-matched
control's $\mathbf{0.83}$, inside it and next to \texttt{code/43}'s own $0.85$
at capacity 384. \textbf{Applied consistently, the band breaks in two ways.}

\emph{(i) The eight-cell comparator set is a restriction, not a census.} The
four anisotropic cells of \texttt{code/27} are excluded from it, and three of
the four fall outside the band: capacity 16 at $+0.003$ (far below), capacity
128 at $\mathbf{+3.37}$ (far above), and capacity 384 at $\mathbf{-2.71}$, whose
denominator is \emph{negative} --- PLACEBO beats CLEAN\_MATCHED there by
$-0.0037$. Only capacity 48, at
$+0.196$, sits inside. Two of the three offenders, capacities 128 and 384, are
cells §4.3 and Appendix A cite \emph{approvingly} as anisotropic evidence that
the mechanism strengthens with covariance structure. A diagnostic used to
disqualify a control we dislike must be run on evidence we like, and run there
it says the anisotropic harness has a different --- in one cell sign-inverted
--- relationship between honest-signal value and fold-reuse increment than the
isotropic one.

\emph{(ii) It was never run on \texttt{code/47}'s Sweep C, which is inside the
band's own declared scope} (``the isotropic and real-feature families''). Run
there, on Sweep C's five operating-point cells:
%
\begin{center}\small
\begin{tabular}{lccccc}
\toprule
AUROC$_0$ & $0.70$ & $0.80$ & $0.90$ & $0.95$ & $0.985$ \\
\midrule
$(\text{LEAKY}-\text{CM})/(\text{CM}-\text{PLACEBO})$ &
  $0.910$ & $0.205$ & $0.061$ & $\mathbf{0.047}$ & $0.029$ \\
\bottomrule
\end{tabular}
\end{center}
%
--- \emph{three of five outside the $0.06$--$0.85$ band}, and four of five
outside it as actually measured, whose floor is $0.0631$ rather than a rounded
$0.06$. The one cell comfortably inside, AUROC$_0=0.80$, is bit-identical to
\texttt{code/02d}'s capacity-128 cell, \emph{already} one of the eight band
members --- so the single Sweep C cell ever exposed to the diagnostic was one it
could not fail independently.

\textbf{The pattern's shape is why we retire the diagnostic's general use.}
The ratio falls monotonically with the operating
point, $0.910$ at $0.70$ to $0.029$ at $0.985$, which is what §5 predicts --- the
numerator collapses toward the ceiling faster than the denominator does. The
ratio is therefore \emph{confounded with the operating point}, and a band drawn
from cells at one operating point cannot discriminate a degraded control from a
well-behaved one measured at another. That bites concretely on §5.4, whose two
transport ratios both divide by the AUROC$_0=0.95$ Sweep C cell at $0.047$, a
third below the band's floor: by this paper's own former criterion the
denominator of its own headline transport check is a ``degraded'' cell. We
therefore report the ratio as a descriptive, operating-point-dependent
within-harness quantity, computed and shown wherever we quote it, and withdraw
its use as a pass/fail criterion --- including the use above that disqualified
\texttt{code/49}'s control at $1.60$. That control is still disqualified, on the
unconfounded ground that it trains on $85\%$ of what its comparator trains on,
so most of its larger gap is the control falling rather than the leaky arm
rising, which \texttt{code/79} now measures directly.
```

---

## §4.3 (Case Study 3) — Parts A and B of the selection-budget controls, per capacity

Relocated from §4.3. The new §4.3.2 retains the four-capacity mean ($+0.0023 \to +0.0007$) and the ``none of the four BCa intervals excluding zero'' clause; the per-capacity values and the full Part A / Part B narrative are here.

```latex
\textbf{The control carries a second budget asymmetry, and sizing both of them
dissolves the estimate above.} The parenthetical at the
head of this subsubsection concerns the \emph{retrain}, and \texttt{code/67}
sizes that one (reported below, alongside the adaptivity control it bounds). A
separate asymmetry concerns the \emph{selection run}: LEAKY's selection run trains on $100\%$ of
\texttt{tr\_idx} and argmaxes on the $112$-sample fold it later reports on,
while CLEAN\_MATCHED's trains on $85\%$ and argmaxes on a $68$-sample carve-out
taken out of its own training set. Both halves of that asymmetry disadvantage
the control, and inside one fold they are coupled --- raising
\texttt{ES\_HOLD\_FRACTION} to match the selection-set size necessarily worsens
the selection run's training deficit --- which is the trade Appendix A.3
describes. \texttt{code/75} runs two controls at the primary estimate's exact
operating point and all four of its capacities, after asserting that its
recomputed \texttt{ES}$=0.15$ arms reproduce \texttt{code/02d}'s shipped
per-seed AUROCs bit-for-bit.

\emph{(A) The fold-matched carve-out.} Appendix A.3 named a fold-matched
variant and did not run it. The matching value is $1/(K_{\text{CV}}-1)=0.25$,
not $1/K_{\text{CV}}=0.20$: \texttt{ES\_HOLD\_FRACTION} is a fraction of
\texttt{tr\_idx}, and $0.25$ of $448$ is exactly the $112$-sample fold, where
$0.20$ gives $90$. At $0.25$ the gaps are $+0.0012$, $+0.0002$, $+0.0016$ and
$+0.0020$ at capacities $16/48/128/384$, against the shipped $+0.0019$,
$+0.0011$, $+0.0036$, $+0.0024$. \textbf{The interval criterion and Wilcoxon disagree at this cell.}
Capacity 128 moves $+0.0036$
($p=0.0012$) to $+0.0016$ with Wilcoxon $p=0.14$ but a BCa interval of
$[+0.0000156,+0.0033730]$, whose lower limit is strictly positive. Since §5.3
and Table~\ref{tab:magnitude-triangle} commit this
paper to the interval criterion, and Part B below is judged by it, we state both
readings: under the interval criterion this one cell of the twelve still excludes
zero, and under Wilcoxon none of the twelve does. It is the only cell in the
Part A grid where the two criteria disagree. We do not rest anything on it ---
Part B and the factorial below both remove the asymmetry Part A can only trade.

\emph{(B) A selection run that also sees $100\%$ of the data.} The second
asymmetry cannot be removed inside a fold, because \texttt{tr\_idx} and
\texttt{val\_idx} exhaust the pool. \texttt{code/75} therefore carves the
early-stopping set out \emph{before} folding: $94$ stratified samples ---
the same size as a fold --- are held out of the cross-validation pool
entirely, and CLEAN\_MATCHED\_OOF's selection run trains on all of
\texttt{tr\_idx} and argmaxes on that out-of-fold set. This removes the second
asymmetry, and as a by-product the first: the blind retrain comes out bitwise
identical to the selection run in $100\%$ of (seed, fold) pairs, asserted at
runtime. \textbf{Against that control the fold-reuse gap is $+0.0003$,
$+0.0003$, $-0.0001$ and $+0.0026$ across the four capacities --- mean
$+0.0007$ against the primary mean $+0.0023$, with \emph{none} of the four BCa
intervals excluding zero, and the flagship capacity-128 cell at $-0.0001$
($p=0.54$, $46/100$ seeds positive).} Measured directly on the same reduced
pool, the selection-run asymmetry alone is $+0.0012$, $+0.0028$, $+0.0031$ and
$+0.0009$, individually significant at capacities 48 and 128 --- i.e.\ at those
two capacities it accounts for essentially the whole in-fold gap.
```

---

## §4.3 (Case Study 3) — the downstream-averaging readout, in full

Relocated from §4.3. §4.3.3 now states the surviving half in five sentences (averaging attenuates whatever gap is present; against a corrected control only the direct out-of-fold readout retains a gap, $+0.0059$; the per-fold-model readout does not, $+0.0015$, interval covering zero; the line is withdrawn from the evidence carrying the mechanism's severity). The full original passage, including \texttt{code/55}'s $2.3$--$5.8\times$ per-capacity figures:

```latex
\subsubsection{The mechanism at its source: what the pipeline's downstream averaging hides}

\textbf{Is the measured severity small only because the pipeline averages it
away?} The synthetic harness --- like MultiHaluDet's own
\texttt{run\_pipeline.py:108} that it mirrors --- averages five fold-models'
test-set activations before the
meta-learner scores them, so a fold-specific leaked signal could be largely
cancelled before the reported metric ever sees it. \textbf{That alternative is
supported, and we report it as such.} Reading the \emph{same} trained
fold-models without the averaging (\texttt{code/55}) --- both readings taken on
\texttt{code/02d}'s shipped arms, so every trained model is shared and the
comparison isolates the averaging step alone --- raises the gap at every
capacity by roughly $2.3$--$5.8\times$, to
$+0.0073$/$+0.0066$/$+0.0084$/$+0.0084$ at capacities 16/48/128/384, and makes
it significant at all four capacities rather than the two the averaged readout
of the sweep below reaches. We keep both readings: the averaged
number is the correct estimate of inflation in what the audited pipeline
actually reports, and the un-averaged number the correct estimate of the
mechanism's severity at its source, larger at every capacity by the
$2.3$--$5.8\times$ quoted above (mean $3.9\times$, median $3.7\times$; we quote
the range rather than a central value because the four capacities disagree
by more than a factor of two). The honest summary is therefore
narrower than ``this mechanism is small'': it is small \emph{as reported},
partly because a downstream ensembling step attenuates it, and a pipeline with
the same leak but without that step should be expected to show more
(Appendix A.4).

\textbf{This comparison is measured against the same uncorrected control as the
primary sweep.} \texttt{code/55} line 122
reads \texttt{ES\_HOLD\_FRACTION = \_M.ES\_HOLD\_FRACTION}, importing the
uncorrected $0.15$ from \texttt{code/02d}, and the whole script is a re-readout
of \texttt{code/02d}'s own trained models. Its CLEAN\_MATCHED arm is therefore
not merely similar to the arm whose asymmetry accounts for the entire primary
gap --- it \emph{is} that arm. \texttt{code/83} repeats the readout comparison on
the corrected geometry, at capacity 128, with both a shipped-construction control
and a fully corrected one:

\begin{center}\small
\begin{tabular}{lcc}
\toprule
Readout & vs.\ shipped control & vs.\ corrected control \\
\midrule
Averaged (what the pipeline reports) & $+0.0030$ $[+0.0010,+0.0053]$ & $-0.0001$ $[-0.0022,+0.0021]$ \\
Un-averaged, per fold-model & $+0.0089$ $[+0.0061,+0.0122]$ & $+0.0015$ $[-0.0016,+0.0044]$ \\
OOF matrix read directly & $+0.0088$ $[+0.0066,+0.0114]$ & $\mathbf{+0.0059}$ $\mathbf{[+0.0044,+0.0075]}$ \\
\bottomrule
\end{tabular}
\end{center}

\textbf{The finding splits in two, and only one half survives.} Against the
shipped control a suppression of the same order reproduces --- \texttt{code/83}'s
own recomputation on this table gives $2.95\times$ (rounded to $3.0\times$)
at capacity 128, against \texttt{code/55}'s independently-computed
$2.3$--$5.8\times$ across all four capacities ($2.3\times$ at capacity 128
specifically) --- so \emph{that averaging
attenuates whatever gap is present} is robust and stands. But the claim it was
used to support --- that the mechanism is therefore not benign at its source ---
requires the un-averaged gap against a \emph{sound} control to be non-zero, and
at the per-fold-model readout it is not (table above, corrected column). What
the averaging was suppressing at that readout was largely the control asymmetry,
not the leak. The one readout where a gap does survive correction is the OOF
matrix read directly --- and that
readout is in-sample for every arm, so it is usable comparatively but is not a
performance estimate. We therefore report the source-versus-reported distinction
as real in direction and much smaller than \texttt{code/55} suggested, and we
withdraw this line from the set of evidence carrying the mechanism's severity.
```

---

## §4.4 (Case Study 4) — the two marginal calibration sweeps' disagreement, in full

Relocated from §4.4. The main text now reports only the resolved answer: the joint $(\rho,\text{sep})$ sweep of \texttt{code/81} identifies the calibration factor along the real-profile manifold, and the corrected range $+0.0029$--$+0.0040$ stands. The chronology of the disagreement that preceded it:

```latex
\textbf{Across the correlation sweep the ratios are strikingly stable; across
the separation sweep they are not.} The table below gives each estimator's
recovery ratio (estimate / true bias) on both sweeps, and the corrected value
each implies when applied to the observed cell.

\begin{center}\small
\begin{tabular}{lccc}
\toprule
Calibration sweep & $\Delta_{\text{boot}}$ & $\Delta_{\text{boot}}^{\text{std}}$ & $\Delta_{\text{wc}}$ \\
\midrule
Correlation grid ($\rho=0\to0.98$): ratio & $0.50$--$0.55\times$ & $0.28$--$0.30\times$ & $1.27$--$1.34\times$ \\
$\quad\to$ corrected value & $+0.0039$ & $+0.0030$ & $+0.0040$ \\
Separation grid (true bias $>0.002$): ratio range & $0.47$--$1.25\times$ & $0.28$--$0.52\times$ & $1.23$--$2.48\times$ \\
$\quad$nearest-to-real-data condition: ratio & $1.25\times$ & $0.52\times$ & $2.48\times$ \\
$\quad\to$ corrected value & $+0.0017$ & $+0.0017$ & $+0.0022$ \\
\bottomrule
\end{tabular}
\end{center}

The separation grid's ranges are $2.6\times$, $1.9\times$ and $2.0\times$ wider
than the correlation grid's, and $\Delta_{\text{boot}}$'s \emph{crosses one}. The two sweeps are marginal and
disjoint: the separation sweep runs at $\rho=0$ throughout, and the correlation
sweep holds $\mu$ fixed at the real cells' plug-in profile so separation never
varies. \textbf{The high-$\rho$, high-separation corner --- which is where the
real data sits --- is simulated by neither}, so the calibration factor at the
real data's own condition is not identified by either marginal sweep. The joint
$(\rho,\text{sep})$ sweep is the experiment that identifies it, and we ran it;
it is reported two paragraphs below and it resolves the disagreement.
At the separation grid's nearest-to-real-data condition (true bias $0.0052$
against the real data's $0.0036$; the next-nearest grid condition is three
orders of magnitude away), $\Delta_{\text{boot}}$ \emph{overstates} by about
$25\%$ rather than understating by half, and this correction moves the
headline \emph{down} rather than up. The cross-estimator agreement does not
discriminate between the two readings: the three
estimators agree to $\pm15\%$ under the correlation-grid correction and $\pm22\%$ under the separation-grid one, so
their concordance is a property of the calibration procedure, not evidence about
which condition is the right one to read at.
```

---

## §5.2 (Magnitude triangle) — qualifications (i)-(iii) in full, including the arm-matching sensitivity and its later correction

Relocated from §5.2, which now states these three qualifications --- shared denominator, small-$n$/capacity-specific upper end, and matching-convention sensitivity --- in one condensed paragraph, giving only the numbers ($9.241\times$; $6.7\times$/$8.0\times$ at capacity 384; $13.1$--$28.7\times$ shipped and $8.9$--$20.8\times$ corrected under control-arm matching). The full passage, including the derivations and the capacity-128/384 grid, and the correction history of the control-arm-matching check across \texttt{code/82} and \texttt{code/84}:

```latex
\emph{(i) The two axes' compared cells share a denominator cell.} The
operating-point axis's sound maximum, $14.266\times$, is the $K=45$ column's
$(0.70)/(0.95)$ ratio; the mechanism axis's minimum divides the real-feature
severity by the same Sweep C cell at AUROC$_0=0.95$. Those two denominators are
bit-identical at $0.0006530612244898$, a structural non-independence worth
disclosing regardless of how close the resulting ratios land. Under this
paper's numerator ($+0.0060$, fold-matched carve-out) that minimum is
$9.241\times$, clearly below $14.266\times$: the two axes overlap substantially
rather than merely touching at a boundary.

\emph{(ii) The mechanism axis has $n=2$, and its upper end is
capacity-specific.} Its two points are the real-feature harness and the
fidelity extension, both at capacity 128, both on the same $400$ real feature
vectors, differing in leakage mechanic --- so its ``span'' is a range over two
observations sharing a denominator, and the propagated intervals overlap
the operating-point axis's almost entirely. The $24.3\times$ upper end rested
entirely on the fidelity extension, which existed at capacity 128 only;
\texttt{code/79} now runs it at $384$, where the gap falls $4.8\times$ under
the shipped control ($3.6\times$ fold-matched) and the fold-matched transport
ratio falls to $6.7\times$ --- \emph{below} the operating-point axis's sound
maximum of $14.3\times$ (the shipped-control ratio falls to $8.0\times$, also
below it). So the axis's upper end is a property of one capacity of one
harness, and at capacity 384 under the fold-matched control the mechanism axis
does not exceed the operating-point axis at all. The whole grid, rather than
its maximum:

\begin{center}\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Capacity 128} & \multicolumn{2}{c}{Capacity 384} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Control & Gap & Ratio & Gap & Ratio \\
\midrule
Shipped ($|{\rm sel}|=39$) & $+0.0250$ & $38.3\times$ & $+0.0052$ & $8.0\times$ \\
Fold-matched ($|{\rm sel}|=64$) & $+0.0159$ & $24.3\times$ & $+0.0044$ & $6.7\times$ \\
\bottomrule
\end{tabular}
\end{center}

\emph{(iii) The $0.0030$ match is on the LEAKY arm only.} \texttt{code/58} reads
\texttt{achieved\_operating\_point} from a leaky quantity in all three rows. The
three harnesses' control arms agree only to $0.0252$ AUROC and their placebo
arms to $0.0412$ --- $8.3\times$ and $13.5\times$ worse. Since the reported
quantity is LEAKY minus CONTROL, matching on LEAKY alone leaves the control
spread inside the ratio, and the convention that maximizes the ratio is the one
that was used. \texttt{code/82} recomputes the whole check under each
convention, changing nothing else:

\begin{center}\small
\begin{tabular}{lcc}
\toprule
Matched on & Arm spread across harnesses & Mechanism axis (interpolated) \\
\midrule
LEAKY arm (as shipped) & $0.0030$ & $15.1$--$42.0\times$ \\
CONTROL arm & $0.0252$ & $13.1$--$28.7\times$ \\
PLACEBO arm & $0.0412$ & $13.0$--$24.0\times$ \\
\bottomrule
\end{tabular}
\end{center}

\textbf{\texttt{code/82}'s table above sources the shipped $+0.0093$/$+0.0250$
numerators, not the corrected $+0.0060$/$+0.0159$ this paper now reports
(§4.3); \texttt{code/84} closes that gap}, rerunning the identical
control-arm-matched probit interpolation against the corrected fold-matched
numerators: $\mathbf{8.9\text{--}20.8\times}$, \emph{lower} than the
shipped-numerator figure and further inside the operating-point axis's
$2.2$--$14.3\times$ span, not outside it. Under either the shipped or the
corrected control-arm matching the mechanism axis runs inside the
operating-point axis's $2.2$--$14.3\times$ span: the two \emph{overlap}, and the
mechanism axis is no longer above it throughout. The primary leaky-arm-matched
comparison, once corrected, already overlaps the operating-point axis
($9.2$--$24.3\times$) without needing this sensitivity check to establish it.
None of the three matching conventions is uniquely correct --- these harnesses
differ in generative process, dimension, sample size, optimizer, scaler and
mechanic all at once, which is why the axis is labelled ``mechanism \emph{and}
harness'' --- but a claim that survives only under the convention and the
numerator that inflate it is not a claim.
```

---

## §5.3 — the extreme-value digression: exact $E[\max]$, $\sqrt{2\ln K}$, and the $c_4(K)$ bias correction

Relocated from §5.3. The main text retains the load-bearing statement --- the selection-stage winner's curse is present and scales with $K$ as extreme-value theory predicts (observed/expected $1.000$, $1.017$, $1.021$, $0.996$ at $K=2,5,15,45$), while its propagation to the reported metric is absent --- and the conclusion that the extreme-value form is supported at the selection stage and is not a predictor of reported-metric severity. The $\sqrt{2\ln K}$-versus-exact comparison and the $s/c_4(K)$ disclosure:

```latex
  \textbf{This design supplies the sampling-noise SD the extreme-value form
  requires.}
  The only $\sigma$ available in the main trajectory sweep was one training
  trajectory's dispersion rather than the sampling-noise SD of $K$ exchangeable
  estimates, and named ``$K$ independently-seeded models'' as a design that
  would supply the right quantity. This is that design, and two diagnostics
  confirm the substitution worked: with the correct $\sigma$, $\sigma$ is nearly
  flat in $K$ (range ratio $1.34$, against $2.66$ for the trajectory $\sigma$)
  and its correlation with the gap collapses from $-0.84$ to $-0.05$, which is
  what genuine selection noise should look like. \textbf{On that $\sigma$ the
  observed selection-stage statistic matches the \emph{exact} $E[\max$ of $K$
  standard normals$]$ to within $2\%$ at every
  $K$} --- not the $\sqrt{2\ln K}$ closed-form approximation to that
  expectation, which is itself $20$--$40\%$ off at these $K$ and worsens with
  $K$, since $\sqrt{2\ln K}$ is only an asymptotic approximation, poor at small
  $K$ --- as the ratios above show. One further disclosure: the
  within-$2\%$ match uses a small-sample bias-corrected $\sigma$ estimator
  ($s/c_4(K)$), while the $\sqrt{2\ln K}$ ratio above it uses the plain
  sample SD ($\text{ddof}=1$); the two differ by up to $25\%$ at $K=2$, so
  part of the improvement over the $\sqrt{2\ln K}$ comparison is the bias
  correction and not solely the switch from an asymptotic form to the exact
  expectation. It does \emph{not} carry through to the
  downstream gap ($c=-0.071$, $R^2=-0.48$), but that is the propagation failure
  just described rather than a verdict on the theory. \textbf{The extreme-value
  form is therefore supported where it applies --- the
  selection stage --- and is not a predictor of reported-metric severity in this
  harness} (Appendix~\ref{app:k-form}).
```

---

## §5.3 — the rank-biserial terminology correction

Relocated from §5.3's variance-compression control. The main text names the third stabilized coordinate the \emph{sign statistic} $(n_+-n_-)/(n_++n_-)$ and states that it is invariant to any monotone reparameterization of AUROC. The correction explaining why it is not the matched-pairs rank-biserial correlation:

```latex
  would need. \textbf{That third coordinate is the sign statistic, not
  ``the matched-pairs rank-biserial correlation''.} The conventional
  matched-pairs rank-biserial is $(T_{+}-T_{-})/(T_{+}+T_{-})$ over the signed
  \emph{ranks} of $|{\cdot}|$, which uses magnitude and is \emph{not}
  monotone-invariant --- monotone invariance is exactly the property this
  control needs, which only the
  sign statistic has.
```

---

## Appendix B.1 (former) — Case Study 2: the scope of the shipped replay artifact

Removed from the paper as $\approx$85--90\% duplicative of §4.2 and the §4 evidentiary grading, which state the $171$\,MB/$4.0$\,MB boundary and the twelve non-replayable numbers directly.

```latex
\subsection{Case Study 2: the scope of the shipped replay artifact}

\textbf{Case Study 2 (GUARDIAN) is verifiable here from a derived artifact.} This paper's own prior, unpublished pipeline. The raw
hidden-state cache it was originally computed from is a 171\,MB file
too large to ship, so \texttt{code/48} also emits a small ($4.0$\,MB)
derived artifact --- per-layer, per-sample cross-validation fold
assignments and probe scores for all 32 layers, for the sequential split,
the reversed split, and all 50 randomized reps --- and \texttt{code/51}
recomputes Case Study 2's \emph{primary} numbers from that artifact alone
and asserts agreement with the committed JSON. \emph{Scope, corrected
after a review found the previous ``every number'' claim false:} the $12$
numbers of §4.2's \texttt{StandardScaler}$\times C$
regularization-robustness sweep (five $\Delta_{\text{sel}}$ values, five
general-gap values, and the two like-for-like unscaled-versus-scaled
values at $C=1$) are \emph{not} replayable from the artifact, because
\texttt{code/48} computes that sweep by refitting probes from the raw
hidden states and never writes the resulting scores into the shipped
\texttt{.npz}. \texttt{code/51} prints this scope boundary rather than
asserting past it.
```

---

## Appendix B.2 (former) — Case Study 2: computational provenance for the separability and base-rate claims

Removed from the paper as $\approx$90\% duplicative of §4.2. Two sentences are retained in §4.2: the half-membership probe's $0.5214$--$0.7885$ range (mean $0.7259$, above $0.70$ at $25$ of $32$ layers) and the $8.0$-point base-rate difference ($0.7300$ against $0.8100$).

```latex
\subsection{Case Study 2: computational provenance for the separability and base-rate claims}
\label{sec:cs2-provenance}

\textbf{A claim we had asserted without computing it, now computed.} An
independent review searched every script and every result file in this
repository for the number §4.2's retraction rests on --- that GUARDIAN's two
sequential halves are ``separable by a hidden-state probe at AUROC
$0.734$--$0.776$'' --- and found it existed only as a hardcoded string in
\texttt{code/48}'s module docstring and as a hardcoded \texttt{note} field in
the JSON that script emits. No half-membership probe had been fitted anywhere.
The same was true of ``argmax all-data AUROC picks L19.'' Both were prose. A
paper whose subject is numbers that cannot be re-derived from what ships had
shipped two of them, and that is the correct reading of the finding.

\texttt{code/61} computes both. \emph{Half-membership probe:} for each layer,
a \texttt{LogisticRegression(max\_iter=1000, C=1.0)} is fit on the unscaled raw
hidden states to predict whether a sample lies in \texttt{H[:400]} or
\texttt{H[400:]}, scored by stratified 5-fold out-of-fold AUROC
(\texttt{random\_state=42}, matching \texttt{code/48}'s probe configuration).
The result: $0.5214$ at L0 rising to $0.7885$ at L25, mean $0.7259$ across the
32 layers, above $0.70$ at $25$ of them. The direction and the conclusion of
the original claim hold; its stated range did not come from this or any other
computation and is replaced by the measured one. \emph{All-data argmax:} the
same probe configuration applied to the hallucination label over all $700$
samples peaks at \textbf{L21} ($0.7829$), not L19.

\textbf{A cheaper, more parsimonious confound we had missed.} The same review
noted that the two halves differ in \emph{label base rate}, which is checkable
directly in the already-shipped \texttt{case\_study\_2\_probe\_scores.npz}
without the raw cache: \texttt{sequential\_\_y\_sel} has positive rate
$0.7300$ ($n=400$) against \texttt{sequential\_\_y\_ho} at $0.8100$ ($n=300$),
a difference of $8.0$ percentage points, while the $50$ randomized comparison
reps are stratified and agree to within $0.17$ points ($0.7650$ against
$0.7633$ on average). An $8$-point base-rate difference is on its own
sufficient to make the halves separable and their AUROCs non-comparable, and it
requires no appeal to a general population difference.

\textbf{A base-rate-matched control, and what it shows.} Subsampling positives
at random from the positive-enriched half until both halves sit at a common
rate of $0.73$ --- no reweighting, no duplication --- and re-running the full
32-layer computation over $20$ independent draws gives
$\Delta_{\text{sel}}=-0.0050$ (SD $0.0058$), against $-0.0038$ on the unmatched
sequential split, with L11 selected in all $20$ draws. \textbf{Matching the
base rates does not rescue the sequential split.} The base-rate difference is a
verifiable component of the confound rather than its entirety; the deeper
problem, visible in the near-monotone rise of the per-layer gap with depth
reported below, is that the halves differ in the representations themselves.
Either way the sequential split cannot support the estimate, which is why the
primary result uses randomized stratified splits.
```

---

## Appendix B.3 (former) — Case Study 2: $\Delta_{\text{sel}}$'s mechanical null, the decomposition, and all 32 per-layer gaps

Removed from the paper. §4.2 states the mechanical null ($+0.0429$), the decomposition ($A=+0.042865$, $B=+0.0173$, $40.4\%$ cancellation) and the three-number summary of the 32-layer sweep ($+0.192$ mean, L11's own $0.188$, $18$ of the other $31$ larger). The full 32-value vector and the per-rep distribution are here.

```latex
\subsection{Case Study 2: $\Delta_{\text{sel}}$'s mechanical null, and the decomposition it makes available}
\label{app:delta-sel-null}

\textbf{Positive in expectation, not mechanically positive.} $\Delta_{\text{sel}}$
is positive in expectation even under a pure-noise null: $l^\star=\arg\max_l
\text{cv}_l$ while $\text{gap}_l=\text{cv}_l-\text{ho}_l$ contains the same
$+\text{cv}_l$ term, so selecting the argmax of $\text{cv}$ tends to select a
layer whose gap is upward-biased relative to the mean gap. It is \emph{not}
positive for any data whatsoever --- unlike Case Study 4's retired
$\Delta_{\text{wc}}$ (Theorem~\ref{thm:wc-nonneg}), $3$ of the $50$ reps here
come out negative --- and an earlier revision titled this subsection ``the
mechanical positivity of $\Delta_{\text{sel}}$,'' which overstated it. What is
true is that the statistical significance of a measured $\Delta_{\text{sel}}$
\emph{against zero} is not evidence of anything beyond
argmax-over-a-noisy-statistic.

\textbf{The null we should have compared against, and did not.} Having conceded
that $\mathbb{E}[\Delta_{\text{sel}}]>0$ under noise, the previous revision
still reported the effect against $H_0:\Delta_{\text{sel}}=0$. The reference
that answers the question actually being asked is the value the same estimator
returns when the layer ordering carries no \emph{transferable} information.
\texttt{code/64} constructs it directly from the shipped
\texttt{case\_study\_2\_probe\_scores.npz}: for each of $20{,}000$ draws, and
independently within each of the $50$ reps, permute which layer's held-out
AUROC $\text{ho}_l$ is paired with which layer's $\text{cv}_l$, then recompute
$\Delta_{\text{sel}}$ with the argmax rule unchanged. Both marginal vectors,
the argmax, and the layer count are preserved; only the cv-to-ho correspondence
is destroyed. The replayed $\Delta_{\text{sel}}$ is asserted against the
committed JSON ($+0.025549$ against $+0.025549$) before any draw is taken.

\textbf{Result: the null mean is $+0.0429$} (SD $0.0043$, $95\%$
$[+0.0350,+0.0517]$), and the observed $+0.0255$ falls below the entire
interval, $P(\text{null}\leq\text{observed})=0.0000$ over $20{,}000$ draws.

\textbf{The decomposition.} Since $\mathbb{E}_\pi[\text{ho}_{\pi(l)}]
=\overline{\text{ho}}$, the null's mean is analytic, and the statistic splits
exactly into
%
\[
\Delta_{\text{sel}} \;=\; \underbrace{\big(\text{cv}_{l^\star}
-\overline{\text{cv}}\big)}_{A\ =\ +0.042865}
\;-\;\underbrace{\big(\text{ho}_{l^\star}
-\overline{\text{ho}}\big)}_{B\ =\ +0.017316}\;=\;+0.025549 .
\]
%
$A$ is the winner's curse on the selection criterion --- and is the null's own
mean, which the simulation reproduces to $7\times10^{-5}$ ($+0.042936$),
a consistency check on both. $B$ is
\emph{transferred layer quality}: how much better than an average layer the
CV-selected layer is on data that played no part in selecting it. Over the $50$
reps, $B$ has SD $0.0135$, $t=9.07$, $p=4.6\times10^{-12}$, and is negative in
only $7$. So CV-argmax layer selection is not pure noise-chasing: it recovers a
genuinely better-than-average layer, and that recovery offsets
$B/A=40.4\%$ of the raw curse.

\textbf{Why we regard this as a stronger finding than the one it replaces.} The
zero-comparison established something the paper had already conceded was true
by construction. The null-comparison establishes two things that were not
known: that CV-argmax layer selection transfers real signal at this sample
size, and that the residual optimism a practitioner pays, $+0.0255$, is what
survives \emph{after} that transfer rather than the whole of the curse. It also
sharpens the localization point in §4.2: the argmax layer wanders across $14$
distinct values from L5 to L23, and yet the layers it lands on are on average
better than a randomly chosen layer --- instability of the argmax and
informativeness of the argmax are both true here, and neither alone describes
the situation.


\medskip\noindent\textbf{The full 32-layer decomposition.} GUARDIAN's raw cache contains the
complete (700, 32, 4096) Mistral-7B representations at full dimensionality, not
just L11 (an earlier version of this section disclosed this as unavailable;
that was incorrect). \texttt{code/48} repeats the identical CV-vs-held-out
comparison at all 32 layers under GUARDIAN's own sequential
\texttt{H[:400]}/\texttt{H[400:]} split, sanity-checked to reproduce the
published L11 numbers exactly. Under this split the mean optimism gap across
all 32 layers is $+0.192$ (SD $0.053$), statistically indistinguishable from
L11's own $+0.188$, and $18$ of the other $31$ layers show a \emph{larger} gap
than L11.

Under GUARDIAN's own sequential split, L11 is selected and this component is
$-0.004$ --- indistinguishable from zero, so the 18.8-point gap is not
disproportionately concentrated at the selected layer once the
population-difference confound is present in every layer equally. Reversing
which half selects picks a different layer entirely (L17, not L11) and gives
$+0.074$ --- a large swing in both which layer is picked and what the component
measures, itself direct evidence that the sequential-split number tracks split
order rather than a stable property of the selection procedure. Recomputing
under randomized stratified $400/300$ splits, each rep evaluated at its own
selected layer, gives $\Delta_{\text{sel}}=+0.0255$ (SD $0.0186$; one-sample
$t=9.71$; Wilcoxon $p=7.7\times10^{-12}$) --- small but, for the first time,
correctly measured and distinguishable from zero. The selected layer spans 14
distinct layers from L5 to L23 across reps, which is itself the argmax
instability the component measures; the full distribution is
L5:1, L10:4, L11:2, L12:1, L13:1, L14:12, L15:8, L16:5, L18:2, L19:7, L20:2,
L21:1, L22:2, L23:2. \textbf{GUARDIAN's own selected layer, L11, is chosen in
only $2$ of the $50$ randomized reps}; the modal choice is L14. Per rep,
$\Delta_{\text{sel}}$ has median $+0.0269$, quartiles $+0.0150$ and $+0.0333$,
range $-0.0261$ to $+0.0690$, and is positive in $47$ of $50$ reps --- so the
mean is not carried by a few large reps, but three reps do come out negative
and a reader should have that rather than only the SD. The \emph{general} CV-vs-held-out gap averaged across all
32 layers is $+0.192$ under the sequential split but only $-0.0023$ (SD
$0.0541$, $t=-0.30$, $p=0.76$) under the same randomized splits ---
indistinguishable from zero.

\textbf{The 32 per-layer gaps, since ``full 32-layer decomposition'' should
mean more than a mean and an SD.} Under GUARDIAN's own sequential split the
per-layer optimism gaps are, from L0 to L31:
$0.067$, $0.067$, $0.114$, $0.111$, $0.133$, $0.148$, $0.153$, $0.165$,
$0.158$, $0.184$, $0.178$, $\mathbf{0.188}$ (L11, the selected layer),
$0.172$, $0.177$, $0.197$, $0.197$, $0.199$, $0.206$, $0.228$, $0.211$,
$0.209$, $0.224$, $0.227$, $0.238$, $0.246$, $0.247$, $0.249$, $0.248$,
$0.251$, $0.246$, $0.240$, $0.261$. Two things are visible here that the
summary statistics hide. The gap rises close to monotonically with depth, from
$0.067$ at L0 to $0.261$ at L31 --- so it is a property of \emph{where in the
network you probe}, tracking how much the two population halves differ in the
deeper representations, and not of the selection. And L11's $0.188$ sits below
the $+0.192$ mean and is exceeded by $18$ of the other $31$ layers, which is
the concrete form of the claim that the 18.8-point headline was not specific to
the selected layer. This is the pattern the randomized-split correction was
built to remove, and it is why the corrected $\Delta_{\text{sel}}$ is an order
of magnitude smaller.
```

---

## Appendix B.4 (former) — Case Study 2: replication count, and what the $p$-value means

Removed from the paper as $\approx$90\% duplicative of §4.2's ``What that $p$-value does and does not mean'' paragraph.

```latex
\subsection{Case Study 2: replication count, and what the $p$-value means}

\textbf{How many reps, and what the $p$-value does and does not mean.} An
earlier version ran $8$ reps and gave no stopping rule, which leaves a
reader unable to exclude that $8$ was chosen after seeing the result. It is
now $50$. The binding constraint is not runtime ($50$ reps takes about $15$
minutes on one core) but the size of the shipped replay artifact, which
stores every per-layer, per-sample probe score for every rep so
\texttt{code/51} can re-derive the headline number without the $171$\,MB raw
cache: it grows at roughly $100$\,KB/rep and reaches $4.0$\,MB at $50$, which
is as large as we were willing to put in a submission archive. The RNG
stream is unchanged, so the original $8$ reps are bit-identical and still
give $+0.0270$ (SD $0.0212$); the move to $+0.0255$ (SD $0.0186$) is what
tightening the estimate does, not a different measurement. More reps do not,
however, fix the statistical problem. All $50$ resample splits of the
\emph{same} $700$ samples, so the SD across reps is split-to-split
variability conditional on this one dataset, not sampling variability across
independent draws from a population, and the one-sample $t$-test's i.i.d.\
assumption is violated. Its $p$ should be read as a within-dataset
robustness statistic --- ``this is not an artifact of which particular split
we happened to take'' --- and \emph{not} as evidence that the effect
generalizes beyond GUARDIAN's own $700$-sample TruthfulQA sample. The
qualitative conclusion (a small, consistently positive selection-specific
component, against a general gap indistinguishable from zero) is what we
assert. Section~4.4's Mechanism-4 estimate, drawn from 24
published result cells, is the closest thing in this paper to a genuinely
replicated version of the same claim.
```

---

## Appendix B.5 (former) — Case Study 2: probe-configuration robustness

Removed from the paper as $\approx$90\% duplicative of §4.2's robustness-check paragraph, which states the $9{,}600$-fit \texttt{n\_iter\_} maximum of $125$ and the $C$-sweep verdict.

```latex
\subsection{Case Study 2: probe-configuration robustness}

\textbf{A robustness check on the probe configuration.} The layer probe here
is \texttt{LogisticRegression} at $d=4096$, $n=400$, \texttt{max\_iter=1000},
default $C=1.0$, on unscaled raw hidden states. A review noted that this
could plausibly be so heavily regularized toward its prior that both
estimates are dominated by regularization rather than fit --- which would
make ``the general gap is $\approx0$'' a statement about the probe, not
about exchangeability. Two things were checked. \emph{Convergence:} the
alternative's premise that \texttt{max\_iter=1000} might not be reached is
false, and measurably so --- across the $9{,}600$ probe fits behind the
primary result the largest \texttt{n\_iter\_} is $125$, with zero
\texttt{ConvergenceWarning}s. \emph{Regularization:} adding
\texttt{StandardScaler} and sweeping $C$ over five decades
$\{0.01,0.1,1,10,100\}$ on the same splits leaves both conclusions intact.
$\Delta_{\text{sel}}$ ranges over
$+0.0324$/$+0.0346$/$+0.0369$/$+0.0360$/$+0.0412$ (every cell $p<3\times
10^{-5}$), and the general gap over
$-0.0044$/$-0.0017$/$+0.0001$/$+0.0027$/$+0.0088$ (SDs $\approx0.055$, every
cell indistinguishable from zero). Like-for-like on the same $20$ reps,
scaling raises $\Delta_{\text{sel}}$ modestly, from $+0.0306$ unscaled to
$+0.0369$ scaled at $C=1$ --- so if anything the unscaled configuration this
paper reports is the mildly conservative one, and regularization was not
suppressing a general gap that scaling reveals. We report the unscaled
result as primary only because it is what GUARDIAN itself ran.
```

---

## Appendix B.3 — the full 24-cell per-cell table (Table `tab:cs4-cells`)

Relocated from Appendix B.3 (`sec:cs4-percell`) as part of the 30--35pp compression pass. The main text there keeps every piece of analysis that reads the disaggregated cells (the 17 sorted non-degenerate $\Delta_{\text{wc}}$ values, the effect of dropping the single largest cell, the "how to read the null" discussion) and the headline aggregate numbers ($+0.0021$ as computed, $+0.0029$ to $+0.0040$ calibration-corrected); per the paper's own stated policy ("Every headline number is stated in the main text and none of the appendices is required to verify one"), this specific row-by-row table is exactly the kind of per-cell detail that policy names as relocatable. Nothing here is new: every number below is unchanged from the version that was in the compiled PDF.

```latex
\begin{table}[!ht]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lllrrrrrrrl}
\toprule
& & & \multicolumn{3}{c}{$\Delta_{\text{boot}}$ (primary)} & \multicolumn{3}{c}{$\Delta_{\text{wc}}$ (retired)} & & \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Model & Dataset & Probe & Est. & Null mean & $p_{2}$ & Est. & Null mean & $p_{\uparrow}$ & Sel.\ AUROC & Flags \\
\midrule
llama3.1-8b & fever & linear & $+0.0007$ & $+0.0065$ & $0.001$ & $+0.0000$ & $+0.0137$ & $1.000$ & $0.9548$ & D \\
llama3.1-8b & fever & mlp & $+0.0051$ & $+0.0080$ & $0.004$ & $+0.0176$ & $+0.0197$ & $0.616$ & $0.9561$ & --- \\
llama3.1-8b & halueval\_qa & linear & $+0.0024$ & $+0.0049$ & $0.001$ & $+0.0081$ & $+0.0171$ & $1.000$ & $0.9983$ & S \\
llama3.1-8b & halueval\_qa & mlp & $+0.0014$ & $+0.0032$ & $0.001$ & $+0.0027$ & $+0.0094$ & $1.000$ & $0.9979$ & S \\
llama3.1-8b & synthetic & linear & $+0.0014$ & $+0.0017$ & $0.079$ & $+0.0012$ & $+0.0021$ & $0.982$ & $1.0000$ & S \\
llama3.1-8b & synthetic & mlp & $+0.0008$ & $+0.0012$ & $0.001$ & $+0.0017$ & $+0.0027$ & $1.000$ & $1.0000$ & S \\
llama3.1-8b & truthfulqa & linear & $-0.0000$ & $+0.0030$ & $0.092$ & $+0.0000$ & $+0.0080$ & $1.000$ & $0.9137$ & DB \\
llama3.1-8b & truthfulqa & mlp & $+0.0016$ & $+0.0031$ & $0.191$ & $+0.0040$ & $+0.0075$ & $1.000$ & $0.9139$ & --- \\
mistral-7b & fever & linear & $+0.0005$ & $+0.0009$ & $0.058$ & $+0.0000$ & $+0.0000$ & $1.000$ & $0.9598$ & D \\
mistral-7b & fever & mlp & $+0.0015$ & $+0.0075$ & $0.001$ & $+0.0055$ & $+0.0144$ & $1.000$ & $0.9597$ & --- \\
mistral-7b & halueval\_qa & linear & $+0.0019$ & $+0.0031$ & $0.004$ & $+0.0047$ & $+0.0122$ & $1.000$ & $0.9986$ & S \\
mistral-7b & halueval\_qa & mlp & $+0.0019$ & $+0.0025$ & $0.088$ & $+0.0058$ & $+0.0083$ & $0.868$ & $0.9987$ & S \\
mistral-7b & synthetic & linear & $+0.0014$ & $+0.0022$ & $0.001$ & $+0.0044$ & $+0.0076$ & $0.988$ & $0.9992$ & S \\
mistral-7b & synthetic & mlp & $+0.0014$ & $+0.0034$ & $0.001$ & $+0.0033$ & $+0.0110$ & $0.895$ & $0.9992$ & S \\
mistral-7b & truthfulqa & linear & $+0.0004$ & $+0.0065$ & $0.001$ & $+0.0000$ & $+0.0126$ & $1.000$ & $0.9039$ & D \\
mistral-7b & truthfulqa & mlp & $+0.0012$ & $+0.0099$ & $0.001$ & $+0.0000$ & $+0.0293$ & $1.000$ & $0.9081$ & D \\
qwen2.5-7b & fever & linear & $+0.0122$ & $+0.0133$ & $0.438$ & $+0.0347$ & $+0.0318$ & $0.277$ & $0.9265$ & --- \\
qwen2.5-7b & fever & mlp & $+0.0076$ & $+0.0091$ & $0.107$ & $+0.0165$ & $+0.0193$ & $0.897$ & $0.9310$ & --- \\
qwen2.5-7b & halueval\_qa & linear & $+0.0017$ & $+0.0038$ & $0.001$ & $+0.0044$ & $+0.0113$ & $1.000$ & $0.9971$ & S \\
qwen2.5-7b & halueval\_qa & mlp & $+0.0017$ & $+0.0038$ & $0.001$ & $+0.0053$ & $+0.0137$ & $1.000$ & $0.9980$ & S \\
qwen2.5-7b & synthetic & linear & $+0.0005$ & $+0.0007$ & $0.421$ & $+0.0002$ & $+0.0002$ & $1.000$ & $1.0000$ & S \\
qwen2.5-7b & synthetic & mlp & $+0.0026$ & $+0.0037$ & $0.002$ & $+0.0087$ & $+0.0121$ & $1.000$ & $1.0000$ & S \\
qwen2.5-7b & truthfulqa & linear & $+0.0000$ & $+0.0104$ & $0.002$ & $+0.0000$ & $+0.0271$ & $1.000$ & $0.9145$ & DB \\
qwen2.5-7b & truthfulqa & mlp & $+0.0007$ & $+0.0130$ & $0.002$ & $+0.0000$ & $+0.0349$ & $1.000$ & $0.9250$ & D \\
\bottomrule
\end{tabular}
\caption{All 24 Case Study 4 cells, with \emph{each estimator's own}
permutation null beside it. $\Delta_{\text{boot}}$ is computed by exact
enumeration of all $3^3=27$ resamples; it is $\geq0$ for any data
(Theorem~\ref{thm:boot-nonneg}), exactly as $\Delta_{\text{wc}}$ is
(Theorem~\ref{thm:wc-nonneg}), so neither column's uniform sign is evidence and
no signed-rank test against zero is reported for either. $p_{2}$ is two-sided;
$p_{\uparrow}$ is upper-tail. ``Sel.\ AUROC'' is the naive selection AUROC.
Flags: \textbf{D} = rotation estimator degenerate (all three rotations picked
the same layer); \textbf{B} = $\Delta_{\text{boot}}$ degenerate (grand-mean
argmax bootstrap-stable); \textbf{S} = ceiling-saturated (selected layer at
AUROC $\geq0.975$). \textbf{No cell of either estimator reaches $p<0.05$ in the
upper tail} (minimum $p_{\uparrow}=0.277$, $\Delta_{\text{wc}}$; $1.000$
median). Two-sided, $16$ of $24$ $\Delta_{\text{boot}}$ cells and $11$ of $24$
$\Delta_{\text{wc}}$ cells do, all of them \emph{below} their null --- §4.4
explains why a conservative estimator on a stable argmax is expected to sit
below this particular null. Until a recent revision this table carried a single
null-mean/$p$ pair, printed beside $\Delta_{\text{boot}}$ and naturally read as
belonging to it; they were $\Delta_{\text{wc}}$'s in $24$ of $24$ rows, whose
null averages $2.60\times$ $\Delta_{\text{boot}}$'s ($+0.0136$ against
$+0.0052$), so the displayed reference was systematically the larger of the two.}
\label{tab:cs4-cells}
\end{table}
```

---

## Appendix B.9 (former) — Case Study 4: ceiling composition of the 24 cells

Removed from the paper as $\approx$85\% duplicative of §4.4's ceiling-composition paragraph, which states the $+0.0016$/$+0.0026$ split and the $1.5$--$5.5\times$ multiplier range. The by-dataset breakdown and the stricter-saturation reading are here.

```latex
\subsection{Case Study 4: ceiling composition of the 24 cells}

\textbf{Ceiling composition of the 24 cells.} These cells are not comparable
to each other, and reporting only their pooled mean hides that. In $12$ of $24$
the \emph{selected} layer already sits at AUROC $\geq0.975$ (all six
\texttt{halueval\_qa} cells and all six \texttt{synthetic} cells), leaving
almost no headroom for a selection effect to be visible; the remaining $12$
(\texttt{fever}, \texttt{truthfulqa}) are genuinely non-saturated.

\textbf{Under $\Delta_{\text{boot}}$, the estimator this paper reports}, the
split is ceiling-saturated $+0.0016$ (BCa CI $[+0.0013,+0.0019]$, $12$ cells)
against non-saturated $+0.0026$ (CI $[+0.0011,+0.0056]$, $12$ cells): a
$\mathbf{1.6\times}$ difference. No sub-selection is needed, because
$\Delta_{\text{boot}}$ is defined on all $24$ cells. Under
$\Delta_{\text{boot}}^{\text{std}}$ the same split is $+0.0003$ against
$+0.0015$, a $5.5\times$ difference --- the direction is robust across
estimators, the multiplier is not, ranging over $1.5$--$5.5\times$ depending on
which of the three one uses, and we quote a range rather than any single
figure.

\textbf{An earlier revision reported this decomposition on the retired
$\Delta_{\text{wc}}$ and did not say so.} Those numbers were:
ceiling-saturated $+0.0042$ (CI $[+0.0029,+0.0057]$, $12$ cells, \emph{none}
degenerate); non-saturated $+0.0065$ (CI $[+0.0022,+0.0151]$, $p=0.0625$)
across all $12$ of its cells, but $+0.0157$ (CI $[+0.0074,+0.0276]$,
$p=0.0625$) across the $5$ the rotation estimator can actually see --- from
which it drew a ``$3.7\times$ difference''. That figure is superseded on two
counts: it is computed on an estimator the paper no longer reports, and its
numerator comes from a $5$-cell sub-selection its denominator does not undergo.
The comparable all-cell $\Delta_{\text{wc}}$ figure is $1.5\times$, in line
with the $1.6\times$ from $\Delta_{\text{boot}}$. By dataset (under the retired
$\Delta_{\text{wc}}$, for continuity with earlier revisions):
\texttt{fever} $+0.0124$, \texttt{halueval\_qa} $+0.0052$, \texttt{synthetic}
$+0.0033$, \texttt{truthfulqa} $+0.0007$ (the last is $0.0000$ in four of its
six cells for the degeneracy reason above, and should not be read as a dataset
effect). Under a stricter saturation reading (every layer, not just the
selected one, above $0.975$ -- true for the six \texttt{halueval\_qa} cells)
the split is $+0.0052$ (6 cells) versus $+0.0054$ (18 cells). The mechanism's
severity is small on every slicing, and larger where there is room for it to
show, consistent with the operating-point relationship in §5 --- though §5 also
reports the limits of how far that relationship transports.
```

---

## Appendix B.10 (former) — Mechanism 5: sample-size dependence of the F1 and accuracy gaps

Removed from the paper as $\approx$90\% duplicative of §4.5's sample-size paragraph and its table.

```latex
\subsection{Mechanism 5: sample-size dependence of the F1 and accuracy gaps}

\textbf{These gaps are a function of sample size, and the sample sizes were
fixed and unstated.} All five cells above run at $N=700$, hence
$n_{\text{test}}=140$ (LEAKY's selection set) and $n_{\text{val}}=112$
(HONEST's). That matters because a threshold chosen by argmax over an
$81$-point grid is a winner's curse over a \emph{noisy} criterion, and the
noise scales with $1/\sqrt{n}$ --- so the magnitude of this gap is not a
property of the mechanism alone. \textbf{It also means the phrase
``MultiHaluDet's own reported regime'' above matches their operating point and
\emph{not} their sample size}, which their released configuration does not
pin: \texttt{config.py} fixes only \texttt{test\_size}$=0.20$, while
\texttt{src/data/loader.py} defaults \texttt{max\_samples}$=10000$, so their
$n_{\text{test}}$ could be more than an order of magnitude larger than $140$.
Holding the operating point at $0.985$ and scaling $N$ ($200$ seeds/cell) gives
an F1 gap, at the repo-faithful Youden threshold (\texttt{code/62}), of
$+0.0225$ at $n_{\text{test}}=140$, $+0.0123$ at $350$, $+0.0066$ at $700$ and
$+0.0036$ at $2000$ --- a \textbf{$6.3\times$ decline}, close to the
$3.8\times$ a pure $1/\sqrt{n}$ account would give, so most but not all of it
is selection-noise scaling. The same sweep at the F1-argmax convention
(\texttt{code/46}, \texttt{SWEEP=N}) gives
$+0.0212$/$+0.0111$/$+0.0067$/$+0.0034$ and a $6.2\times$ decline: the
threshold-rule correction of §4.5 changes the \emph{level} of this sweep but
not its \emph{shape}, which is worth stating because the two questions are
independent. Accuracy behaves the same way ($+0.0215$ to $+0.0036$). Every cell
remains overwhelmingly significant ($p<10^{-31}$ throughout under both rules,
CIs excluding zero), so the mechanism does not vanish --- but \textbf{the
$+0.022$ to $+0.052$ range this paper quotes for Mechanism 5 is an estimate at
$n_{\text{test}}=140$, and a pipeline reporting at a test-set size five times
larger should expect roughly a third of it.} We report the $n_{\text{test}}=140$
figures as the headline only because they are the ones measured at all five
operating points; they are not a size-matched estimate of MultiHaluDet's own
inflation, and we do not present them as one.
```

---

## Appendix C.2 (former, second half) — the joint fit's estimator sensitivity, and the $\ln K$ qualification restated

Relocated from Appendix C. The retained Appendix C states the column-deletion range ($b=0.09$--$0.21$, $c=-2.80$ to $-3.17$) and the estimator-sensitivity headline ($b=0.293$ under log-space OLS against $0.166$ under Gauss-Newton; $c$ within $[-3.17,-2.38]$ over both). The full discussion:

```latex
\textbf{$b$ is more sensitive to the choice of estimator than to column
deletion, and the quoted range does not cover that.} An independent review
observed that the $0.09$--$0.21$ band above is a column-deletion range under
\emph{one} fitting method. \texttt{code/57} fits the multiplicative form by
Gauss-Newton least squares on the raw gaps, which is the right choice for this
paper's purpose --- it minimizes squared error in the same units as the additive
model, so $R^2=0.976$ is commensurable with the additive $0.688$ at equal
parameter count --- but it is not the only standard estimator for that form.
Refitting the identical functional form on the identical $20$ cells by ordinary
least squares in log space (\texttt{code/65},
\texttt{results/joint\_surface\_estimator\_sensitivity.json}) gives
$\mathbf{b=0.293}$ against the reported $0.166$: a shift of $0.127$, larger
than the $0.118$ span the four column deletions produce, and \emph{outside} the
$0.09$--$0.21$ range this paper quotes. Over both estimators and all deletions
$b$ ranges $0.01$ to $0.43$. The two estimators are not in conflict; they
minimize absolute and relative error respectively, and on a grid whose cell
means span two orders of magnitude ($+0.00012$ to $+0.01100$) they weight it
very differently, log-space OLS giving the tiny high-AUROC cells the same
leverage as the large low-AUROC ones. This is a fitted-parameter stability
caveat, not a validity failure of the surface --- but it means $b$ should not be
read as a measured exponent at all, and no conclusion in this paper rests on
its value.

\textbf{The coefficient the paper's conclusions do rest on survives both.} The
operating-point coefficient $c$ is $-2.97$ under Gauss-Newton NLS and $-2.43$
under log-space OLS, and over both estimators and all four column deletions it
stays within $[-3.17,-2.38]$ --- a span of $29\%$ of its own mean, against
$191\%$ for $b$, with the same sign and order of magnitude throughout. The
per-subset coefficients under both estimators are in
\texttt{results/joint\_surface\_estimator\_sensitivity.json}. §5.3 and
§5.4's claims are about the operating point, so the contrast is the relevant
one: the axis the paper leans on is stable under a change of estimator, and the
axis it already calls second-order is not.

\textbf{This forces a qualification of one of this paper's own headline
numbers.} The $K$-relationship of §5.3 is
measured along a single row of this grid, at $\text{AUROC}_0=0.80$. Across the
grid as a whole, $\ln K$ \emph{alone} explains essentially none of the variance
($R^2=0.038$, and at LOO $-0.192$ it is worse than predicting the mean). The
$K$-law is therefore real \emph{within} an operating point and not a
description of severity in general --- the table shows $K$ spans of
$1.8\times$/$4.2\times$/$7.1\times$/$4.3\times$/$1.6\times$ (collapsing to
$1.1$--$1.4\times$ with the $K=15$ column removed) at
$\text{AUROC}_0=0.70$/$0.80$/$0.90$/$0.95$/$0.985$, i.e.\ a real effect where
there is headroom and essentially none at $0.985$, every cell there sitting
between $+0.00012$ and $+0.00019$ regardless of whether $15$ or $405$
candidates were tried. §5.5 notes two caveats on those spans that we repeat
here because they are easy to miss: the span is \emph{not} monotone in
headroom, and each of the three largest is set at its low end by the $K=15$
column, the least-contrastive column in the grid --- excluding it, the spans
are $1.1$--$1.4\times$. This
qualification is \emph{stronger} on the corrected grid than on the degenerate
one it replaces ($R^2=0.038$ against $0.114$), so unlike the interaction claim
it does not depend on the removed cells. Candidate count is a second-order
modifier of a first-order operating-point effect, not a co-equal axis.
```

---

## Appendix C.3 (former) — the joint fit's out-of-sample check

Relocated from Appendix C; §5.5 states the conclusion (two of the five held-out Sweep A cells are degenerate, and the seed-count mismatch makes the ordering uninterpretable, so this harness does not currently support an out-of-sample test).

```latex
\subsection{The joint fit's out-of-sample check}

\textbf{The out-of-sample check is also
weaker than the previous version of this paper presented it as.} The intended
check predicts the five Sweep A cells at $K\in\{5,10,25,75,225\}$, which are
not in the grid. Two things undercut it, both found by instrumenting rather
than by argument. First, \emph{two of those five held-out cells are themselves
degenerate}: at $K=5$ the LEAKY and CLEAN\_MATCHED arms are bitwise identical
in $90\%$ of folds and at $K=10$ in $65\%$, for exactly the reason $K=3$ was
dropped, so they are not valid targets. Second, Sweep A runs $200$ seeds while
this grid runs $100$, and at these effect sizes that matters more than the
models do --- the same $(K{=}225,\text{AUROC}_0{=}0.80)$ configuration gives
$+0.00522$ at $n=200$ and $+0.00393$ at $n=100$, a $33\%$ difference. For the
record the numbers are $R^2=0.279$ (additive), $0.795$ (interaction) and
$0.634$ (multiplicative) over all five, and $0.365$, $0.567$ and $-0.249$ over
the three non-degenerate ones. We read nothing into the ordering: the
interaction model scoring highest out-of-sample while being non-significant
in-sample is exactly the pattern this seed-count mismatch would produce by
chance, and treating it as support for the interaction would repeat, in a
subtler form, the error §5.5 already corrects. This harness does not
currently support an out-of-sample test of the joint surface, and building one
would mean re-running Sweep A at matched seed counts on non-degenerate $K$
values.
```

---

## Appendix C.4 (former) — the operating-point axis in variance-stabilized coordinates: discussion prose

Relocated from Appendix C. The five-row per-cell table itself is retained in the paper; this is the surrounding discussion.

```latex
Three things are visible here that the summary hides. The \emph{gap SD} column
is the compression account's own prediction made explicit: dispersion really
does collapse across this axis, by $18.1\times$, so compression is a large real
component of the raw decline rather than an absent one --- but the mean's
decline outruns it, which is the sense in which the account fails as a
\emph{complete} explanation. The \emph{$0.90$-versus-$0.95$ inversion} shows up
in all three stabilized columns and in none of the raw ones; it is small and we
do not read a mechanism into it, but it is the reason §5.3 no longer calls the
relationship monotone in a stabilized coordinate. And the rank column falls
almost to nothing at $0.985$ ($+0.021$; $49$ of $100$ seeds positive against
$47$ negative), which is a coordinate-free way of saying what the raw
$p=0.219$ at that cell already said: at the ceiling this contrast is close to
a coin flip.
```

---

## Appendix D (former) — Extended Detail for the Checklist and the Automated Scanner

Removed from the paper in full: §6.2 and §2 already state everything load-bearing here (the Kapoor \& Narayanan / REFORMS mapping, the blinded second-rater protocol and its $7/7$ agreement with $\kappa$ undefined, the two specific scanner-failure causes, and the single non-regex data point). §6 carries a one-line pointer to this file.

```latex
\section{Extended Detail for the Checklist and the Automated Scanner}

\subsection{Benchmarking against existing leakage taxonomies and checklists}

\textbf{Kapoor \& Narayanan's cross-field leakage taxonomy.} The closest prior
art to this paper's central move is Kapoor \& Narayanan (2023), who survey
leakage across 17 scientific fields and 294 papers and propose an 8-type
taxonomy: L1.1 (no test set), L1.2 (pre-processing computed on train+test
combined), L1.3 (feature selection on train+test), L1.4 (duplicates across
the split), L2 (illegitimate/proxy features), L3.1 (temporal leakage), L3.2
(non-independence between train and test samples), and L3.3 (test
distribution not matching the distribution of scientific interest). Mapping
this paper's five mechanisms onto it: Mechanisms 4 and 5 (test-set-driven
best-layer and threshold selection) are both direct cases of \textbf{L1.3}
--- the test set is reused for a modeling decision and then for the reported
metric, leaving no genuinely held-out data behind that decision. (An earlier
draft of this paper mapped both to L1.1, ``no test set'' -- that was wrong:
these pipelines do construct a test split and do train on a disjoint set;
what they do illegitimately is \emph{select} using the test split, which is
precisely L1.3's scope, not L1.1's.) Mechanism 1 (full-dataset
fit-then-score feature leakage) is an instance of L1.2, generalized from a
preprocessing statistic to an entire fitted feature-extraction model, and
Mechanism 3 (per-fold checkpoint selection) is a narrower-bandwidth variant
of the same pattern, restricted to which training iterate is kept. Mechanism
2 (CV-based layer/hyperparameter selection optimism --- Varma \& Simon's
problem, and the subject of Cawley \& Talbot's (2010) canonical treatment)
does not map cleanly onto any single one of the eight leaf types:
it is closest in spirit to L1.2/L1.3 but is really a distinct,
model-selection-specific pattern their taxonomy does not name explicitly.
The asymmetry to read off this is about the taxonomies, not about the
mechanism: three of this paper's five mechanisms are instances of an existing
general \emph{leakage} taxonomy, while Case Study 2's pattern is not one of its
eight named types even though the \emph{model-selection} literature has
characterized it since 2010. So the pattern is not unnamed in the field at
large; it is unnamed by the two leakage instruments this literature is most
likely to reach for. That is consistent with this paper's claim that
hidden-state probing's two-stage structure surfaces at least one leakage
pattern general
cross-field taxonomies do not yet enumerate by name.

\textbf{Benchmarking against REFORMS and the ML Reproducibility Checklist.}
REFORMS (Kapoor et al. 2024, \emph{Science Advances}) is a 32-item, 8-module
consensus checklist for ML-based science: study goals (3 items),
computational reproducibility (5), data quality (7), data preprocessing (3),
modeling (6), data leakage (3), metrics and uncertainty quantification (3),
and generalizability and limitations (2). Its dedicated data-leakage module
covers exactly three questions -- train/test separation, dependencies
between train and test instances, and feature legitimacy -- all three
subsumed by this paper's five-mechanism taxonomy; and Mechanism 2 again
falls outside its scope, for the same reason it falls outside Kapoor \&
Narayanan's leaf types, since that module treats train/test separation and
feature legitimacy as the leakage surface rather than iterative
model/architecture selection against a validation signal. REFORMS' other
seven modules cover reporting practices this paper does not audit
(study-goal framing, uncertainty quantification beyond the specific severity
estimates here, external-validity discussion), so neither instrument is a
superset of the other. The NeurIPS Machine Learning Reproducibility
Checklist (Pineau et al. 2021, \emph{JMLR}) is reporting-practice-oriented
rather than leakage-specific: its four sections ask whether code,
hyperparameters, dataset statistics, compute budgets, and multi-seed
variance are disclosed, none of which individually detects a leakage bug.
This paper's own reproducibility practice (full per-seed result arrays,
vendored external repositories, BCa bootstrap/permutation tests on every
severity gap) satisfies the bulk of the Pineau checklist's disclosure items
as a byproduct of the statistical rigor the severity claims already require
--- but that checklist would not, on its own, have caught any of the sixteen
issues this paper's own Appendix A documents finding and fixing in its own
instrument. Reporting-practice checklists and leakage-detection checklists
are complementary, not interchangeable, and a paper can satisfy the former
while still committing the latter.

\subsection{The blinded second-rater protocol}

\textbf{How the hits were classified, and the reliability limits of that
classification.} A further independent review pointed out that the
true-positive/false-positive labels were, as originally produced, a single
rater's manual reading; that the rater was this paper's author; and that the
rater already knew where both real bugs were. That is three distinct threats at
once --- no inter-rater reliability, no blinding, and an interested rater ---
and none was disclosed. We state them, and we did something about the first
two: an independent second reading of the same 7 sites was obtained under a
protocol that withheld this paper's draft, its checklist, and the location of
the two known bugs, giving the second rater only the flagged
file/line/snippet and the vendored third-party source. \textbf{The second rater
was a language model, not a second human}, which bounds what the agreement is
worth: it checks that the classification survives a reader with no stake in the
result and no knowledge of the answer, not that two domain experts converge.
\textbf{Agreement was 7 of 7 --- the second rater classified every one of the
seven as a false positive, with the same provenance argument}
(\texttt{haloscope} selects on \texttt{gt\_label\_val} from
\texttt{wild\_q\_indices2}, provably disjoint from the test indices;
\texttt{embed\_viz.py} fits a projection whose output reaches only a matplotlib
call). Percent agreement is therefore $100\%$; \emph{Cohen's $\kappa$ is
undefined here}, because both raters used a single category, and we report the
raw agreement rather than a chance-corrected coefficient that this degenerate
marginal makes uncomputable. Two-rater agreement on seven non-independent sites
from two repositories is a weak reliability check, not a strong one.

Two further outcomes of that blind read cut in different directions.
\emph{Supporting the paper:} asked separately, and without being told where any
bug was, to read the two case-study repositories for leakage, the second rater
independently located both known true positives ---
\texttt{run\_pipeline.py:139-141} (Mechanism 5) and \texttt{saplma.py:35-41}
together with its duplicate at \texttt{src/analysis/metrics.py:85-91}
(Mechanism 4) --- direct evidence that the two false negatives (D.3) are failures of the \emph{scanner} rather than artifacts of the authors'
privileged knowledge of where to look. \emph{Cutting against it:} the same
rater declined to count MultiHaluDet's \texttt{trainer.py} checkpoint selection
(Mechanism 3) as qualifying, on the ground that the leaked fold's features go
on to train the meta-learner against \texttt{y\_train} only, so the reported
test metric is not inflated \emph{directly}; the rater called it ``a mild
stacking impurity'' instead. We disagree --- §4.3 measures a small but real
and, at two of four capacities, significant inflation from exactly this path
--- but the disagreement is genuine, it is about whether a site clears a binary
threshold rather than about what the code does, and it is precisely the kind of
judgement a single non-blinded rater cannot surface. It is also consistent with
our own finding that this mechanism's magnitude is at the low end of the band.

The blind read also surfaced two sites in
\texttt{HallucinationPatternDetection} that this paper had \emph{not}
previously reported and that we verified directly against the pinned commit.
(i) \texttt{scripts/06\_analyze\_attention.py:33-52} computes a per-layer AUROC
over the \emph{entire labelled set with no split at all}, takes
\texttt{best = max(per\_layer, key=...auroc)}, and writes \texttt{best\_auroc}
into the metrics JSON that \texttt{scripts/07\_aggregate\_results.py:74-86}
folds into the headline table --- Mechanism 4 in a purer form than the
\texttt{saplma.py} site, since there is no held-out split behind it at all.
(ii) \texttt{src/analysis/metrics.py:57-65} (\texttt{score\_to\_metrics}) picks
a Youden threshold by \texttt{np.argmax(tpr-fpr)} on the same labels it then
scores \texttt{accuracy\_at\_best\_thr} and \texttt{f1\_at\_best\_thr} against,
and \texttt{07\_aggregate\_results.py:85-86} reports both --- Mechanism 5, in
the \emph{second} audited repository. \textbf{So Mechanism 5 is not specific to
MultiHaluDet}: the same test-label-tuned operating-point selection occurs
independently in both externally published pipelines this paper audits, which
we did not know when §4.5 was written. Neither new site changes any severity
number here --- both are additional instances of mechanisms already quantified
--- and neither is included in §6.1's 7-hit count, which is the scanner's output, not a human audit's.

\subsection{Why the scanner fails: two specific causes}

\textbf{The scanner does not work, for two specific reasons.} It produced 7 raw
hits, concentrated in 2 repos, and manual reading showed all 7 are false
positives: 3 flag \texttt{HallucinationPatternDetection}'s
\texttt{embed\_viz.py} for ``fit-then-score'' -- these are t-SNE/UMAP/PCA calls
used only for plotting; 4 flag \texttt{haloscope}'s layer/threshold selection --
reading the code shows selection correctly uses a separate validation split,
the opposite of leakage, and a positive data point for that NeurIPS 2024
pipeline. More tellingly, it missed both known true positives, and \textbf{both
misses have identifiable causes in this scanner's own construction.} (i) Case
Study 3's selection site (\texttt{trainer.py}, lines 137-144) binds
\texttt{best\_auc}/\texttt{best\_model}; the scanner's
\texttt{CHECKPOINT\_KEYWORDS} list (\texttt{code/04}, line 55) contains only
\texttt{best\_epoch}, \texttt{best\_checkpoint} and \texttt{checkpoint}. This is
a plain keyword-list gap -- and a telling one, since we had already read those
exact two token names while manually diagnosing the bug and still did not add
them. (ii) Case Study 4's selection site (\texttt{saplma.py}, lines 32-41)
\emph{is} matched by \texttt{ARGMAX\_KEYWORDS}, which already contains
\texttt{best\_layer}; what fails is the scanner's design choice to require an
argmax token and a \texttt{test}-named variable \emph{on the same line}, and,
before that, in the same file. That file contains no \texttt{test}-named
identifier anywhere -- the test split lives one call away, in
\texttt{probes.py::train\_layerwise\_probes} -- so a same-file, same-line
conjunction cannot fire on it however the keyword lists are extended. Zero true
positives among the 5 newly-scanned repos, two false negatives on the two known
positives, seven-for-seven false positives on its raw hits: this particular
scanner does not work. We deliberately do not read that as evidence that the
leakage class is inherently resistant to static detection. Cause (i) would be
fixed by a longer keyword list, and cause (ii) points at a specific, tractable
design requirement -- the argmax and the test-derived metric must be connected
across statements and across function boundaries, i.e.\ by a dataflow analysis
rather than line matching. What this exercise establishes is narrower and more
useful: \emph{line-level regex matching} is the wrong tool here, because every
one of these mechanisms is a relationship between a selection statement and the
provenance of the quantity it selects on, and that relationship is routinely
split across lines, functions, and files.

\subsection{One data point on a non-regex approach}

\textbf{One data point on what a non-regex approach can do, and why we do not
generalize from it either.} The three obvious alternatives to line-level regex
are an AST pass, an interprocedural dataflow analysis, and a language-model
read of the source. We built none of the first two. The blinded second-rater
protocol above is, incidentally, an instance of the third --- a language model
reading the vendored source with no access to this paper --- and it located
both true positives in the two known-buggy repositories, plus the two additional sites reported in D.2, where the regex scanner found none. That is
one run, on two repositories already known to contain bugs, with no
false-positive rate measured on the five presumed-clean repos and no protocol
for what it should be asked or how its output should be adjudicated. It is
enough to say the failure above is \emph{this scanner's}, not the failure of
automated detection as a category --- which is all §6 claims --- and nowhere
near enough to say a language-model scanner works. Measuring that properly, on
a pre-registered corpus with both classes present, is the same follow-on work.
```

---

## Appendix C (Extended Detail for the Severity Surface) — relocated in full

Relocated in a later compression pass (43 pages to 42, then this final relocation). §5.3--§5.5 state every verdict below in the main text; the tables and derivations here are the evidence behind those verdicts, not required to trust them. The full original appendix:

```latex
\subsection{The degenerate $K=3$ column, and the degeneracy gate that replaced it}

§5.5 states this outcome; here is the evidence behind it. Refitting the
original $K\in\{3,15,45,135\}$ grid without its $K=3$ column moves the
interaction test from $p=0.027$ to $p=0.378$, while dropping any \emph{other}
$K$ column leaves it significant ($p=0.043$, $0.041$, $0.044$). At three epochs
there are too few candidates for the two selection rules to diverge: both
almost always land on the last epoch and CLEAN\_MATCHED's retrain reproduces
LEAKY's trajectory step for step, leaving the two models \emph{bitwise
identical} in $81$--$100\%$ of folds, so the ``gap'' is $0.00000$ by
construction --- two of that grid's twenty cells, $K{=}3$ at
$\text{AUROC}_0=0.95$ and $0.985$, are exactly $0.00000$, the signature. Every
cell now carries a degeneracy record --- the bitwise-identical fraction, the
largest parameter difference, and how often each arm's argmax lands on the last
epoch --- with the fit gated on it and refusing to run if any surviving cell
exceeds $50\%$. \textbf{The gate is not a clean bill of health}: degeneracy
decays with $K$ rather than switching off, and the per-column
bitwise-identical fraction across
$\text{AUROC}_0\in\{0.70,0.80,0.90,0.95,0.985\}$ is

\begin{center}\small
\begin{tabular}{lccccc}
\toprule
 & $0.70$ & $0.80$ & $0.90$ & $0.95$ & $0.985$ \\
\midrule
$K=15$  & $0.244$ & $0.362$ & $0.420$ & $0.406$ & $0.312$ \\
$K=45$  & $0.024$ & $0.026$ & $0.024$ & $0.054$ & $0.038$ \\
$K=135$ & $0.014$ & $0.018$ & $0.022$ & $0.054$ & $0.034$ \\
$K=405$ & $0.010$ & $0.016$ & $0.022$ & $0.054$ & $0.032$ \\
\bottomrule
\end{tabular}
\end{center}

\noindent --- $1$--$5\%$ for $K\in\{45,135,405\}$ but $24$--$42\%$ for $K=15$,
worst at $(K{=}15,\text{AUROC}_0{=}0.90)$ --- the cell producing §5.5's
$7.1\times$. $K=15$ therefore passes the gate but carries the least contrast,
and every §5.5 conclusion is accordingly refit five ways, reporting the range
rather than a single number. \emph{One disclosure on this grid's consistency:}
$(K{=}45,\text{AUROC}_0{=}0.80)$ has two legitimate values, because it runs at
two sample sizes --- Sweep A at $n=200$ seeds gives $+0.004155$, Sweeps B/C/D
and this grid at $n=100$ give $+0.003649$ --- differing only by Monte Carlo
error over seeds $100$--$199$. We state the $n=200$ value rather than let a
reader find two numbers for one cell and infer a drift.

\subsection{The joint fit's stability under column deletion and change of estimator}

Deleting each $K$ column in turn, the interaction stays non-significant every
time ($p=0.18$, $0.22$, $0.38$, $0.64$) and the multiplicative fit stays strong
($R^2=0.967$--$0.994$), with a stable operating-point coefficient ($c=-2.80$ to
$-3.17$) and an unstable candidate-count exponent ($b=0.09$ to $0.21$, lowest
when $K=15$ is removed). \textbf{$b$ is more sensitive to the choice of
estimator than to column deletion.} \texttt{code/57} fits by Gauss-Newton least
squares on the raw gaps, so $R^2=0.976$ is commensurable with the additive
$0.688$ at equal parameter count; refitting the identical form on the identical
$20$ cells by ordinary least squares in log space (\texttt{code/65}) gives
$\mathbf{b=0.293}$ against the reported $0.166$ --- a shift larger than the
$0.118$ span the four column deletions produce, and outside the $0.09$--$0.21$
range. Over both estimators and all deletions $b$ ranges $0.01$ to $0.43$. The
two estimators minimize absolute and relative error respectively and weight a
grid spanning two orders of magnitude ($+0.00012$ to $+0.01100$) very
differently; this is a fitted-parameter stability caveat rather than a validity
failure, but $b$ should not be read as a measured exponent and no conclusion
rests on its value. \textbf{The coefficient the conclusions do rest on survives
both:} $c$ is $-2.97$ under Gauss-Newton and $-2.43$ under log-space OLS,
staying within $[-3.17,-2.38]$ over both estimators and all four deletions ---
a span of $29\%$ of its own mean against $191\%$ for $b$. \emph{One caveat on
interval precision, from §5.4:} that section's transport-check reference cell
appears in three shipped result files with a bit-identical point estimate, but
three separate BCa bootstraps of the same per-seed vector put its lower
endpoint at $+0.0000623$, $+0.0000653$ and $+0.0000796$, a $28\%$ spread. All
three exclude zero, so no conclusion changes, but that endpoint should be read
as ``small and positive'' rather than as $6.53\times10^{-5}$.

\subsection{The operating-point axis in variance-stabilized coordinates, cell by cell}

Computed on the same $100$ paired LEAKY-minus-CLEAN\_MATCHED contrasts per cell
that produce the raw column, shipped per seed by \texttt{code/63}
(\texttt{code/69}).

\begin{center}\small
\begin{tabular}{lrrrrr}
\toprule
AUROC$_0$ & raw gap & gap SD & probit gap & Cohen's $d$ & sign stat.\ $r$ \\
\midrule
0.70  & $+0.00932$ & $0.02486$ & $+0.0246$ & $+0.375$ & $+0.313$ \\
0.80  & $+0.00365$ & $0.00987$ & $+0.0116$ & $+0.370$ & $+0.360$ \\
0.90  & $+0.00109$ & $0.00504$ & $+0.0050$ & $+0.216$ & $+0.091$ \\
0.95  & $+0.00065$ & $0.00295$ & $+0.0053$ & $+0.222$ & $+0.131$ \\
0.985 & $+0.00019$ & $0.00138$ & $+0.0034$ & $+0.139$ & $+0.021$ \\
\midrule
decline & --- & $18.1\times$ & $7.2\times$ & $2.7\times$ & $15.0\times$ \\
\bottomrule
\end{tabular}
\end{center}

\noindent The \emph{gap SD} column makes the compression account's own
prediction explicit: dispersion really does collapse across this axis, by
$18.1\times$, so compression is a large real component of the raw decline
rather than an absent one --- but the mean's decline outruns it, which is the
sense in which the account fails as a \emph{complete} explanation. The
$0.90$-versus-$0.95$ inversion shows up in all three stabilized columns and in
none of the raw ones, and at $0.985$ the sign statistic falls almost to nothing
($+0.021$; $49$ of $100$ seeds positive against $47$ negative), a
coordinate-free way of saying what the raw $p=0.219$ at that cell already said.

\subsection{Is the $K$-regularity's functional form identified? No}

§5.3 states this verdict; here is the evidence behind it. \texttt{code/63} now
ships per-seed data, re-running \texttt{code/47}'s own \texttt{run\_one\_seed}
at the same seeds for all ten Sweep A cells ($200$ seeds each) and all five
Sweep C cells ($100$ each); every recomputed cell mean reproduces the shipped
JSON to full float64, asserted at runtime, so the artifact adds resolution and
changes no published number. \textbf{The residuals sit below the noise:} the
mean per-cell Monte-Carlo standard error is $7.72\times10^{-4}$ against an RMS
fit residual of $2.63\times10^{-4}$, so an $R^2$ computed in that regime
measures how \emph{monotone} six points are, not how \emph{logarithmic}.

\begin{center}\small
\begin{tabular}{lrrr}
\toprule
Form & $R^2$ & AICc & LOO $R^2$ \\
\midrule
$a(1-e^{-K/\tau})$ saturating & $\mathbf{0.9914}$ & $\mathbf{-92.75}$ & $\mathbf{0.981}$ \\
$a+b/K$ & $0.9787$ & $-87.28$ & $0.885$ \\
$a+b\ln\ln K$ & $0.9757$ & $-86.49$ & $0.938$ \\
$a+b\sqrt{\ln K}$ & $0.9595$ & $-83.43$ & $0.889$ \\
$a+b\ln K$ (previously reported) & $0.9386$ & $-80.93$ & $0.825$ \\
$aK^{b}$ & $0.8827$ & $-77.05$ & $0.654$ \\
$a+bK$ (linear) & $0.6825$ & $-71.07$ & $-0.139$ \\
\bottomrule
\end{tabular}
\end{center}

\noindent Seven two-parameter forms on the same six cells (\texttt{code/68});
AICc uses $k=3$, LOO $R^2$ is leave-one-cell-out. Four alternatives beat $\ln
K$ simultaneously on all three criteria. A parametric bootstrap perturbing each
cell mean by its own measured Monte-Carlo SE and refitting all seven forms,
$20{,}000$ draws, puts $\mathbf{P(\ln K \text{ best}) = 0.063}$ on AICc; the
saturating exponential wins $0.448$ of draws and $a+b/K$ another $0.175$. The
only form the data \emph{do} reject is the linear one; §5.3 gives the
practical consequence of $\ln K$'s unbounded growth against the two winners'
\emph{ceiling}.

\textbf{What the six cells do support.} The gap is monotone non-decreasing in
$K$ (strictly, at every step) and concave in $K$ (secant slopes non-increasing
at every step), the property all six non-rejected forms share. Concavity in
$\ln K$ --- the stronger statement --- fails on one adjacent triple,
$K=15/25/45$, by $1.96\times10^{-4}$, a quarter of the mean per-cell
Monte-Carlo SE and therefore not a fact about the data either. The $\ln K$
slope, with a seed-level bootstrap interval ($20{,}000$ draws resampling seeds
within cells), is $b=+0.00110$, $95\%$ CI $[+0.00040,+0.00180]$ --- an interval
spanning a factor of $4.5$, which is the honest precision here. The adaptive
data analysis literature's $\sqrt{\log K/n}$ bounds (§2) are a principled
reason to expect a concave family rather than any particular member of it,
which is the form of the statement §5.3 makes.
```

---

*End of extended technical detail. See `draft/latex/main.tex` for the paper itself, `PROVENANCE_LOG.md` for the full correction-history record, and `code/53_verify_paper_numbers.py` for the machine-checked provenance of every number named above.*
