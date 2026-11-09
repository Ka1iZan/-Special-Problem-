# Statistical analysis plan - draft for protocol freeze

This plan is proposed before model results. It is not preregistered and does not report an observed treatment effect. Real descriptive corpus measurements are in `DATA_READINESS.md`; model comparison tests need trained predictions and matched runs.

## Estimands and units

Within a fixed target, annotation budget, source initialization and evaluation set, estimate the mean paired difference `Delta = mean_r(F1_A,r - F1_B,r)`. Store F1 as fractions in [0,1]; report differences in F1 percentage points. For retention, record `F1_before - F1_after` for each source and class; positive values mean degradation. Use an equal-language mean of source micro-F1 for the source aggregate, not a pool dominated by the larger corpus.

The independent experimental unit for training variability is a **paired replication block**, not a token, entity, language, class or checkpoint. Each block records subset seed, training seed, target subset IDs, source checkpoint, data/evaluation manifests, evaluation policy and common training configuration. Replay comparisons additionally record memory IDs, replay schedule, monitoring set and alignment configuration. The analysis rejects duplicate IDs, missing pairs and mismatched metadata.

If every block reuses one source checkpoint, inference is conditional on that checkpoint. To include source-initialization variability, independently train source checkpoints across blocks. A crossed design with multiple target subsets and multiple optimizer seeds per subset can separate those effects; a single seed controlling both only estimates their combined variability. Do not describe those as separate variance estimates.

## Principal contrasts

The proposed initial confirmatory focus is the 10% budget, chosen before model outcomes. Five and 25% provide exploratory learning curves until the final plan says otherwise.

| Contrast | Endpoint | Interpretation |
|---|---|---|
| Alignment minus multi-source fine-tuning | Target micro-F1 | Incremental alignment benefit |
| Sentence-random replay minus multi-source fine-tuning | Mean source micro-F1 after adaptation | Replay retention benefit from the same source start |
| Proposed minus alignment + group-uniform replay | Mean source micro-F1 after adaptation | Incremental benefit of drift weighting |

Across three targets this is one family of nine two-sided tests. Apply Holm adjustment to all nine; the runner refuses an incomplete family. Other contrasts, classes, budgets and overlap analyses remain exploratory unless included before protocol freeze. `configs/analysis_plan.json` is the machine-readable proposal.

Always show proposed-versus-control target effects alongside retention. A nonsignificant target loss is not evidence of no sacrifice. To claim preserved target performance, first justify a non-inferiority margin delta in F1 points, then predeclare the one-sided hypothesis and multiplicity rule. A suitable lower confidence bound must exceed `-delta`; retain the source-retention requirement separately. No margin has been invented or implemented here.

## Replications and statistical tests

Use five pairs for pilots and descriptive stability checks. Prefer at least ten independent pairs for principal comparisons if compute allows, with the final count informed by pilot variance, a scientifically meaningful effect and the planned multiplicity correction. Ten is a planning starting point, not a power guarantee. Do not add seeds selectively until a p-value crosses a threshold.

The default paired test is a two-sided sign-flip randomization test on the mean difference. It requires independent blocks and exchangeability of method assignments under the null, or an appropriate sign-symmetry assumption. It is not assumption-free. The implementation enumerates all sign assignments for up to 20 nonzero pairs; above that it uses a reproducible Monte Carlo test with a plus-one correction. The exact p-value includes ties.

With five nonzero pairs, the minimum possible two-sided exact p-value is `2 / 2^5 = 0.0625`; with six it is 0.03125. Zero differences and ties can make resolution worse, and multiplicity further limits rejection. This mathematical constraint is tested in the code. Avoid choosing a t-test versus Wilcoxon after inspecting five observations or switching to one-sided inference after seeing the direction. [SciPy documents the paired permutation construction](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

Report mean and SD by method, individual paired differences, mean and median difference in points, a 95% paired-run bootstrap interval, raw and Holm-adjusted p-values, and sample size. Paired standardized `d_z` is secondary and undefined for effectively constant differences. The raw F1-point difference is the primary effect size. A narrow bootstrap interval from very few or identical observed pairs should not be presented as precise population certainty.

## Two uncertainty sources

**Paired-run bootstrap:** resample entire paired blocks and calculate mean run-level F1 differences. This estimates training/subset variability conditional on the fixed evaluation set.

**Paired-document bootstrap:** hold trained checkpoints fixed, resample test documents with replacement, keep all sentences/entities in each selected document, sum TP/FP/FN, and recompute corpus micro-F1. Use the same document draw for both methods. This addresses evaluation-sample variability conditional on the checkpoints.

**Crossed run/document bootstrap:** when both dimensions are available, independently resample paired blocks and documents, using the same sampled documents across both methods and all sampled blocks. Compute corpus F1 within each selected run, then average its paired difference. The documents are shared across runs; they are not independent documents newly nested under each seed. This is a multi-bootstrap-style extension inspired by [MultiBERTs](https://arxiv.org/abs/2106.16163), not a claim that its published implementation has been reproduced exactly.

The scripts implement these percentile intervals, enforce matched gold hashes, and distinguish their uncertainty targets. They do not infer a p-value by counting signs of uncentered bootstrap effects. Intervals are pointwise, not simultaneous. Sparse draws with undefined F1 fail explicitly rather than being silently discarded. If parent documents are missing, sentence resampling requires explicit opt-in and a stated independence limitation. More independent article identifiers cannot be manufactured from row IDs.

## Drift and diagnostic analysis

Pair each final source-language/entity drift with the corresponding source degradation. Prefer development data for iterative monitoring; reserve final test measurements for frozen analyses. Start with plots and within-condition rank associations. A pooled Spearman p-value treating all classes, sources and checkpoints as independent is inappropriate: they share model parameters and data.

For exploratory uncertainty, resample whole independent training blocks while retaining their source/class measurements together, and separate method, budget and target strata. A mixed model or within-group analysis can be considered after enough blocks exist. Repeated time checkpoints require additional temporal dependence handling. The current generic inference scripts do not yet implement this drift association analysis because no drift histories exist.

Seen/unseen entity comparisons, per-class F1 and corpus-domain sensitivities require support counts and a predeclared scoring policy. Low support or zero denominators should be shown as undefined, not quietly assigned a favorable score.

## Replacement text for Chapter III.N

> Principal comparisons will use matched replication blocks sharing the target subset, source checkpoint, preprocessing and evaluation protocol. Target entity micro-F1 and source retention will be summarized by mean, standard deviation and paired differences in F1 percentage points. Paired randomization tests will be specified before final evaluation, with Holm adjustment across the declared principal comparison family. Confidence intervals over paired runs will characterize variability due to the replicated training procedure on fixed evaluation sets. Where document identifiers and predictions are available, paired document resampling will separately characterize evaluation-sample uncertainty by recomputing corpus-level entity F1 from resampled counts. Replications will not be conflated with tokens, entity classes or repeated checkpoints. Pilot results will inform a feasible final replication count, and small-sample limits will be reported. Evidence of no statistically detectable target degradation will not be interpreted as non-inferiority without a prespecified practically meaningful margin.

For general NLP test-selection context, see [Dror et al.](https://aclanthology.org/P18-1128/). The estimands, family of comparisons and sampling units above are specific recommendations for this project.
