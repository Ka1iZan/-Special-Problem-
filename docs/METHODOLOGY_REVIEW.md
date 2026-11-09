# Methodology review and implementation decisions

Reviewed 9 November 2026 against the supplied 11-page Chapters I-III draft, context handoff, starter kit, and the actual linked releases. The draft and kit are research inputs, not an instruction to execute every proposed experiment. The original PDF has not been edited. Confidential annotation material and derived private notes are excluded from Git.

## Scientific interpretation

The study has two connected questions: **what transfers across these three languages**, and **whether entity alignment plus selective replay improves adaptation and retention**. Its contribution should be a controlled empirical result about this training strategy, rather than a new encoder architecture or a claim that prototypes/replay are new.

Keep the study staged. First establish reliable monolingual and zero-shot measurements. Next compare target-only and source-initialized learning on identical target examples. Only then add alignment and replay. A clean negative result about prototype drift is still informative if the evaluation isolates the mechanism.

## Decisions needed before final experiments

| Issue | Implementation decision or proposed protocol amendment | Why it matters |
|---|---|---|
| Scope | Three languages, four corpora; describe TAG as an operational Tagalog/Filipino resource group | Chapter I still names only Tagalog and Cebuano, whereas Chapter III adds Hiligaynon and Fil-NER |
| Transfer direction | Correct equation (14) to `CEB + HIL -> TAG` | The current expression includes HIL as both source and target |
| Corpus versions | Pin release revisions, hashes, adapter version and mapping version | The current HiligayNER download has 11,150 records, different from the draft's headline size |
| Gold selection | CebuaNER's annotator versions must not be concatenated; FilNER folder selection follows the user's latest-modification rule | Alternative annotations are not independent training examples; recency alone does not prove adjudication |
| Splits | Preserve official splits; create any new split once at document/duplicate-group level | Sentence splitting before grouping can leak the same article across partitions |
| Label mapping | Unknown labels fail conversion; preserve original labels and explicit `IGNORE` | A lookup table cannot resolve incompatible span conventions or nested labels |
| Label budget | Count target training labels and target development labels separately | A full labeled development set is extra target supervision, even if gradients never use it |
| Zero-shot selection | Source development data only; no target labels for checkpoint or hyperparameter choice | Otherwise the claimed zero-shot setting contains target supervision |
| Replication | Separate subset seed and training seed; save one immutable target subset per block | A shared numeric seed alone does not establish matched experimental inputs |
| TAG combination | Balance languages, then the two TAG corpora within TAG; also report each corpus separately | Pooling can hide domain and annotation differences |
| Sequence length | Audit tokenizer lengths; reject silent truncation | Truncation can remove entities and change the evaluation set |

Full monolingual supervision is a reference condition, not a mathematical upper bound: transfer might outperform it. Avoid claiming intrinsic language difficulty from corpus F1 differences because domain and annotation quality differ.

## Correct the drift reference before testing the mechanism

The draft computes pre-adaptation prototypes from source training, then current prototypes from a small memory. Comparing these directly confounds representation change with a change in sampled examples.

Use two distinct sets of frozen objects:

1. **Alignment anchors:** source class centroids from the chosen source training population, optionally averaged with equal language weights. These guide labeled target entity vectors.
2. **Monitoring anchors:** class centroids from a fixed source monitoring set, computed with the source checkpoint before any target update. Recompute current centroids on exactly those same examples and span positions.

For monitoring set M and checkpoint parameters theta:

`D[l,c,t] = 1 - cosine(mean_M[l,c](h_theta_t), mean_M[l,c](h_theta_0))`.

Disable dropout during measurement, use FP32 accumulation, and keep pooling and encoder layer fixed. Drift at step zero should be approximately zero. Missing classes and zero-norm prototypes are unavailable signals, not evidence of large drift. Freeze/checksum anchors. Log each group separately.

Prefer a fixed monitoring subset separate from the replay buffer when feasible, with its size and computational cost reported. If the replay buffer also supplies monitoring examples, disclose that coupling and use identical examples in the replay controls.

Drift is not forgetting. Classifier weights can compensate for geometric movement, and stable centroids can hide boundary errors. Test its relationship to independently evaluated source degradation. Do not infer causality from correlation.

## Isolate drift from group balancing

Uniformly sampling sentences differs from uniformly sampling language/entity groups and then sentences. A sentence containing PER and LOC belongs to two groups and can receive more probability under group sampling.

Keep conventional sentence-uniform replay as a baseline. Add **alignment + group-uniform replay** as the direct control for the proposed method. It uses the same buffer, group index, monitoring schedule and replay count; only the drift weights change. At zero drift, the two group samplers must agree.

For all replay comparisons, match actual replay examples/tokens, optimizer update opportunities, target exposure, and source checkpoint. Report wall time and monitoring passes. Equal memory size and replay frequency alone do not guarantee equal exposure when methods stop at different epochs.

Recommended ablations are ordinary multi-source fine-tuning, alignment only, sentence-uniform replay, alignment plus sentence-uniform replay, alignment plus group-uniform replay, and the proposed alignment plus drift-weighted group replay. The extra group-uniform control is a proposed amendment to the draft table, not a result already established.

## Evaluation contract

Score exact word spans and types. Sum TP/FP/FN over a language's evaluation set before calculating micro-F1. Do not average sentence F1 or substitute token accuracy. The implemented strict IOB2 extractor requires B to begin an entity and reports invalid predicted I tags; it does not silently repair gold sequences.

Masking training loss does not define masked evaluation. Until an audit selects a policy, the evaluator rejects `IGNORE`. An explicit diagnostic policy excludes complete predicted spans touching ignored gold positions and retains original offsets. It never removes masked tokens and joins the surrounding text. Report excluded spans and affected gold support; this conditional score is not automatically comparable with unmasked benchmark scores. Mapping unsupported types to O can be valid under an explicitly defined coarse ontology, but is a different scientific choice from treating their annotation as unknown. Decide from the shared ontology, not a blanket rule that all named entities must be masked.

For entity-overlap diagnostics, normalize only for lookup: Unicode NFC, case-folding and whitespace normalization, retaining punctuation. Build the lookup from source training entities only. Partition both predicted and gold mentions by their own surface forms so false positives have a defined seen/unseen assignment. Report support and recall as well as F1. Under adaptation, source-unseen entities may be seen in target training; report this separately. No result can establish absence from XLM-R's pretraining corpus using this lookup.

Keep natural-distribution evaluation primary. Name masking/substitution and cue ablation are secondary diagnostics that change the input distribution. Unseen entity strings are stronger evidence against exact supervised name reuse, not proof of language-independent reasoning.

## Initial execution sequence

1. Validate canonical data, mappings, duplicates, split provenance and scoring.
2. Use TLUnified-NER for one GPU update and gradient/finite-loss checks.
3. Overfit 32 training examples; aim for at least 0.95 training entity F1 as a diagnostic, then inspect any failure.
4. Run one full monolingual model, selecting on development data. Freeze tuning before using the test set.
5. Resolve the other corpus adapters and reproduce six directed zero-shot evaluations from the three source models.
6. Pilot one target and one budget with matched source checkpoints/subsets.
7. Add ablations and a limited confirmatory replication set after measuring runtime and variance.

The current implementation reaches a tested CPU foundation and a prepared GPU baseline runner. It does not yet implement the full adaptation/prototype-monitoring training loop. The included prototype functions are starter helpers; their PyTorch tests remain a separate gate.

## Reference context

[PRAM](https://aclanthology.org/2023.findings-acl.201/) is relevant prior work for prototype-based zero-resource alignment. The proposed supervised target-span loss and drift replay policy should be described as this study's choices, not as a reproduction of PRAM. [Hugging Face's token-classification guidance](https://huggingface.co/docs/transformers/v5.22.2/en/tasks/token_classification) supports first-subword labeling and loss masking; the additional corpus and retention controls above are specific protocol recommendations.
