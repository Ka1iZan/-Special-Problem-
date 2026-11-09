# Data readiness and descriptive audit

Audit date: 9 November 2026. These are corpus measurements, not model results. Public sources and the supplied private source were accessible. Raw text and complete record-level audit output are excluded from Git.

## Public versions

| Resource | Pinned revision | Readiness |
|---|---|---|
| [TLUnified-NER](https://huggingface.co/datasets/ljvmiranda921/tlunified-ner) | `1d43a4a60df51660aa8746e8952c536ba9483c72` | Canonical files exported with original splits; ready for tokenizer/GPU preflight |
| [CebuaNER](https://github.com/mebzmoren/CebuaNER) | `1239289791b86f4694509f022c9b802dd17c9ed7` | Two aligned annotator versions; choose and document the gold policy before export |
| [HiligayNER](https://github.com/jvlzloons/HiligayNER) | `cfd02cb0fdb2f1aadb75106c7a779c17db687ada` | Character-offset adapter and annotation review required |

Download provenance and file hashes are recorded in `data/raw/public_v1/manifest.json`. Reproduce the descriptive audit with `scripts/audit_public_releases.py`. Counts refer to pinned files, not older README or draft totals.

## TLUnified-NER pilot

| Released split | Records | Word tokens | PER spans | LOC spans | ORG spans |
|---|---:|---:|---:|---:|---:|
| Train | 6,252 | 198,579 | 6,404 | 3,353 | 3,160 |
| Development | 782 | 25,069 | 792 | 375 | 445 |
| Test | 782 | 25,100 | 833 | 383 | 363 |
| Total | 7,816 | 248,748 | 8,029 | 4,111 | 3,968 |

All released label sequences passed strict BIO validation. No exact normalized cross-split duplicates were found under NFC, case-folding and collapsed whitespace. This does not establish absence of near-duplicates, shared articles or translations. Released row IDs are preserved, but parent article identifiers are unavailable; document independence cannot be claimed from row IDs.

The maximum word lengths are 150/198/101 for train/development/test. XLM-R subword lengths remain a separate preflight. The runner refuses overlength examples rather than truncating them. Actual tokenizer results determine whether explicit windows are needed.

The adapter reads integer-label names from the Parquet metadata. The canonical order of labels is not assumed to equal the source's integer order. `validation` becomes `dev`, with original example IDs and labels retained.

## CebuaNER

Both released annotation files contain 4,258 blank-separated blocks and 184,033 token rows, with matching token sequences. The release's agreement notebook identifies the files as annotator outputs. There are 3,005 differing token labels, giving raw token-label agreement of 98.367%. This is not entity-level agreement, kappa, or NER performance.

The labels include PER/LOC/ORG and OTHER. The first annotation file has no invalid BIO blocks in the raw audit and contains nine duplicate-text groups. Do not concatenate the two files as extra examples. Select an official/adjudicated version if identifiable, or define and document an annotation-resolution policy.

The files contain a DOCSTART marker and four columns. The generic converter refuses to guess its document boundary semantics. The original CRF notebook slices records into train/test/development; this is not a separately released split manifest. Audit duplicate groups before reproducing those partitions or defining a new documented split.

## HiligayNER

The pinned JSONL contains 11,150 text records with character offsets and BIO labels. Observed labels in this file are PER/LOC/ORG; a notebook's DATE label definition does not establish DATE examples in the downloaded data.

The raw audit identifies one invalid-offset record, 33 records with overlapping annotations, 3,077 records whose annotation boundaries do not all match a naive whitespace tokenizer, 66 invalid BIO sequences among otherwise unambiguous records, and 30 duplicate normalized-text groups. Categories can overlap and must not be added as a total rejection count.

A whitespace-boundary mismatch is not automatically an annotation error: punctuation and tokenizer conventions can explain it. Build a deterministic offset-preserving adapter, inspect ambiguous cases and report resolution/exclusion counts. Do not silently shift offsets or repair labels.

The source notebook uses a random 72/8/20 train/development/test split; the JSONL has no split metadata. Duplicate/document grouping and a frozen split manifest are needed before a research-ready export.

## Restricted resource

The supplied private resource and annotation manual were accessible. Version selection follows the user's latest-folder-modification rule within each batch; modification time does not prove final adjudication. Private manifests record chosen versions and hashes. Annotation-column and span-projection questions still require review, so no training-ready FilNER export is claimed. No private source URLs, annotation examples or manual-derived tables are published here.

## Next data gates

1. Tokenizer preflight and a GPU pilot on TLUnified only.
2. Resolve CebuaNER's gold version and split policy.
3. Review HiligayNER offset/BIO exceptions and freeze an adapter and split.
4. Resolve private annotation layers and coarse-ontology projection with provenance.
5. Audit cross-corpus duplicates, document overlap, domain and language balance before combining TAG resources.
