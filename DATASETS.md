# Datasets: Sources, Licenses, and Attribution

ForgeMind is derived from three third-party datasets. This document records
what we use from each, under which license, and what that means for
redistribution. Licenses below were verified against the actual LICENSE /
license files in each source repository (not guessed). Anything we could
not verify is explicitly marked **requires verification**.

---

## 1. AIFaultBench

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/mehilshah/AIFaultBench (mirrored on Zenodo: https://zenodo.org/records/21782307, DOI 10.5281/zenodo.21782307) |
| **License** | **CC BY 4.0** for the benchmark (reproduction scripts, metadata, packaging) — verified from the dataset card (`license: cc-by-4.0`) and the `LICENSE` file shipped inside the dataset. |
| **What we use** | `index.csv` (fault metadata), and per-fault `bug_report.txt`, `reproduction.json`, `repro.py`, `repro_stdout.log`, `repro_stderr.log`, `manifest.json`. Converted into evidence-grounded fault-analysis and error-analysis examples. |
| **Attribution requirements** | CC BY 4.0 requires attribution. Cite: Shah, Mehil B and Rahman, Mohammad Masudur and Khomh, Foutse — *AIFaultBench: A Reproducible Benchmark of Real-World AI Software Faults* (2026), DOI 10.5281/zenodo.21782307. The dataset ships a `CITATION.cff` with the canonical BibTeX, reproduced in our README. |
| **Redistribution considerations** | CC BY 4.0 permits sharing and adaptation with attribution; our processed examples therefore may be redistributed with the citation above and a link to the source dataset. **Important:** the *bug reports* inside AIFaultBench were recovered from GitHub issues, and the faults originate in 105 upstream open-source repositories. The benchmark authors state that "each original bug report and source repository remains under its respective upstream license." Our per-example metadata therefore records `project`, `issue_url`, and a license note; the upstream license situation for individual issue texts is **requires verification** per repository before commercial redistribution. |

---

## 2. GHPR

| | |
|---|---|
| **Source** | Dataset: https://github.com/soroushj/ghpr-dataset — Raw objects: https://github.com/soroushj/ghpr-dataset-raw |
| **License** | **CC BY 4.0** — verified from the `LICENSE` file in both repositories ("Attribution 4.0 International" legal code). |
| **What we use** | `ghpr.csv` (issue ↔ PR pairs: titles, bodies, timestamps, PR size stats) plus selected raw PR JSON objects (`title`, `body`, merge info) from `ghpr-dataset-raw`. Converted into issue → resolution examples. |
| **Attribution requirements** | CC BY 4.0 attribution: credit *GHPR* by Soroush Javdan and link both repository URLs (see README citation section). |
| **Redistribution considerations** | CC BY 4.0 permits redistribution of the dataset itself with attribution. The **issue and PR texts** were authored on GitHub in 13 CNCF-graduated projects (kubernetes, envoy, containerd, prometheus, etc.) and remain governed by those projects' terms; the dataset authors redistributed them under CC BY 4.0, but the upstream projects' own licenses for contributed text are **requires verification** per project. Our metadata records the source repository and issue/PR numbers for every example. |

---

## 3. Defects4J

| | |
|---|---|
| **Source** | https://github.com/rjust/defects4j |
| **License** | **MIT** for the Defects4J framework — verified from `license.txt` in the repository ("Copyright (c) 2014-2024 René Just, Darious Jalai, and Defects4J contributors" + MIT permission grant). |
| **What we use** | **Framework metadata only**: per-project `active-bugs.csv` (bug IDs, buggy/fixed revisions, report IDs/URLs), `patches/<bug>.src.patch` (developer fix diffs shipped inside the framework), `patches/<bug>.test.patch` (regression test diffs), and `trigger_tests/<bug>` (failing-test output). We deliberately do **not** download, bundle, or redistribute any benchmarked project's source tree. |
| **Attribution requirements** | MIT requires the copyright notice and permission notice to be included with substantial portions of the framework software. Our use is transformative (derived training examples with per-example provenance), and we credit Defects4J and link the repository (see README). The canonical paper citation is: Just, R., Jalali, D., and Ernst, M. — *Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies for Java Programs*, ISSTA 2014. |
| **Redistribution considerations** | The patch and trigger-test files live inside the MIT-licensed framework repository, which redistributes them as part of the benchmark. However, the *code changes they describe* originate in upstream projects with their own licenses (Apache-2.0, BSD, LGPL, and others — per project). We therefore: (a) store per-example `project` metadata so every patch's origin is traceable; (b) mark the upstream-project license situation **requires verification** per project before any commercial redistribution; (c) do not ship project source code at all. |

---

## What ForgeMind adds

ForgeMind's processed examples are **transformative derivatives**: for each
record we construct instruction-format conversations whose user content
quotes the source's own fault/issue/patch material and whose assistant
content is generated *strictly from* that material (no invented root
causes, fixes, or metadata). Per-example `metadata` records:
`source`, `task`, `language` (where supported), `project`, `license`,
plus source-native identifiers (`bug_id`, `issue_number`/`pull_number`,
`report_url`, revisions) and a `group_key` used for leakage-safe splitting.

## Summary table

| Dataset | Dataset license | Upstream content license | Redistribution of our examples | Verification status |
|---|---|---|---|---|
| AIFaultBench | CC BY 4.0 (verified) | Upstream repos keep own licenses | OK with attribution | Upstream per-repo: requires verification |
| GHPR | CC BY 4.0 (verified) | CNCF project terms | OK with attribution | Upstream per-project: requires verification |
| Defects4J | MIT framework (verified) | Per-project upstream licenses; no source code redistributed by us | Metadata + patches redistributed by the MIT framework; per-project check advised | Upstream per-project: requires verification |

## Limitations of the leakage-prevention and quality design

- Group keys make splits leakage-safe **within** a source (same bug /
  issue / PR always in one split). Cross-source overlap (e.g., the same
  upstream project appearing in two datasets) is *not* detected — the
  sources' native identifiers differ. For ForgeMind's three sources this
  overlap is expected to be negligible (AIFaultBench covers AI/ML
  repositories; GHPR covers CNCF infrastructure; Defects4J covers Java
  libraries), but it is not formally excluded.
- AIFaultBench does not ship gold fixes; assistant content is grounded in
  reproduction evidence only and says so explicitly.
- GHPR's issue→PR association comes from GitHub fix-keyword linkage
  (as recorded by the dataset); we describe it as "linked/merged
  upstream", not as our own verification.
- Defects4J root-cause text is derived from the developer patch's removed
  lines; the surrounding fix narrative is intentionally minimal.
