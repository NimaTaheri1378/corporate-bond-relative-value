# WRDS extraction layer

Step 03b uses supervised subprocess extraction. Each partition runs in an isolated child process and writes its own Parquet, SQL sidecar, payload JSON, result JSON, and child logs. Existing non-empty partitions are skipped by default, so interrupted runs are resumable.

The core dated return panels use quarter-sized extraction windows by default. TRACE uses month-sized windows. This is intentionally more conservative than annual pulls: it reduces memory pressure, gives better resume granularity, and avoids one killed worker taking down the full job.

Common commands:

```bash
python scripts/run_step03_extract.py --phase plan --core-window quarter --trace-window month
python scripts/run_step03_extract.py --phase core --workers 4 --engine subprocess --core-window quarter
python scripts/run_step03_extract.py --phase trace --workers 3 --engine subprocess --trace-pilot-months 6
```


## Step 03c: TRACE streaming pilot before one scale-up

Step 03b proved that the core non-TRACE raw lake can be extracted reliably, but monthly TRACE pulls were too large for the child process. Step 03c replaces the pandas whole-window TRACE extraction path with a streaming server-side cursor writer:

- bounded date windows, no global `ORDER BY`;
- lean TRACE column set by default;
- server-side fetch batches written incrementally to Parquet;
- exact row-count validation for pilot windows;
- subprocess isolation for each window;
- pilot-first, then one full scale-up pass.

The intended workflow is:

1. run the Step 03c pilot patch;
2. inspect the generated pilot validation manifest and figures;
3. only after acceptance, run one full scale command from `data/manifests/extractions/step03c_trace_scale_recommendation.md`.


## Step 03d: TRACE cursor metadata fix

The Step 03c pilot validated table access and daily row counts, but WRDS server-side cursor metadata returned `None` for the named cursor before fetch. Step 03d fixes the extractor by treating the explicitly selected `task.columns` as the stable schema source for streaming chunks. This keeps the good properties of the Step 03c design:

- bounded date windows;
- no global `ORDER BY`;
- lean TRACE column set by default;
- subprocess isolation;
- exact count validation on pilot windows;
- pilot first, then one full scale-up pass only.


## Step 03e: one full TRACE scale pass

Step 03d validated the streaming TRACE extractor on high-volume pilot windows with exact row-count checks. Step 03e is the single full-scale TRACE extraction pass. It uses weekly partitions, a lean TRACE column set, server-side cursors, incremental Parquet writes, subprocess isolation, exact count validation, automatic resume for already-written non-empty partitions, and a small number of automatic retries inside one scale wrapper.

Operational rules:

- do not overwrite successful partitions;
- do not run another pilot unless schema or extraction code changes;
- keep WRDS parallelism moderate because the bottleneck is WRDS/network/I/O rather than local GPU;
- use local CPU for Parquet compression through Arrow threads;
- inspect the Step 03e validation archive before moving to cleaning and security-master joins.


## Step 03f: TRACE failed-window resume

Step 03e completed most weekly TRACE Enhanced partitions but left a small number of failed windows. Step 03f is a repair step, not a new full extraction. It reads the Step 03e manifest, selects only failed or missing windows, quarantines any invalid weekly outputs, optionally splits failed weekly windows into daily tasks, and writes a separate resume manifest with exact row-count validation.

Operational rules:

- never rerun all Step 03e windows unless the raw lake is intentionally reset;
- run `diagnose` first, then `pilot`, then one `resume` pass if WRDS access is OK;
- keep failed-window repair outputs in the canonical raw lake with non-overlapping daily partition labels;
- preserve Step 03e manifests and logs for auditability;
- do not run extraction mode on the login node.


## Step 03g: fixed Arrow schema for TRACE streaming chunks

Step 03e downloaded 1,067 of 1,188 weekly TRACE partitions correctly. The 121 failures were not authentication failures and not row-count mismatches. They were PyArrow append failures caused by pandas/Arrow inferring different numeric types across chunks inside the same output file, for example `days_to_sttl_ct` switching between `double` and `int64`.

Step 03g fixes the extractor by writing every TRACE chunk through an explicit Arrow schema derived from the selected TRACE columns before appending to Parquet. The repair policy is:

- preserve all already validated Step 03e weekly partitions;
- quarantine only failed weekly output directories before repair;
- split failed weekly windows into daily windows;
- pilot two failed windows first;
- if the pilot validates, run one failed-window resume pass only.

