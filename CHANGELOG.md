# Changelog

This log tracks edits made by the assistant in response to your requests.

## 2025-08-28
- Added builtin dataset support and quick-run toggles in `refinement_gnn_experiments.ipynb`:
  - Added TUDataset loader (`MUTAG`, etc.), label remapping, and integration with variants/loaders.
  - Added smoke-test cell and one-liner quick run.
  - Ensured `label_order` is set for builtin runs.
- Kept local CSV+PT workflow intact for histology graphs.
- Fixed errors in `refinement_gnn_experiments.ipynb` when running comparisons:
  - Repaired malformed f-strings causing `NameError` and `UnboundLocalError` in `run_architecture` and `save_variant_artifacts`.
  - Re-ran training successfully; artifacts saved under `*_refine_ckpts/` and plots generated.

### 2025-08-28 (later)
- Switched default data mode to builtin (`DATA_MODE = 'builtin'`, default `MUTAG`).
- Saved per-epoch curves per architecture-variant (`*_curves.json`).
- Added epoch vs validation accuracy plots per-architecture and per-variant; saved under `plots/`.
- Cleaned up duplicate loader/util cells and fixed a minor syntax issue in the aggregation cell.

## 2025-08-27
- Created proposal document `trans_dynasim_gdc_proposal.md` (paper-style summary of aims, methods, comparisons, and expected benefits) aligned to the notebook work.
