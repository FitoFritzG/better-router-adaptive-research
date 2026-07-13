# Task 2 Review — Dataset Acquisition, Provenance, and Licensing

## Verdict

- **Specification compliance:** APPROVED
- **Code quality:** APPROVED
- **Security and privacy:** APPROVED
- **Licensing posture:** APPROVED WITH CONSERVATIVE NON-REDISTRIBUTION

## Scope reviewed

- Pinned RouterBench 0-shot artifact metadata.
- SHA-256 streaming verification.
- Existing-file idempotency and fail-closed mismatch handling.
- Temporary-file download followed by atomic replacement.
- Atomic local download manifest.
- Dataset, code, paper, DOI, and license documentation.
- Forty-eight-row synthetic fixture spanning four task groups and four model arms.
- CLI help, package build, wheel import, typing, linting, tests, coverage, and secret scan.

## Requirements coverage

| Requirement | Evidence | Result |
|---|---|---|
| Official provenance recorded | `data/README.md`, `LICENSES.md` | PASS |
| Exact selected file declared | `routerbench_0shot.pkl` at revision `a4dcf98...` | PASS |
| Expected SHA-256 pinned | `ba4f77...d36d` in code and tests | PASS |
| Existing verified file reused | `test_existing_verified_file_is_reused_without_network` | PASS |
| Existing mismatch rejected | `test_existing_mismatched_file_raises_without_force` | PASS |
| `force=True` replaces only verified data | `test_force_replaces_mismatched_file_and_writes_manifest` | PASS |
| Partial/corrupt download preserves original | checksum and network-failure tests | PASS |
| Manifest records URL, time, size, checksum | download implementation and manifest test | PASS |
| Synthetic fixture has 40–80 rows | 48 committed rows | PASS |
| Four task groups and four arms present | fixture contract test | PASS |
| Fixture cannot be mistaken for evidence | `data_origin` marker and fixture README | PASS |
| No upstream dataset redistributed | `.gitignore`, license documentation | PASS |
| Pickle is not deserialized | acquisition module and security note | PASS |

## Findings corrected

The first review run found CRLF line endings in the synthetic CSV, causing
`git diff --check` to report trailing whitespace. The fixture was normalized to
LF and the entire verification gate was rerun.

A wheel smoke test was also corrected to provide declared runtime dependencies
while importing the package from an isolated wheel target.

## Final evidence

The verified local gate recorded:

- 36/36 tests passing;
- 90.12% total branch coverage;
- Ruff passing;
- strict mypy passing;
- sdist and wheel build passing;
- wheel import passing;
- downloader CLI available;
- YAML and JSON parsing passing;
- secret scan passing;
- `git diff --check` passing;
- normalized fixture line endings.

## Residual limitations

- The 99.6 MB upstream artifact was not downloaded inside the restricted
  execution environment. The pinned URL, revision, byte count, and SHA-256 were
  verified from the official Hugging Face file record.
- The dataset card does not declare a license. This task therefore implements
  local acquisition only and prohibits redistribution.
- Pickle conversion is intentionally deferred to the next separately reviewed
  pipeline stage.
