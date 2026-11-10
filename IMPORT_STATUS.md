# NEXUS repository import status — verified 2026-11-04

**Status of original project transfer: COMPLETE (99/99 byte-identical).**

The previous `IMPORT_STATUS.md` said 32 binaries were pending. That status became obsolete after subsequent commits. Independent remote-tree verification on 2026-11-04 compared each original file's path, byte length, and Git blob SHA-1 against the checksums derived from the local original files in `nexus_local_git_sha_audit.json`:

- Original files: **99**
- Original files found in `main`: **99**
- Exact Git blob hash + byte-length matches: **99**
- Missing originals or mismatches: **0**
- Additional repository documents at that audit: `README.md`, `IMPORT_STATUS.md`, `nexus_file_manifest.json`.
- The file inventory and SHA-256 reference remains `nexus_file_manifest.json`.

The above is a **repository file-transfer audit**, not verification of the underlying scientific claims. Loose source files and archived ZIP contents are not interchangeable: many experiment outputs remain inside the archived ZIPs. New engineering commits may add files beyond the original 99; they are not counted as original import files.

**Latest engineering status:** G5 pretrained-model testing is still blocked; see `reviews/2026-11-04_G5_capture_hardening_review.md`. No fabricated hardware or pretrained benchmark evidence is implied.
