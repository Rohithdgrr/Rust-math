# Security Policy

## Supported Versions

Only the latest published release receives security patches. Older versions are
supported on a best-effort basis while `mathverse-core` (the only dependency)
remains compatible.

| Version | Supported |
|---------|-----------|
| latest (>= 0.2) | Yes |
| < 0.2 | No |

## Reporting a Vulnerability

Please do **not** open a public issue for security problems. Report them privately:

- **Preferred:** GitHub Security Advisory
  (https://github.com/Rohithdgrr/Rust-math/security/advisories/new)
- **Fallback:** open an issue with `[SECURITY]` in the title and minimal details.

Include:

- The crate and version affected.
- A minimal reproduction (inputs + expected vs. actual output).
- Impact assessment, if known.

## Response Targets

| Stage | Target |
|-------|--------|
| Acknowledgement | 72 hours |
| Triage / assessment | 1 week |
| Fix release | 30 days for moderate+ severity |

This crate is pure math with no unsafe code (`#![forbid(unsafe_code)]`). The main
security-relevant surface is numerical robustness (NaN/infinity handling,
overflow in domain checks), not memory safety. Please still report any case where
an input produces incorrect, panic-ing, or silently wrong results.
