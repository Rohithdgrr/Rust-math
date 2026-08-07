# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | ✅ Active           |
| 0.1.x   | ⚠️ Security fixes only |
| < 0.1   | ❌ Unsupported      |

## Reporting a Vulnerability

Please report security vulnerabilities through GitHub's **private vulnerability
reporting** on the repository:

> https://github.com/Rohithdgrr/Rust-math/security/advisories

Do **not** open a public issue for security vulnerabilities.

## Response Times

| Step                    | Target    |
|-------------------------|-----------|
| Acknowledgment          | 24 hours  |
| Initial assessment      | 7 days    |
| Fix target (critical)   | 30 days   |

## What to Include

- The crate version affected
- A minimal reproduction (prefer a `cargo test`-style snippet)
- Impact description (e.g. panic on untrusted input, wrong numerical result)
- Suggested fix, if you have one

## Scope

This crate performs numerical computation over untrusted inputs (e.g. parsing
user-supplied complex values). Panics on crafted input are treated as security
issues. Wrong numerical results are correctness bugs — report them through the
normal issue tracker instead.

## Disclosure Policy

We follow coordinated disclosure: the reporter is credited (if desired), a fix
is prepared, and details are published after a release is available.
