# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in `mathverse-calculus`, please report it by:

1. **Email**: Send details to the maintainer (see repository `SECURITY.md`)
2. **GitHub Advisory**: Use the [GitHub Security Advisory](https://github.com/Rohithdgrr/Rust-math/security/advisories/new) feature

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 72 hours
- **Assessment**: Within 7 days
- **Fix target**: Within 30 days (critical), 90 days (non-critical)

## Security Practices

This crate follows these security practices:

- `#![forbid(unsafe_code)]` — no unsafe Rust code
- No network or file I/O operations
- Pure computational library with no side effects
- Dependencies are audited via `cargo audit` in CI
- No secrets or credentials handled

## Dependency Security

We monitor dependencies for known vulnerabilities using:
- `cargo audit` in CI pipeline
- GitHub Dependabot alerts
