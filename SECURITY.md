# Security Policy

## Supported Versions

We provide security updates for the following versions of AlignTune:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in AlignTune, please report it to us privately by emailing:

📧 **[security@lexsi.ai](mailto:security@lexsi.ai)**

### What to Include

When reporting a vulnerability, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** of the vulnerability
4. **Affected versions** (if known)
5. **Suggested fix** (if you have one)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - **Critical**: 1-7 days
  - **High**: 7-30 days
  - **Medium**: 30-90 days
  - **Low**: Next planned release

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Investigation**: We'll investigate and assess the severity
3. **Updates**: We'll keep you informed of our progress
4. **Resolution**: We'll notify you when the issue is fixed
5. **Credit**: With your permission, we'll acknowledge your contribution

## Security Best Practices

### For Users

When using AlignTune, follow these security best practices:

1. **Keep Updated**: Always use the latest version
2. **Validate Inputs**: Sanitize user inputs before passing to models
3. **Secure Credentials**: Never hardcode API keys or tokens
4. **Review Dependencies**: Regularly audit dependencies for vulnerabilities
5. **Use HTTPS**: Always use secure connections for API calls
6. **Limit Permissions**: Run with minimum necessary privileges
7. **Monitor Logs**: Watch for suspicious activity

### For Contributors

When contributing to AlignTune:

1. **Code Review**: All code must be reviewed before merging
2. **Dependency Updates**: Keep dependencies up to date
3. **Input Validation**: Always validate and sanitize inputs
4. **Secrets Management**: Never commit secrets or credentials
5. **Security Testing**: Test for common vulnerabilities
6. **Documentation**: Document security-relevant changes

## Known Security Considerations

### Model Loading

- Only load models from trusted sources
- Verify model checksums when available
- Be cautious with user-provided model paths

### Data Processing

- Sanitize dataset inputs to prevent injection attacks
- Validate file paths to prevent directory traversal
- Limit file sizes to prevent resource exhaustion

### API Usage

- Use API rate limiting where applicable
- Implement proper authentication for services
- Validate all external API responses

### Training Environment

- Run training in isolated environments when possible
- Monitor resource usage to detect abuse
- Implement proper access controls

## Third-Party Dependencies

AlignTune depends on several third-party libraries. We:

- Regularly update dependencies
- Monitor security advisories
- Use tools like `pip-audit` to check for vulnerabilities

To check your installation:

```bash
pip install pip-audit
pip-audit
```

## Disclosure Policy

When a security issue is fixed:

1. We will prepare a security advisory
2. We will release a patch version
3. We will publicly disclose the issue after users have had time to update
4. We will credit the reporter (with permission)

## Security Updates

Security updates will be:

- Released as patch versions (e.g., 0.2.1)
- Announced in the CHANGELOG
- Posted as GitHub Security Advisories
- Communicated via email to security@lexsi.ai subscribers

## Contact

For security-related questions or concerns:

- **Email**: [security@lexsi.ai](mailto:security@lexsi.ai)
- **General Support**: [support@lexsi.ai](mailto:support@lexsi.ai)
- **Website**: [https://lexsi.ai](https://lexsi.ai)

## Bug Bounty Program

We currently do not have a formal bug bounty program, but we:

- Appreciate and acknowledge security researchers
- Provide credit in release notes (with permission)
- May consider rewards on a case-by-case basis for critical vulnerabilities

## Compliance

AlignTune is designed to be used in compliance with:

- Data protection regulations (GDPR, CCPA, etc.)
- Industry security standards
- Research ethics guidelines

Users are responsible for ensuring their use of AlignTune complies with applicable laws and regulations.

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

**Thank you for helping keep AlignTune and its users safe!** 🔒
