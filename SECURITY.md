# Security

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly by opening a private issue or contacting the maintainer directly. Do not open public issues for security vulnerabilities.

## Implemented Protections

### Web Application

- **CSRF Protection**: All mutation requests (POST/PUT/DELETE) require `X-Requested-With` header
- **Security Headers**: CSP, X-Frame-Options (DENY), X-Content-Type-Options (nosniff), Referrer-Policy, Permissions-Policy, COOP, CORP, HSTS (when HTTPS)
- **Rate Limiting**: All endpoints protected with per-minute limits (default: 200/min)
- **Path Traversal Prevention**: Job IDs validated via regex, file paths resolved and checked against upload/output directories
- **File Upload Validation**: Extension whitelist + magic bytes verification
- **Input Validation**: All configuration parameters validated with type and range checks; annotation coordinates bounded
- **File Size Limits**: 2 GB max upload, 100 MB max JSON
- **Automatic Cleanup**: Temporary files removed after 1 hour

### Secret Management

- `FLASK_SECRET_KEY` should be set via environment variable for production
- Warning emitted at startup if not configured
- `.env` files excluded from version control via `.gitignore`
- See `.env.example` for required variables

### Data Privacy (GDPR)

- Video files containing personal data are processed locally
- Upload/output directories are excluded from git
- Automatic cleanup reduces data retention
- No external API calls or data transmission during processing

## Security Considerations

- This tool is designed for **local/intranet use**. It does not implement authentication
- For public deployment, add authentication and consider Redis-backed rate limiting
- The development server (Werkzeug) should not be used in production — use gunicorn
- YOLO model files (`.pt`) are downloaded from Ultralytics on first run
