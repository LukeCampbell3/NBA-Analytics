package security

import "net/http"

func SetSecurityHeaders(header http.Header) {
	header.Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
	header.Set("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; object-src 'none'")
	header.Set("X-Content-Type-Options", "nosniff")
	header.Set("Referrer-Policy", "strict-origin-when-cross-origin")
	header.Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
	header.Set("Cross-Origin-Opener-Policy", "same-origin")
}
