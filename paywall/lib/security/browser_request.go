package security

import (
	"errors"
	"mime"
	"net/http"
)

var ErrUnsafeBrowserRequest = errors.New("unsafe browser request")

func ValidateStateChangingRequest(request *http.Request, expectedOrigin, sessionNonce string, csrfKey []byte) error {
	switch request.Method {
	case http.MethodPost, http.MethodPut, http.MethodDelete:
	default:
		return ErrUnsafeBrowserRequest
	}
	if err := ValidateExactOrigin(request.Header.Get("Origin"), expectedOrigin); err != nil {
		return ErrUnsafeBrowserRequest
	}
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return ErrUnsafeBrowserRequest
	}
	cookie, err := request.Cookie(CSRFCookieName)
	if err != nil || VerifyCSRF(csrfKey, sessionNonce, cookie.Value, request.Header.Get("X-CSRF-Token")) != nil {
		return ErrUnsafeBrowserRequest
	}
	return nil
}
