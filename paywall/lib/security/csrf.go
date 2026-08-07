package security

import (
	"crypto/hmac"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"strings"
)

var ErrInvalidCSRF = errors.New("invalid csrf token")

func IssueCSRF(signingKey []byte, sessionNonce string) (string, error) {
	if len(signingKey) < 32 || sessionNonce == "" {
		return "", ErrInvalidCSRF
	}
	random := make([]byte, 32)
	if _, err := rand.Read(random); err != nil {
		return "", err
	}
	encodedRandom := base64.RawURLEncoding.EncodeToString(random)
	signature := signHMAC(signingKey, "csrf:"+sessionNonce+":"+encodedRandom)
	return encodedRandom + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

func VerifyCSRF(signingKey []byte, sessionNonce, cookieToken, headerToken string) error {
	if len(signingKey) < 32 || sessionNonce == "" || cookieToken == "" ||
		!hmac.Equal([]byte(cookieToken), []byte(headerToken)) {
		return ErrInvalidCSRF
	}
	parts := strings.Split(cookieToken, ".")
	if len(parts) != 2 {
		return ErrInvalidCSRF
	}
	random, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil || len(random) != 32 {
		return ErrInvalidCSRF
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return ErrInvalidCSRF
	}
	expected := signHMAC(signingKey, "csrf:"+sessionNonce+":"+parts[0])
	if !hmac.Equal(signature, expected) {
		return ErrInvalidCSRF
	}
	return nil
}
