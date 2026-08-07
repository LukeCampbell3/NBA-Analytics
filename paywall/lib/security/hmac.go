package security

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
)

func IdentityHMAC(key []byte, provider, providerUserID string) string {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(provider + ":" + providerUserID))
	return hex.EncodeToString(mac.Sum(nil))
}
