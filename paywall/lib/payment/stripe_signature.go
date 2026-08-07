package payment

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

var ErrInvalidWebhookSignature = errors.New("invalid webhook signature")

type StripeSignatureVerifier struct {
	secret    []byte
	tolerance time.Duration
	now       func() time.Time
}

func NewStripeSignatureVerifier(secret string, tolerance time.Duration) (*StripeSignatureVerifier, error) {
	if secret == "" || tolerance <= 0 || tolerance > 15*time.Minute {
		return nil, fmt.Errorf("invalid stripe webhook verifier configuration")
	}
	return &StripeSignatureVerifier{secret: []byte(secret), tolerance: tolerance, now: time.Now}, nil
}

func (v *StripeSignatureVerifier) Verify(rawBody []byte, signatureHeader string) error {
	var timestampText string
	var signatures [][]byte
	for _, field := range strings.Split(signatureHeader, ",") {
		name, value, ok := strings.Cut(strings.TrimSpace(field), "=")
		if !ok {
			continue
		}
		switch name {
		case "t":
			if timestampText != "" {
				return ErrInvalidWebhookSignature
			}
			timestampText = value
		case "v1":
			decoded, err := hex.DecodeString(value)
			if err == nil && len(decoded) == sha256.Size {
				signatures = append(signatures, decoded)
			}
		}
	}
	timestamp, err := strconv.ParseInt(timestampText, 10, 64)
	if err != nil || len(signatures) == 0 {
		return ErrInvalidWebhookSignature
	}
	eventTime := time.Unix(timestamp, 0)
	delta := v.now().Sub(eventTime)
	if delta < 0 {
		delta = -delta
	}
	if delta > v.tolerance {
		return ErrInvalidWebhookSignature
	}
	mac := hmac.New(sha256.New, v.secret)
	_, _ = mac.Write([]byte(timestampText))
	_, _ = mac.Write([]byte("."))
	_, _ = mac.Write(rawBody)
	expected := mac.Sum(nil)
	for _, signature := range signatures {
		if hmac.Equal(signature, expected) {
			return nil
		}
	}
	return ErrInvalidWebhookSignature
}
