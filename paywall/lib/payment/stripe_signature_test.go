package payment

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"testing"
	"time"
)

func stripeTestHeader(secret string, timestamp int64, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(fmt.Sprintf("%d.", timestamp)))
	_, _ = mac.Write(body)
	return fmt.Sprintf("t=%d,v1=%s", timestamp, hex.EncodeToString(mac.Sum(nil)))
}

func TestStripeSignatureCoversRawBodyAndTimestamp(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	verifier, err := NewStripeSignatureVerifier("whsec_test", 5*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	verifier.now = func() time.Time { return now }
	body := []byte(`{"id":"evt_123"}`)
	header := stripeTestHeader("whsec_test", now.Unix(), body)
	if err := verifier.Verify(body, header); err != nil {
		t.Fatal(err)
	}
	if err := verifier.Verify([]byte(`{"id":"evt_modified"}`), header); !errors.Is(err, ErrInvalidWebhookSignature) {
		t.Fatalf("modified body error = %v", err)
	}
	expiredHeader := stripeTestHeader("whsec_test", now.Add(-6*time.Minute).Unix(), body)
	if err := verifier.Verify(body, expiredHeader); !errors.Is(err, ErrInvalidWebhookSignature) {
		t.Fatalf("expired signature error = %v", err)
	}
}
