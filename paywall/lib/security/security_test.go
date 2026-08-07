package security

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestSessionRejectsTamperingAndEnforcesRotation(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	oldKey := []byte("old-session-key-material-is-32-bytes!")
	newKey := []byte("new-session-key-material-is-32-bytes!")
	oldRing, err := NewSessionKeyRing("old", map[string][]byte{"old": oldKey}, "example.com", "paid-site")
	if err != nil {
		t.Fatal(err)
	}
	entitlementExpiry := now.Add(30 * 24 * time.Hour)
	oldToken, _, err := oldRing.Issue("acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", 3, now, 7*24*time.Hour, 10*time.Minute, "individual", entitlementExpiry)
	if err != nil {
		t.Fatal(err)
	}
	rotatingRing, err := NewSessionKeyRing("new", map[string][]byte{"new": newKey, "old": oldKey}, "example.com", "paid-site")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rotatingRing.Verify(oldToken, now); err != nil {
		t.Fatalf("previous key token rejected during rotation: %v", err)
	}
	parts := strings.Split(oldToken, ".")
	parts[1] = parts[1][:len(parts[1])-1] + "A"
	if _, err := rotatingRing.Verify(strings.Join(parts, "."), now); !errors.Is(err, ErrInvalidSession) {
		t.Fatalf("tampered token error = %v, want ErrInvalidSession", err)
	}
	newToken, claims, err := rotatingRing.Issue("acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", 3, now, 7*24*time.Hour, 10*time.Minute, "individual", entitlementExpiry)
	if err != nil {
		t.Fatal(err)
	}
	if !claims.AuthorizationLeaseValid(now.Add(9*time.Minute)) || claims.AuthorizationLeaseValid(now.Add(10*time.Minute)) {
		t.Fatal("authorization lease boundary is incorrect")
	}
	if _, err := oldRing.Verify(newToken, now); !errors.Is(err, ErrInvalidSession) {
		t.Fatalf("new-key token accepted by old-only ring: %v", err)
	}
	if _, err := rotatingRing.Verify(newToken, now.Add(7*24*time.Hour)); !errors.Is(err, ErrExpiredSession) {
		t.Fatalf("expired token error = %v, want ErrExpiredSession", err)
	}
}

func TestCSRFIsBoundToSessionNonce(t *testing.T) {
	key := []byte("csrf-signing-key-material-is-32-bytes!")
	token, err := IssueCSRF(key, "session-nonce-a")
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyCSRF(key, "session-nonce-a", token, token); err != nil {
		t.Fatal(err)
	}
	if err := VerifyCSRF(key, "session-nonce-b", token, token); !errors.Is(err, ErrInvalidCSRF) {
		t.Fatalf("cross-session csrf error = %v", err)
	}
	if err := VerifyCSRF(key, "session-nonce-a", token, token+"x"); !errors.Is(err, ErrInvalidCSRF) {
		t.Fatalf("mismatched double-submit token error = %v", err)
	}
}

func TestEncryptedFieldUsesAccountAsAssociatedData(t *testing.T) {
	key := []byte("0123456789abcdef0123456789abcdef")
	field, err := EncryptField("pii-current", key, "acc_a", "cus_secret")
	if err != nil {
		t.Fatal(err)
	}
	plaintext, err := DecryptField(map[string][]byte{"pii-current": key}, "acc_a", field)
	if err != nil || plaintext != "cus_secret" {
		t.Fatalf("decrypt = %q, %v", plaintext, err)
	}
	if _, err := DecryptField(map[string][]byte{"pii-current": key}, "acc_b", field); err == nil {
		t.Fatal("ciphertext decrypted for the wrong account")
	}
}

func TestExactOriginRequiresHTTPSAndExactHost(t *testing.T) {
	if err := ValidateExactOrigin("https://example.com", "https://example.com"); err != nil {
		t.Fatal(err)
	}
	for _, origin := range []string{"http://example.com", "https://evil.example", "https://example.com/path", "https://example.com.evil"} {
		if err := ValidateExactOrigin(origin, "https://example.com"); !errors.Is(err, ErrInvalidOrigin) {
			t.Errorf("origin %q error = %v", origin, err)
		}
	}
}
