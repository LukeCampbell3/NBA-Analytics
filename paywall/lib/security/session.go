package security

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

var (
	ErrInvalidSession = errors.New("invalid session")
	ErrExpiredSession = errors.New("expired session")
)

type SessionClaims struct {
	Version           int    `json:"version"`
	Issuer            string `json:"iss"`
	Audience          string `json:"aud"`
	Subject           string `json:"sub"`
	SessionEpoch      uint64 `json:"session_epoch"`
	Plan              string `json:"plan,omitempty"`
	IssuedAt          int64  `json:"iat"`
	AuthzExpiry       int64  `json:"authz_exp"`
	EntitlementExpiry int64  `json:"entitlement_exp,omitempty"`
	Expiry            int64  `json:"exp"`
	Nonce             string `json:"nonce"`
}

type sessionHeader struct {
	Algorithm string `json:"alg"`
	Type      string `json:"typ"`
	KeyID     string `json:"kid"`
}

type SessionKeyRing struct {
	currentKeyID string
	keys         map[string][]byte
	issuer       string
	audience     string
}

func NewSessionKeyRing(currentKeyID string, keys map[string][]byte, issuer, audience string) (*SessionKeyRing, error) {
	if currentKeyID == "" || issuer == "" || audience == "" {
		return nil, fmt.Errorf("session key id, issuer, and audience are required")
	}
	copied := make(map[string][]byte, len(keys))
	for keyID, key := range keys {
		if keyID == "" || len(key) < 32 {
			return nil, fmt.Errorf("each session key must contain at least 32 bytes")
		}
		copied[keyID] = append([]byte(nil), key...)
	}
	if _, ok := copied[currentKeyID]; !ok {
		return nil, fmt.Errorf("current session key is missing")
	}
	return &SessionKeyRing{currentKeyID: currentKeyID, keys: copied, issuer: issuer, audience: audience}, nil
}

func (r *SessionKeyRing) Issue(
	accountID string,
	sessionEpoch uint64,
	now time.Time,
	identityLifetime time.Duration,
	authzLifetime time.Duration,
	plan string,
	entitlementExpiry time.Time,
) (string, SessionClaims, error) {
	if accountID == "" || sessionEpoch == 0 || identityLifetime <= 0 || authzLifetime < 0 || authzLifetime > identityLifetime {
		return "", SessionClaims{}, fmt.Errorf("invalid session parameters")
	}
	if authzLifetime > 0 && (plan == "" || !entitlementExpiry.After(now)) {
		return "", SessionClaims{}, fmt.Errorf("an authorization lease requires a current entitlement")
	}
	if authzLifetime == 0 && (plan != "" || !entitlementExpiry.IsZero()) {
		return "", SessionClaims{}, fmt.Errorf("identity-only sessions cannot carry entitlement claims")
	}
	nonceBytes := make([]byte, 16)
	if _, err := rand.Read(nonceBytes); err != nil {
		return "", SessionClaims{}, err
	}
	authzExpiry := now
	if authzLifetime > 0 {
		authzExpiry = now.Add(authzLifetime)
		if entitlementExpiry.Before(authzExpiry) {
			authzExpiry = entitlementExpiry
		}
		if authzExpiry.Unix() <= now.Unix() {
			return "", SessionClaims{}, fmt.Errorf("entitlement expires too soon for an authorization lease")
		}
	}
	claims := SessionClaims{
		Version:      1,
		Issuer:       r.issuer,
		Audience:     r.audience,
		Subject:      accountID,
		SessionEpoch: sessionEpoch,
		Plan:         plan,
		IssuedAt:     now.Unix(),
		AuthzExpiry:  authzExpiry.Unix(),
		Expiry:       now.Add(identityLifetime).Unix(),
		Nonce:        base64.RawURLEncoding.EncodeToString(nonceBytes),
	}
	if !entitlementExpiry.IsZero() {
		claims.EntitlementExpiry = entitlementExpiry.Unix()
	}
	token, err := r.signClaims(claims)
	if err != nil {
		return "", SessionClaims{}, err
	}
	return token, claims, nil
}

func (r *SessionKeyRing) RefreshAuthorization(
	claims SessionClaims,
	now time.Time,
	authzLifetime time.Duration,
	plan string,
	entitlementExpiry time.Time,
) (string, SessionClaims, error) {
	if claims.Version != 1 || claims.Issuer != r.issuer || claims.Audience != r.audience ||
		claims.Subject == "" || claims.Nonce == "" || now.Unix() >= claims.Expiry ||
		authzLifetime <= 0 || plan == "" || !entitlementExpiry.After(now) {
		return "", SessionClaims{}, fmt.Errorf("invalid authorization refresh")
	}
	authzExpiry := now.Add(authzLifetime)
	if entitlementExpiry.Before(authzExpiry) {
		authzExpiry = entitlementExpiry
	}
	if authzExpiry.Unix() <= now.Unix() {
		return "", SessionClaims{}, fmt.Errorf("entitlement expires too soon for an authorization lease")
	}
	claims.Plan = plan
	claims.AuthzExpiry = authzExpiry.Unix()
	claims.EntitlementExpiry = entitlementExpiry.Unix()
	token, err := r.signClaims(claims)
	if err != nil {
		return "", SessionClaims{}, err
	}
	return token, claims, nil
}

func (r *SessionKeyRing) signClaims(claims SessionClaims) (string, error) {
	header := sessionHeader{Algorithm: "HS256", Type: "PWS", KeyID: r.currentKeyID}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	encodedHeader := base64.RawURLEncoding.EncodeToString(headerJSON)
	encodedClaims := base64.RawURLEncoding.EncodeToString(claimsJSON)
	signed := encodedHeader + "." + encodedClaims
	signature := signHMAC(r.keys[r.currentKeyID], signed)
	return signed + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

func (r *SessionKeyRing) Verify(token string, now time.Time) (SessionClaims, error) {
	if len(token) == 0 || len(token) > 8192 {
		return SessionClaims{}, ErrInvalidSession
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return SessionClaims{}, ErrInvalidSession
	}
	headerBytes, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil || len(headerBytes) > 1024 {
		return SessionClaims{}, ErrInvalidSession
	}
	var header sessionHeader
	if err := strictDecode(headerBytes, &header); err != nil || header.Algorithm != "HS256" || header.Type != "PWS" {
		return SessionClaims{}, ErrInvalidSession
	}
	key, ok := r.keys[header.KeyID]
	if !ok {
		return SessionClaims{}, ErrInvalidSession
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || !hmac.Equal(signature, signHMAC(key, parts[0]+"."+parts[1])) {
		return SessionClaims{}, ErrInvalidSession
	}
	claimsBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil || len(claimsBytes) > 4096 {
		return SessionClaims{}, ErrInvalidSession
	}
	var claims SessionClaims
	if err := strictDecode(claimsBytes, &claims); err != nil {
		return SessionClaims{}, ErrInvalidSession
	}
	if claims.Version != 1 || claims.Issuer != r.issuer || claims.Audience != r.audience ||
		claims.Subject == "" || claims.SessionEpoch == 0 || claims.Nonce == "" ||
		claims.IssuedAt <= 0 || claims.AuthzExpiry < claims.IssuedAt || claims.Expiry < claims.AuthzExpiry ||
		claims.IssuedAt > now.Add(time.Minute).Unix() {
		return SessionClaims{}, ErrInvalidSession
	}
	if claims.AuthzExpiry > claims.IssuedAt {
		if claims.Plan == "" || claims.EntitlementExpiry < claims.AuthzExpiry {
			return SessionClaims{}, ErrInvalidSession
		}
	} else if claims.Plan != "" || claims.EntitlementExpiry != 0 {
		return SessionClaims{}, ErrInvalidSession
	}
	if now.Unix() >= claims.Expiry {
		return SessionClaims{}, ErrExpiredSession
	}
	return claims, nil
}

func (claims SessionClaims) AuthorizationLeaseValid(now time.Time) bool {
	return claims.Plan != "" && now.Unix() < claims.AuthzExpiry && now.Unix() < claims.EntitlementExpiry
}

func signHMAC(key []byte, value string) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(value))
	return mac.Sum(nil)
}

func strictDecode(body []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("trailing JSON data")
	}
	return nil
}
