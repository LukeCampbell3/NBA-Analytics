package config

import (
	"encoding/base64"
	"testing"
)

func testEnvironment() map[string]string {
	key := base64.StdEncoding.EncodeToString([]byte("0123456789abcdef0123456789abcdef"))
	return map[string]string{
		"PUBLIC_ORIGIN":                 "https://example.com",
		"GATEWAY_PUBLIC_PATH_PREFIX":    "/functions/paywall/gateway",
		"DISCORD_CLIENT_ID":             "discord-client",
		"DISCORD_CLIENT_SECRET":         "discord-secret",
		"DISCORD_REDIRECT_URI":          "https://example.com/auth/discord/callback",
		"R2_STATE_ENDPOINT":             "https://state.r2.cloudflarestorage.com",
		"R2_STATE_ACCESS_KEY_ID":        "state-access",
		"R2_STATE_SECRET_ACCESS_KEY":    "state-secret",
		"R2_STATE_BUCKET":               "state-bucket",
		"R2_CONTENT_ENDPOINT":           "https://content.r2.cloudflarestorage.com",
		"R2_CONTENT_ACCESS_KEY_ID":      "content-access",
		"R2_CONTENT_SECRET_ACCESS_KEY":  "content-secret",
		"R2_CONTENT_BUCKET":             "content-bucket",
		"DISCORD_INDEX_KEY_CURRENT":     key,
		"SESSION_KEY_CURRENT_ID":        "session-current",
		"SESSION_KEY_CURRENT":           key,
		"SESSION_ISSUER":                "example.com",
		"SESSION_AUDIENCE":              "paid-site",
		"CSRF_SIGNING_KEY":              key,
		"PAYMENT_SECRET_KEY":            "sk_test",
		"PAYMENT_WEBHOOK_SECRET":        "whsec_test",
		"PAYMENT_PRICE_ID":              "price_123",
		"PII_ENCRYPTION_KEY_CURRENT_ID": "pii-current",
		"PII_ENCRYPTION_KEY_CURRENT":    key,
	}
}

func lookupFrom(values map[string]string) LookupEnv {
	return func(name string) (string, bool) {
		value, ok := values[name]
		return value, ok
	}
}

func TestGatewayConfigurationRequiresCredentialSeparation(t *testing.T) {
	values := testEnvironment()
	if _, err := LoadGateway(lookupFrom(values)); err != nil {
		t.Fatal(err)
	}
	values["R2_CONTENT_ACCESS_KEY_ID"] = values["R2_STATE_ACCESS_KEY_ID"]
	values["R2_CONTENT_SECRET_ACCESS_KEY"] = values["R2_STATE_SECRET_ACCESS_KEY"]
	if _, err := LoadGateway(lookupFrom(values)); err == nil {
		t.Fatal("shared state/content credentials were accepted")
	}
}

func TestGatewayConfigurationRejectsUnencodedKeys(t *testing.T) {
	values := testEnvironment()
	values["SESSION_KEY_CURRENT"] = "not-base64***"
	if _, err := LoadGateway(lookupFrom(values)); err == nil {
		t.Fatal("invalid session key was accepted")
	}
}

func TestWebhookConfigurationHasStrictBodyBounds(t *testing.T) {
	values := testEnvironment()
	values["PAYMENT_WEBHOOK_SECRET"] = "whsec_test"
	values["PAYMENT_WEBHOOK_MAX_BYTES"] = "1048577"
	if _, err := LoadWebhook(lookupFrom(values)); err == nil {
		t.Fatal("oversized webhook limit was accepted")
	}
	values["PAYMENT_WEBHOOK_MAX_BYTES"] = "262144"
	config, err := LoadWebhook(lookupFrom(values))
	if err != nil || config.MaximumBody != 262144 {
		t.Fatalf("webhook config = %#v, %v", config, err)
	}
}
