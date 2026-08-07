package config

import (
	"encoding/base64"
	"fmt"
	"net/url"
	"strconv"
	"strings"

	"github.com/jcthi/nba-analytics/paywall/auth"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type LookupEnv func(string) (string, bool)

type Gateway struct {
	PublicOrigin     string
	PublicPathPrefix string
	Discord          auth.DiscordConfig
	StateR2          storage.R2Config
	ContentR2        storage.R2Config
	IndexKeyCurrent  []byte
	IndexKeyPrevious []byte
	SessionKeys      *security.SessionKeyRing
	CSRFKey          []byte
	AllowedRedirects []string
	IdentityLifetime int
	AuthzLifetime    int
	PaymentSecretKey string
	PaymentPriceID   string
	PaymentPlan      string
	PIICurrentKeyID  string
	PIIKeys          map[string][]byte
}

type Webhook struct {
	StateR2          storage.R2Config
	PaymentSecretKey string
	SigningSecret    string
	PaymentPriceID   string
	PaymentPlan      string
	PIICurrentKeyID  string
	PIICurrentKey    []byte
	MaximumBody      int64
}

func LoadGateway(lookup LookupEnv) (Gateway, error) {
	publicOrigin, err := required(lookup, "PUBLIC_ORIGIN")
	if err != nil {
		return Gateway{}, err
	}
	parsedOrigin, err := url.Parse(publicOrigin)
	if err != nil || parsedOrigin.Scheme != "https" || parsedOrigin.Host == "" || parsedOrigin.Path != "" {
		return Gateway{}, fmt.Errorf("PUBLIC_ORIGIN must be an HTTPS origin without a path")
	}
	publicPathPrefix, err := required(lookup, "GATEWAY_PUBLIC_PATH_PREFIX")
	if err != nil {
		return Gateway{}, err
	}
	if !strings.HasPrefix(publicPathPrefix, "/") || strings.HasSuffix(publicPathPrefix, "/") ||
		strings.ContainsAny(publicPathPrefix, "?#") || strings.Contains(publicPathPrefix, "//") {
		return Gateway{}, fmt.Errorf("GATEWAY_PUBLIC_PATH_PREFIX must be an absolute path without a trailing slash")
	}
	discord, err := loadDiscord(lookup)
	if err != nil {
		return Gateway{}, err
	}
	stateR2, err := loadR2(lookup, "R2_STATE")
	if err != nil {
		return Gateway{}, err
	}
	contentR2, err := loadR2(lookup, "R2_CONTENT")
	if err != nil {
		return Gateway{}, err
	}
	if stateR2.Bucket == contentR2.Bucket ||
		(stateR2.AccessKeyID == contentR2.AccessKeyID && stateR2.SecretAccessKey == contentR2.SecretAccessKey) {
		return Gateway{}, fmt.Errorf("state and content buckets and credentials must be separate")
	}
	indexCurrent, err := requiredKey(lookup, "DISCORD_INDEX_KEY_CURRENT", 32)
	if err != nil {
		return Gateway{}, err
	}
	indexPrevious, err := optionalKey(lookup, "DISCORD_INDEX_KEY_PREVIOUS", 32)
	if err != nil {
		return Gateway{}, err
	}
	csrfKey, err := requiredKey(lookup, "CSRF_SIGNING_KEY", 32)
	if err != nil {
		return Gateway{}, err
	}
	sessionRing, err := loadSessionKeys(lookup)
	if err != nil {
		return Gateway{}, err
	}
	paymentSecret, err := required(lookup, "PAYMENT_SECRET_KEY")
	if err != nil {
		return Gateway{}, err
	}
	priceID, err := required(lookup, "PAYMENT_PRICE_ID")
	if err != nil {
		return Gateway{}, err
	}
	piiCurrentID, piiKeys, err := loadPIIKeys(lookup)
	if err != nil {
		return Gateway{}, err
	}
	redirects := []string{"/app/", "/payment/return"}
	if configured, ok := lookup("ALLOWED_LOGIN_REDIRECTS"); ok && strings.TrimSpace(configured) != "" {
		redirects = nil
		for _, item := range strings.Split(configured, ",") {
			redirects = append(redirects, strings.TrimSpace(item))
		}
	}
	return Gateway{
		PublicOrigin: publicOrigin, PublicPathPrefix: publicPathPrefix,
		Discord: discord, StateR2: stateR2, ContentR2: contentR2,
		IndexKeyCurrent: indexCurrent, IndexKeyPrevious: indexPrevious,
		SessionKeys: sessionRing, CSRFKey: csrfKey, AllowedRedirects: redirects,
		IdentityLifetime: 7 * 24 * 60 * 60, AuthzLifetime: 10 * 60,
		PaymentSecretKey: paymentSecret,
		PaymentPriceID:   priceID, PaymentPlan: "individual",
		PIICurrentKeyID: piiCurrentID, PIIKeys: piiKeys,
	}, nil
}

func LoadWebhook(lookup LookupEnv) (Webhook, error) {
	stateR2, err := loadR2(lookup, "R2_STATE")
	if err != nil {
		return Webhook{}, err
	}
	paymentSecret, err := required(lookup, "PAYMENT_SECRET_KEY")
	if err != nil {
		return Webhook{}, err
	}
	secret, err := required(lookup, "PAYMENT_WEBHOOK_SECRET")
	if err != nil {
		return Webhook{}, err
	}
	priceID, err := required(lookup, "PAYMENT_PRICE_ID")
	if err != nil {
		return Webhook{}, err
	}
	piiCurrentID, piiKeys, err := loadPIIKeys(lookup)
	if err != nil {
		return Webhook{}, err
	}
	maximum := int64(256 * 1024)
	if raw, ok := lookup("PAYMENT_WEBHOOK_MAX_BYTES"); ok && raw != "" {
		parsed, parseErr := strconv.ParseInt(raw, 10, 64)
		if parseErr != nil || parsed < 1024 || parsed > 1024*1024 {
			return Webhook{}, fmt.Errorf("PAYMENT_WEBHOOK_MAX_BYTES must be between 1024 and 1048576")
		}
		maximum = parsed
	}
	return Webhook{
		StateR2: stateR2, PaymentSecretKey: paymentSecret, SigningSecret: secret,
		PaymentPriceID: priceID, PaymentPlan: "individual",
		PIICurrentKeyID: piiCurrentID, PIICurrentKey: piiKeys[piiCurrentID], MaximumBody: maximum,
	}, nil
}

func loadDiscord(lookup LookupEnv) (auth.DiscordConfig, error) {
	clientID, err := required(lookup, "DISCORD_CLIENT_ID")
	if err != nil {
		return auth.DiscordConfig{}, err
	}
	clientSecret, err := required(lookup, "DISCORD_CLIENT_SECRET")
	if err != nil {
		return auth.DiscordConfig{}, err
	}
	redirectURI, err := required(lookup, "DISCORD_REDIRECT_URI")
	if err != nil {
		return auth.DiscordConfig{}, err
	}
	return auth.DiscordConfig{ClientID: clientID, ClientSecret: clientSecret, RedirectURI: redirectURI}, nil
}

func loadR2(lookup LookupEnv, prefix string) (storage.R2Config, error) {
	endpoint, err := required(lookup, prefix+"_ENDPOINT")
	if err != nil {
		return storage.R2Config{}, err
	}
	accessKey, err := required(lookup, prefix+"_ACCESS_KEY_ID")
	if err != nil {
		return storage.R2Config{}, err
	}
	secretKey, err := required(lookup, prefix+"_SECRET_ACCESS_KEY")
	if err != nil {
		return storage.R2Config{}, err
	}
	bucket, err := required(lookup, prefix+"_BUCKET")
	if err != nil {
		return storage.R2Config{}, err
	}
	return storage.R2Config{Endpoint: endpoint, AccessKeyID: accessKey, SecretAccessKey: secretKey, Bucket: bucket}, nil
}

func loadSessionKeys(lookup LookupEnv) (*security.SessionKeyRing, error) {
	currentID, err := required(lookup, "SESSION_KEY_CURRENT_ID")
	if err != nil {
		return nil, err
	}
	currentKey, err := requiredKey(lookup, "SESSION_KEY_CURRENT", 32)
	if err != nil {
		return nil, err
	}
	keys := map[string][]byte{currentID: currentKey}
	if previousID, ok := lookup("SESSION_KEY_PREVIOUS_ID"); ok && previousID != "" {
		previousKey, keyErr := requiredKey(lookup, "SESSION_KEY_PREVIOUS", 32)
		if keyErr != nil {
			return nil, keyErr
		}
		if previousID == currentID {
			return nil, fmt.Errorf("current and previous session key IDs must differ")
		}
		keys[previousID] = previousKey
	} else if raw, ok := lookup("SESSION_KEY_PREVIOUS"); ok && raw != "" {
		return nil, fmt.Errorf("SESSION_KEY_PREVIOUS_ID is required when SESSION_KEY_PREVIOUS is set")
	}
	issuer, err := required(lookup, "SESSION_ISSUER")
	if err != nil {
		return nil, err
	}
	audience, err := required(lookup, "SESSION_AUDIENCE")
	if err != nil {
		return nil, err
	}
	return security.NewSessionKeyRing(currentID, keys, issuer, audience)
}

func loadPIIKeys(lookup LookupEnv) (string, map[string][]byte, error) {
	currentID, err := required(lookup, "PII_ENCRYPTION_KEY_CURRENT_ID")
	if err != nil {
		return "", nil, err
	}
	currentKey, err := exactKey(lookup, "PII_ENCRYPTION_KEY_CURRENT", 32)
	if err != nil {
		return "", nil, err
	}
	keys := map[string][]byte{currentID: currentKey}
	if previousID, ok := lookup("PII_ENCRYPTION_KEY_PREVIOUS_ID"); ok && previousID != "" {
		previousKey, keyErr := exactKey(lookup, "PII_ENCRYPTION_KEY_PREVIOUS", 32)
		if keyErr != nil {
			return "", nil, keyErr
		}
		if previousID == currentID {
			return "", nil, fmt.Errorf("current and previous PII key IDs must differ")
		}
		keys[previousID] = previousKey
	} else if raw, ok := lookup("PII_ENCRYPTION_KEY_PREVIOUS"); ok && raw != "" {
		return "", nil, fmt.Errorf("PII_ENCRYPTION_KEY_PREVIOUS_ID is required when PII_ENCRYPTION_KEY_PREVIOUS is set")
	}
	return currentID, keys, nil
}

func required(lookup LookupEnv, name string) (string, error) {
	value, ok := lookup(name)
	value = strings.TrimSpace(value)
	if !ok || value == "" {
		return "", fmt.Errorf("%s is required", name)
	}
	return value, nil
}

func requiredKey(lookup LookupEnv, name string, minimum int) ([]byte, error) {
	value, err := required(lookup, name)
	if err != nil {
		return nil, err
	}
	decoded, err := decodeKey(value)
	if err != nil || len(decoded) < minimum {
		return nil, fmt.Errorf("%s must be base64-encoded and decode to at least %d bytes", name, minimum)
	}
	return decoded, nil
}

func optionalKey(lookup LookupEnv, name string, minimum int) ([]byte, error) {
	value, ok := lookup(name)
	if !ok || strings.TrimSpace(value) == "" {
		return nil, nil
	}
	decoded, err := decodeKey(strings.TrimSpace(value))
	if err != nil || len(decoded) < minimum {
		return nil, fmt.Errorf("%s must be base64-encoded and decode to at least %d bytes", name, minimum)
	}
	return decoded, nil
}

func exactKey(lookup LookupEnv, name string, length int) ([]byte, error) {
	value, err := required(lookup, name)
	if err != nil {
		return nil, err
	}
	decoded, err := decodeKey(value)
	if err != nil || len(decoded) != length {
		return nil, fmt.Errorf("%s must be base64-encoded and decode to exactly %d bytes", name, length)
	}
	return decoded, nil
}

func decodeKey(value string) ([]byte, error) {
	decoded, err := base64.StdEncoding.DecodeString(value)
	if err == nil {
		return decoded, nil
	}
	return base64.RawStdEncoding.DecodeString(value)
}
