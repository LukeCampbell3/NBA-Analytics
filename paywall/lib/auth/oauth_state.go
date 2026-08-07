package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/jcthi/nba-analytics/paywall/storage"
)

var ErrInvalidOAuthState = errors.New("invalid oauth state")

type OAuthState struct {
	SchemaVersion      int        `json:"schema_version"`
	StateHash          string     `json:"state_hash"`
	Status             string     `json:"status"`
	RedirectAfterLogin string     `json:"redirect_after_login"`
	CreatedAt          time.Time  `json:"created_at"`
	ExpiresAt          time.Time  `json:"expires_at"`
	ConsumedAt         *time.Time `json:"consumed_at,omitempty"`
}

type OAuthStateService struct {
	objects          storage.ObjectStore
	allowedRedirects map[string]struct{}
	now              func() time.Time
	lifetime         time.Duration
}

func NewOAuthStateService(objects storage.ObjectStore, allowedRedirects []string, lifetime time.Duration) (*OAuthStateService, error) {
	if lifetime <= 0 || lifetime > 15*time.Minute {
		return nil, fmt.Errorf("oauth state lifetime must be between zero and 15 minutes")
	}
	allowlist := make(map[string]struct{}, len(allowedRedirects))
	for _, redirect := range allowedRedirects {
		parsed, parseErr := url.ParseRequestURI(redirect)
		if parseErr != nil || redirect == "" || len(redirect) > 256 ||
			!strings.HasPrefix(redirect, "/") || strings.HasPrefix(redirect, "//") ||
			parsed.IsAbs() || parsed.Host != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
			return nil, fmt.Errorf("invalid oauth redirect allowlist entry")
		}
		allowlist[redirect] = struct{}{}
	}
	return &OAuthStateService{objects: objects, allowedRedirects: allowlist, now: time.Now, lifetime: lifetime}, nil
}

func (s *OAuthStateService) Create(ctx context.Context, redirectAfterLogin string) (rawState string, err error) {
	if _, ok := s.allowedRedirects[redirectAfterLogin]; !ok {
		return "", fmt.Errorf("%w: redirect is not allowed", ErrInvalidOAuthState)
	}
	for attempt := 0; attempt < 3; attempt++ {
		random := make([]byte, 32)
		if _, err := rand.Read(random); err != nil {
			return "", err
		}
		rawState = base64.RawURLEncoding.EncodeToString(random)
		digest := hashOAuthState(rawState)
		now := s.now().UTC()
		record := OAuthState{
			SchemaVersion:      1,
			StateHash:          digest,
			Status:             "pending",
			RedirectAfterLogin: redirectAfterLogin,
			CreatedAt:          now,
			ExpiresAt:          now.Add(s.lifetime),
		}
		body, marshalErr := json.Marshal(record)
		if marshalErr != nil {
			return "", marshalErr
		}
		_, err = s.objects.Put(ctx, "oauth-state/"+digest+".json", body, storage.PutCondition{IfNoneMatch: true})
		if err == nil {
			return rawState, nil
		}
		if !errors.Is(err, storage.ErrConflict) {
			return "", err
		}
	}
	return "", storage.ErrConflictRetriesExhausted
}

func (s *OAuthStateService) Consume(ctx context.Context, rawState string) (OAuthState, error) {
	decoded, err := base64.RawURLEncoding.DecodeString(rawState)
	if err != nil || len(decoded) != 32 {
		return OAuthState{}, ErrInvalidOAuthState
	}
	digest := hashOAuthState(rawState)
	key := "oauth-state/" + digest + ".json"
	object, err := s.objects.Get(ctx, key)
	if errors.Is(err, storage.ErrNotFound) {
		return OAuthState{}, ErrInvalidOAuthState
	}
	if err != nil {
		return OAuthState{}, err
	}
	var record OAuthState
	if err := json.Unmarshal(object.Body, &record); err != nil {
		return OAuthState{}, fmt.Errorf("%w: oauth state", storage.ErrMalformed)
	}
	now := s.now().UTC()
	if record.SchemaVersion != 1 || record.StateHash != digest || record.Status != "pending" ||
		record.CreatedAt.IsZero() || !now.Before(record.ExpiresAt) {
		return OAuthState{}, ErrInvalidOAuthState
	}
	record.Status = "consumed"
	record.ConsumedAt = &now
	body, err := json.Marshal(record)
	if err != nil {
		return OAuthState{}, err
	}
	if _, err := s.objects.Put(ctx, key, body, storage.PutCondition{IfMatch: object.ETag}); err != nil {
		if errors.Is(err, storage.ErrConflict) {
			return OAuthState{}, ErrInvalidOAuthState
		}
		return OAuthState{}, err
	}
	return record, nil
}

func hashOAuthState(rawState string) string {
	hash := sha256.Sum256([]byte(rawState))
	return hex.EncodeToString(hash[:])
}
