package auth

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type countingAccountReader struct {
	store *storage.AccountStore
	reads int
}

type failingReconciler struct{}

func (failingReconciler) Reconcile(context.Context, account.Account) (account.Account, error) {
	return account.Account{}, errors.New("provider unavailable")
}

func (r *countingAccountReader) GetAccount(ctx context.Context, accountID string) (account.Account, string, error) {
	r.reads++
	return r.store.GetAccount(ctx, accountID)
}

func TestAuthorizationLeaseIsGrantedOnlyAfterAccountAuthorization(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	key := []byte("session-signing-key-material-32-bytes!")
	keys, err := security.NewSessionKeyRing("current", map[string][]byte{"current": key}, "example.com", "paid-site")
	if err != nil {
		t.Fatal(err)
	}
	identityToken, identityClaims, err := keys.Issue(
		"acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", 1, now, 7*24*time.Hour, 0, "", time.Time{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if identityClaims.AuthorizationLeaseValid(now) {
		t.Fatal("identity-only session unexpectedly has an authorization lease")
	}

	objects := storage.NewMemoryStore()
	store := storage.NewAccountStore(objects)
	value := account.Account{
		SchemaVersion: account.SchemaVersion,
		Revision:      1,
		AccountID:     identityClaims.Subject,
		Status:        account.StatusActive,
		SessionEpoch:  1,
		Entitlement: account.Entitlement{
			Plan:       "individual",
			ValidUntil: now.Add(30 * 24 * time.Hour),
		},
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := store.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	reader := &countingAccountReader{store: store}
	authorizer, err := NewAuthorizer(keys, reader, 7*24*time.Hour, 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	authorizer.now = func() time.Time { return now }
	refreshed, err := authorizer.Authorize(ctx, identityToken, false)
	if err != nil || refreshed.RefreshedToken == "" || refreshed.Plan != "individual" {
		t.Fatalf("authorization refresh = %#v, %v", refreshed, err)
	}
	if reader.reads != 1 {
		t.Fatalf("account reads = %d, want 1", reader.reads)
	}
	if refreshed.Claims.Nonce != identityClaims.Nonce {
		t.Fatal("authorization refresh changed the session nonce and invalidated CSRF binding")
	}
	if _, err := authorizer.Authorize(ctx, refreshed.RefreshedToken, false); err != nil {
		t.Fatal(err)
	}
	if reader.reads != 1 {
		t.Fatalf("valid local lease performed an account read; reads = %d", reader.reads)
	}

	current, etag, _ := store.GetAccount(ctx, value.AccountID)
	current.Revision++
	current.SessionEpoch++
	current.Status = account.StatusSuspended
	if err := store.UpdateAccountIfMatch(ctx, current, etag); err != nil {
		t.Fatal(err)
	}
	if _, err := authorizer.Authorize(ctx, refreshed.RefreshedToken, true); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("fresh authorization after suspension error = %v", err)
	}
}

func TestPendingAccountCannotReceiveAuthorizationLease(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	keys, _ := security.NewSessionKeyRing("current", map[string][]byte{
		"current": []byte("session-signing-key-material-32-bytes!"),
	}, "example.com", "paid-site")
	token, claims, _ := keys.Issue("acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", 1, now, 7*24*time.Hour, 0, "", time.Time{})
	store := storage.NewAccountStore(storage.NewMemoryStore())
	value := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 1, AccountID: claims.Subject,
		Status: account.StatusPending, SessionEpoch: 1, CreatedAt: now, UpdatedAt: now,
	}
	if err := store.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	authorizer, _ := NewAuthorizer(keys, store, 7*24*time.Hour, 10*time.Minute)
	authorizer.now = func() time.Time { return now }
	if _, err := authorizer.Authorize(ctx, token, false); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("pending account authorization error = %v", err)
	}
}

func TestAuthorizationRefreshFailsClosedWhenReconciliationFails(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	keys, _ := security.NewSessionKeyRing("current", map[string][]byte{
		"current": []byte("session-signing-key-material-32-bytes!"),
	}, "example.com", "paid-site")
	token, claims, _ := keys.Issue("acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", 1, now, 7*24*time.Hour, 0, "", time.Time{})
	store := storage.NewAccountStore(storage.NewMemoryStore())
	value := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 1, AccountID: claims.Subject,
		Status: account.StatusActive, SessionEpoch: 1,
		Entitlement: account.Entitlement{Plan: "individual", ValidUntil: now.Add(24 * time.Hour)},
		CreatedAt:   now, UpdatedAt: now,
	}
	if err := store.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	authorizer, err := NewReconcilingAuthorizer(keys, store, failingReconciler{}, 7*24*time.Hour, 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	authorizer.now = func() time.Time { return now }
	if _, err := authorizer.Authorize(ctx, token, false); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("reconciliation failure authorization error = %v", err)
	}
}
