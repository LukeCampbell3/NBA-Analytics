package payment

import (
	"context"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type reconciliationProvider struct {
	snapshot EntitlementSnapshot
	calls    int
}

func (provider *reconciliationProvider) GetAuthoritativeEntitlement(context.Context, SubscriptionReference) (EntitlementSnapshot, error) {
	provider.calls++
	return provider.snapshot, nil
}

func TestReconcilerRefreshesStaleSubscriptionAndThenUsesSnapshot(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	accounts := storage.NewAccountStore(objects)
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	accountID := "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa"
	key := []byte("0123456789abcdef0123456789abcdef")
	subscription, err := security.EncryptField("pii-current", key, accountID, "sub_123")
	if err != nil {
		t.Fatal(err)
	}
	value := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 1, AccountID: accountID,
		Status: account.StatusActive, SessionEpoch: 1,
		Entitlement: account.Entitlement{
			Plan: "individual", Source: "stripe", ValidUntil: now.Add(30 * 24 * time.Hour),
			ProviderVerifiedAt: now.Add(-25 * time.Hour), ProviderUpdatedAt: now.Add(-25 * time.Hour),
		},
		Payment: account.Payment{SubscriptionID: subscription}, CreatedAt: now.Add(-time.Hour), UpdatedAt: now.Add(-time.Hour),
	}
	if err := accounts.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	provider := &reconciliationProvider{snapshot: EntitlementSnapshot{
		Status: account.StatusCanceled, Plan: "individual", Source: "stripe",
		ValidFrom: now.Add(-30 * 24 * time.Hour), ValidUntil: now,
		ProviderVerifiedAt: now, ProviderUpdatedAt: now,
		CustomerID: "cus_123", SubscriptionID: "sub_123",
	}}
	reconciler, err := NewReconciler(provider, accounts, "pii-current", map[string][]byte{"pii-current": key}, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	reconciler.now = func() time.Time { return now }
	updated, err := reconciler.Reconcile(ctx, value)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Status != account.StatusCanceled || updated.Revision != 2 || provider.calls != 1 || updated.Payment.CustomerID == nil {
		t.Fatalf("reconciled account = %#v, provider calls = %d", updated, provider.calls)
	}
	if _, err := reconciler.Reconcile(ctx, updated); err != nil {
		t.Fatal(err)
	}
	if provider.calls != 1 {
		t.Fatalf("fresh snapshot called provider %d times", provider.calls)
	}
}

func TestReconcilerDoesNotCallProviderForNeverSubscribedAccount(t *testing.T) {
	provider := &reconciliationProvider{}
	accounts := storage.NewAccountStore(storage.NewMemoryStore())
	reconciler, err := NewReconciler(
		provider, accounts, "pii-current",
		map[string][]byte{"pii-current": []byte("0123456789abcdef0123456789abcdef")}, 24*time.Hour,
	)
	if err != nil {
		t.Fatal(err)
	}
	value := account.Account{AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", Status: account.StatusPending}
	if _, err := reconciler.Reconcile(context.Background(), value); err != nil {
		t.Fatal(err)
	}
	if provider.calls != 0 {
		t.Fatalf("pending account caused %d provider calls", provider.calls)
	}
}
