package payment

import (
	"context"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type authoritativeProvider struct {
	snapshot EntitlementSnapshot
	lookups  int
}

func (p *authoritativeProvider) CreateCheckout(context.Context, account.Account, string) (Checkout, error) {
	return Checkout{}, nil
}

func (p *authoritativeProvider) VerifyWebhook([]byte, string) (ProviderEvent, error) {
	return ProviderEvent{}, nil
}

func (p *authoritativeProvider) GetAuthoritativeEntitlement(context.Context, SubscriptionReference) (EntitlementSnapshot, error) {
	p.lookups++
	return p.snapshot, nil
}

func TestOutOfOrderEventsReplaceWithAuthoritativeState(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	accounts := storage.NewAccountStore(objects)
	events, err := storage.NewEventStore(objects, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	initial := account.Account{
		SchemaVersion: account.SchemaVersion,
		Revision:      1,
		AccountID:     "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa",
		Status:        account.StatusPending,
		SessionEpoch:  1,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	if err := accounts.CreateAccount(ctx, initial); err != nil {
		t.Fatal(err)
	}
	provider := &authoritativeProvider{snapshot: EntitlementSnapshot{
		Status:             account.StatusActive,
		Plan:               "individual",
		Source:             "stripe",
		ValidFrom:          now,
		ValidUntil:         now.Add(30 * 24 * time.Hour),
		ProviderVerifiedAt: now.Add(time.Second),
		ProviderUpdatedAt:  now,
		CustomerID:         "cus_123",
		SubscriptionID:     "sub_123",
	}}
	processor, err := NewWebhookProcessor(provider, events, accounts, "pii-current", []byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	newer := ProviderEvent{ID: "evt_newer", Type: "customer.subscription.updated", AccountID: initial.AccountID, SubscriptionID: "sub_123"}
	older := ProviderEvent{ID: "evt_older", Type: "customer.subscription.created", AccountID: initial.AccountID, SubscriptionID: "sub_123"}
	if err := processor.Process(ctx, []byte("newer"), newer, "owner-newer"); err != nil {
		t.Fatal(err)
	}
	if err := processor.Process(ctx, []byte("older"), older, "owner-older"); err != nil {
		t.Fatal(err)
	}
	updated, _, err := accounts.GetAccount(ctx, initial.AccountID)
	if err != nil {
		t.Fatal(err)
	}
	if !updated.HasAccess(now.Add(time.Hour)) || updated.Entitlement.ValidUntil != provider.snapshot.ValidUntil {
		t.Fatalf("account did not retain authoritative state: %#v", updated)
	}
	if err := processor.Process(ctx, []byte("older"), older, "owner-replay"); err != nil {
		t.Fatal(err)
	}
	if provider.lookups != 2 {
		t.Fatalf("provider lookups = %d, want 2 (replay must be skipped)", provider.lookups)
	}
}

func TestWebhookCannotResurrectDeletedAccount(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	accounts := storage.NewAccountStore(objects)
	events, _ := storage.NewEventStore(objects, time.Minute)
	now := time.Now().UTC()
	deleted := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 2,
		AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", DisplayName: "deleted-account",
		Status: account.StatusDeleted, SessionEpoch: 2, CreatedAt: now.Add(-time.Hour), UpdatedAt: now,
	}
	if err := accounts.CreateAccount(ctx, deleted); err != nil {
		t.Fatal(err)
	}
	provider := &authoritativeProvider{snapshot: EntitlementSnapshot{
		Status: account.StatusActive, Plan: "individual", Source: "stripe",
		ValidFrom: now, ValidUntil: now.Add(30 * 24 * time.Hour),
		ProviderVerifiedAt: now, ProviderUpdatedAt: now,
		CustomerID: "cus_123", SubscriptionID: "sub_123",
	}}
	processor, err := NewWebhookProcessor(provider, events, accounts, "pii-current", []byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	event := ProviderEvent{ID: "evt_deleted", Type: "customer.subscription.updated", AccountID: deleted.AccountID, SubscriptionID: "sub_123"}
	if err := processor.Process(ctx, []byte("deleted"), event, "owner"); err != nil {
		t.Fatal(err)
	}
	current, _, err := accounts.GetAccount(ctx, deleted.AccountID)
	if err != nil {
		t.Fatal(err)
	}
	if current.Status != account.StatusDeleted || current.Revision != deleted.Revision {
		t.Fatalf("deleted account was changed: %#v", current)
	}
}
