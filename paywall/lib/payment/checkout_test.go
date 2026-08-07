package payment

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type idempotentCheckoutProvider struct {
	mu        sync.Mutex
	checkouts map[string]Checkout
	snapshot  EntitlementSnapshot
}

func (p *idempotentCheckoutProvider) CreateCheckout(_ context.Context, _ account.Account, key string) (Checkout, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if checkout, ok := p.checkouts[key]; ok {
		return checkout, nil
	}
	checkout := Checkout{ID: "cs_" + key, URL: "https://checkout.example/" + key}
	p.checkouts[key] = checkout
	return checkout, nil
}

func (p *idempotentCheckoutProvider) VerifyWebhook([]byte, string) (ProviderEvent, error) {
	return ProviderEvent{}, nil
}

func (p *idempotentCheckoutProvider) GetAuthoritativeEntitlement(context.Context, SubscriptionReference) (EntitlementSnapshot, error) {
	return p.snapshot, nil
}

func TestConcurrentCheckoutRequestsReuseOneProviderMutation(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	accounts := storage.NewAccountStore(objects)
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	value := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 1,
		AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", Status: account.StatusPending,
		SessionEpoch: 1, CreatedAt: now, UpdatedAt: now,
	}
	if err := accounts.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	provider := &idempotentCheckoutProvider{checkouts: make(map[string]Checkout)}
	service, err := NewCheckoutService(provider, accounts, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	service.now = func() time.Time { return now }

	const workers = 20
	start := make(chan struct{})
	results := make(chan Checkout, workers)
	errorsFound := make(chan error, workers)
	var wait sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			checkout, createErr := service.Create(ctx, value.AccountID)
			if createErr != nil {
				errorsFound <- createErr
				return
			}
			results <- checkout
		}()
	}
	close(start)
	wait.Wait()
	close(results)
	close(errorsFound)
	for err := range errorsFound {
		t.Errorf("checkout create failed: %v", err)
	}
	var checkoutID string
	for result := range results {
		if checkoutID == "" {
			checkoutID = result.ID
		}
		if result.ID != checkoutID {
			t.Errorf("checkout ID = %q, want %q", result.ID, checkoutID)
		}
	}
	provider.mu.Lock()
	providerMutations := len(provider.checkouts)
	provider.mu.Unlock()
	if providerMutations != 1 {
		t.Fatalf("logical provider mutations = %d, want 1", providerMutations)
	}
}
