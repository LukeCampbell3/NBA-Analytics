package ledger

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

func TestOneHundredConcurrentResolutionsCreateOneLogicalAccount(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	store := storage.NewAccountStore(objects)
	service, err := NewService(store, []byte("current-index-key-material-32-bytes!!"), nil)
	if err != nil {
		t.Fatal(err)
	}

	const workers = 100
	start := make(chan struct{})
	results := make(chan account.Account, workers)
	errorsFound := make(chan error, workers)
	var wait sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			value, resolveErr := service.ResolveDiscord(ctx, "discord-user-123", "member")
			if resolveErr != nil {
				errorsFound <- resolveErr
				return
			}
			results <- value
		}()
	}
	close(start)
	wait.Wait()
	close(results)
	close(errorsFound)
	for err := range errorsFound {
		t.Errorf("ResolveDiscord() error = %v", err)
	}

	var accountID string
	count := 0
	for result := range results {
		count++
		if accountID == "" {
			accountID = result.AccountID
		}
		if result.AccountID != accountID {
			t.Errorf("resolved account %q, want %q", result.AccountID, accountID)
		}
	}
	if count != workers {
		t.Fatalf("successful resolutions = %d, want %d", count, workers)
	}
	if got := objects.CountPrefix("indexes/discord/"); got != 1 {
		t.Fatalf("identity indexes = %d, want 1", got)
	}
	if got := objects.CountPrefix("accounts/"); got != 1 {
		t.Fatalf("canonical accounts = %d, want 1", got)
	}
}

func TestInterruptedIndexFirstCreationIsRepaired(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	store := storage.NewAccountStore(objects)
	key := []byte("current-index-key-material-32-bytes!!")
	digest := security.IdentityHMAC(key, "discord", "discord-user-456")
	createdAt := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	index := account.IdentityIndex{
		SchemaVersion: account.SchemaVersion,
		AccountID:     "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa",
		CreatedAt:     createdAt,
	}
	if err := store.CreateIdentityIndex(ctx, 1, digest, index); err != nil {
		t.Fatal(err)
	}
	service, _ := NewService(store, key, nil)
	value, err := service.ResolveDiscord(ctx, "discord-user-456", "repaired")
	if err != nil {
		t.Fatal(err)
	}
	if value.AccountID != index.AccountID || value.Status != account.StatusPending {
		t.Fatalf("repaired account = %#v", value)
	}
	if got := objects.CountPrefix("accounts/"); got != 1 {
		t.Fatalf("canonical accounts = %d, want 1", got)
	}
}

func TestConcurrentIndependentMutationsPreserveBothChanges(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	store := storage.NewAccountStore(objects)
	now := time.Now().UTC()
	value := account.Account{
		SchemaVersion: account.SchemaVersion,
		Revision:      1,
		AccountID:     "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa",
		Status:        account.StatusPending,
		SessionEpoch:  1,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	if err := store.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	var wait sync.WaitGroup
	mutationErrors := make(chan error, 2)
	mutations := []struct {
		id string
		fn func(*account.Account) error
	}{
		{"set-name", func(value *account.Account) error { value.DisplayName = "updated"; return nil }},
		{"revoke-sessions", func(value *account.Account) error { value.SessionEpoch = 2; return nil }},
	}
	for _, mutation := range mutations {
		mutation := mutation
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			mutationErrors <- MutateAccount(ctx, store, value.AccountID, mutation.id, mutation.fn)
		}()
	}
	close(start)
	wait.Wait()
	close(mutationErrors)
	for err := range mutationErrors {
		if err != nil {
			t.Errorf("mutation failed: %v", err)
		}
	}
	updated, _, err := store.GetAccount(ctx, value.AccountID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.DisplayName != "updated" || updated.SessionEpoch != 2 || updated.Revision != 3 {
		t.Fatalf("updated account = %#v", updated)
	}
	if got := objects.CountPrefix("account-history/"); got != 2 {
		t.Fatalf("history records = %d, want 2", got)
	}
}

func TestMutationRequiresStableID(t *testing.T) {
	err := MutateAccount(context.Background(), nil, "account", "", func(*account.Account) error { return nil })
	if err == nil || errors.Is(err, storage.ErrConflict) {
		t.Fatalf("missing mutation id error = %v", err)
	}
}
