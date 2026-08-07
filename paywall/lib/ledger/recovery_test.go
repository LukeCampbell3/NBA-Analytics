package ledger

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

func TestRecoverAccountFromExplicitHistoryRevision(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	store := storage.NewAccountStore(objects)
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	accountID := "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa"
	source := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 4, AccountID: accountID,
		DisplayName: "member", Status: account.StatusActive, SessionEpoch: 3,
		CreatedAt: now.Add(-time.Hour), UpdatedAt: now.Add(-time.Minute),
	}
	if err := store.CreateHistory(ctx, account.HistoryRecord{
		SchemaVersion: account.SchemaVersion, MutationID: "source", RecordedAt: now.Add(-time.Minute), Account: source,
	}); err != nil {
		t.Fatal(err)
	}
	recovered, err := RecoverAccount(ctx, store, accountID, 4, "recovery-1", now)
	if err != nil {
		t.Fatal(err)
	}
	if recovered.Revision != 5 || recovered.SessionEpoch != 4 || !recovered.UpdatedAt.Equal(now) {
		t.Fatalf("recovered account = %#v", recovered)
	}
	if _, err := RecoverAccount(ctx, store, accountID, 4, "recovery-1", now); !errors.Is(err, ErrCanonicalAccountExists) {
		t.Fatalf("second recovery error = %v", err)
	}
}

func TestRecoveryRejectsNonLatestHistoryCollision(t *testing.T) {
	ctx := context.Background()
	store := storage.NewAccountStore(storage.NewMemoryStore())
	now := time.Now().UTC()
	accountID := "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa"
	source := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 4, AccountID: accountID,
		Status: account.StatusActive, SessionEpoch: 2, CreatedAt: now.Add(-time.Hour), UpdatedAt: now.Add(-time.Minute),
	}
	if err := store.CreateHistory(ctx, account.HistoryRecord{SchemaVersion: 1, MutationID: "source", RecordedAt: now, Account: source}); err != nil {
		t.Fatal(err)
	}
	newer := source
	newer.Revision = 5
	if err := store.CreateHistory(ctx, account.HistoryRecord{SchemaVersion: 1, MutationID: "newer", RecordedAt: now, Account: newer}); err != nil {
		t.Fatal(err)
	}
	if _, err := RecoverAccount(ctx, store, accountID, 4, "recovery", now); err == nil {
		t.Fatal("recovery overwrote a newer history boundary")
	}
}

func TestSuspendAccountRevokesSessions(t *testing.T) {
	ctx := context.Background()
	store := storage.NewAccountStore(storage.NewMemoryStore())
	now := time.Now().UTC()
	value := account.Account{
		SchemaVersion: account.SchemaVersion, Revision: 1, AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa",
		Status: account.StatusActive, SessionEpoch: 2, CreatedAt: now, UpdatedAt: now,
	}
	if err := store.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	if err := SuspendAccount(ctx, store, value.AccountID, "suspend-1"); err != nil {
		t.Fatal(err)
	}
	updated, _, err := store.GetAccount(ctx, value.AccountID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Status != account.StatusSuspended || updated.SessionEpoch != 3 {
		t.Fatalf("suspended account = %#v", updated)
	}
}
