package storage

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
)

func validTestAccount() account.Account {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	return account.Account{
		SchemaVersion: account.SchemaVersion,
		Revision:      1,
		AccountID:     "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa",
		Status:        account.StatusPending,
		SessionEpoch:  1,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
}

func TestStaleETagUpdateFails(t *testing.T) {
	ctx := context.Background()
	store := NewAccountStore(NewMemoryStore())
	value := validTestAccount()
	if err := store.CreateAccount(ctx, value); err != nil {
		t.Fatal(err)
	}
	first, firstETag, err := store.GetAccount(ctx, value.AccountID)
	if err != nil {
		t.Fatal(err)
	}
	second := first
	first.Revision++
	first.DisplayName = "first"
	if err := store.UpdateAccountIfMatch(ctx, first, firstETag); err != nil {
		t.Fatal(err)
	}
	second.Revision++
	second.DisplayName = "stale"
	if err := store.UpdateAccountIfMatch(ctx, second, firstETag); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale update error = %v, want ErrConflict", err)
	}
}

func TestMalformedAccountFailsClosed(t *testing.T) {
	ctx := context.Background()
	objects := NewMemoryStore()
	store := NewAccountStore(objects)
	value := validTestAccount()
	key, _ := accountKey(value.AccountID)
	if _, err := objects.Put(ctx, key, []byte(`{"schema_version":1,"account_id":"acc_aaaaaaaaaaaaaaaaaaaaaaaaaa","status":"invented"}`), PutCondition{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.GetAccount(ctx, value.AccountID); !errors.Is(err, ErrMalformed) {
		t.Fatalf("malformed account error = %v, want ErrMalformed", err)
	}
}
