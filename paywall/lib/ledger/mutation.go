package ledger

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type MutationStore interface {
	GetAccount(context.Context, string) (account.Account, string, error)
	UpdateAccountIfMatch(context.Context, account.Account, string) error
	CreateHistory(context.Context, account.HistoryRecord) error
	GetHistory(context.Context, string, uint64) (account.HistoryRecord, error)
}

// MutateAccount serializes a revision by claiming its immutable history object
// before replacing the canonical account. mutationID must be stable across retries.
func MutateAccount(
	ctx context.Context,
	store MutationStore,
	accountID string,
	mutationID string,
	mutate func(*account.Account) error,
) error {
	if mutationID == "" {
		return fmt.Errorf("mutation id is required")
	}
	for attempt := 0; attempt < 5; attempt++ {
		current, etag, err := store.GetAccount(ctx, accountID)
		if err != nil {
			return err
		}
		candidate := current
		if err := mutate(&candidate); err != nil {
			return err
		}
		candidate.Revision = current.Revision + 1
		candidate.UpdatedAt = time.Now().UTC()
		history := account.HistoryRecord{
			SchemaVersion: account.SchemaVersion,
			MutationID:    mutationID,
			RecordedAt:    candidate.UpdatedAt,
			Account:       candidate,
		}
		err = store.CreateHistory(ctx, history)
		if errors.Is(err, storage.ErrConflict) {
			existing, historyErr := store.GetHistory(ctx, accountID, candidate.Revision)
			if historyErr != nil {
				return historyErr
			}
			if existing.MutationID != mutationID {
				if err := mutationJitter(ctx, attempt); err != nil {
					return err
				}
				continue
			}
			candidate = existing.Account
		} else if err != nil {
			return err
		}
		if err := store.UpdateAccountIfMatch(ctx, candidate, etag); err == nil {
			return nil
		} else if !errors.Is(err, storage.ErrConflict) {
			return err
		}
		if err := mutationJitter(ctx, attempt); err != nil {
			return err
		}
	}
	return storage.ErrConflictRetriesExhausted
}

func mutationJitter(ctx context.Context, attempt int) error {
	maximum := 5 * time.Millisecond * time.Duration(1<<attempt)
	delay := time.Duration(rand.Int64N(int64(maximum) + 1))
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
