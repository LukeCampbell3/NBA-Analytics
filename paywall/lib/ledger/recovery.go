package ledger

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

var ErrCanonicalAccountExists = errors.New("canonical account already exists")

type RecoveryStore interface {
	MutationStore
	CreateAccount(context.Context, account.Account) error
	GetHistory(context.Context, string, uint64) (account.HistoryRecord, error)
}

// RecoverAccount restores a deleted canonical object from an explicitly chosen
// history revision. Requiring the revision avoids bucket listing and makes the
// operator acknowledge which snapshot is authoritative.
func RecoverAccount(
	ctx context.Context,
	store RecoveryStore,
	accountID string,
	sourceRevision uint64,
	mutationID string,
	now time.Time,
) (account.Account, error) {
	if store == nil || accountID == "" || sourceRevision == 0 || sourceRevision == math.MaxUint64 || mutationID == "" || now.IsZero() {
		return account.Account{}, fmt.Errorf("invalid account recovery")
	}
	if _, _, err := store.GetAccount(ctx, accountID); err == nil {
		return account.Account{}, ErrCanonicalAccountExists
	} else if !errors.Is(err, storage.ErrNotFound) {
		return account.Account{}, err
	}
	source, err := store.GetHistory(ctx, accountID, sourceRevision)
	if err != nil {
		return account.Account{}, err
	}
	recovered := source.Account
	recovered.Revision++
	recovered.SessionEpoch++
	recovered.UpdatedAt = now.UTC()
	history := account.HistoryRecord{
		SchemaVersion: account.SchemaVersion,
		MutationID:    mutationID,
		RecordedAt:    now.UTC(),
		Account:       recovered,
	}
	if err := store.CreateHistory(ctx, history); err != nil {
		if !errors.Is(err, storage.ErrConflict) {
			return account.Account{}, err
		}
		existing, getErr := store.GetHistory(ctx, accountID, recovered.Revision)
		if getErr != nil || existing.MutationID != mutationID || !reflect.DeepEqual(existing.Account, recovered) {
			return account.Account{}, fmt.Errorf("a newer or different history revision prevents recovery")
		}
	}
	if err := store.CreateAccount(ctx, recovered); err != nil {
		if !errors.Is(err, storage.ErrConflict) {
			return account.Account{}, err
		}
		existing, _, getErr := store.GetAccount(ctx, accountID)
		if getErr != nil || !reflect.DeepEqual(existing, recovered) {
			return account.Account{}, ErrCanonicalAccountExists
		}
	}
	return recovered, nil
}

func SuspendAccount(ctx context.Context, store MutationStore, accountID, mutationID string) error {
	return MutateAccount(ctx, store, accountID, mutationID, func(value *account.Account) error {
		value.Status = account.StatusSuspended
		value.SessionEpoch++
		return nil
	})
}
