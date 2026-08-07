package ledger

import (
	"context"
	"crypto/rand"
	"encoding/base32"
	"errors"
	"fmt"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type IdentityAccountStore interface {
	GetIdentityIndex(context.Context, int, string) (account.IdentityIndex, error)
	CreateIdentityIndex(context.Context, int, string, account.IdentityIndex) error
	GetAccount(context.Context, string) (account.Account, string, error)
	CreateAccount(context.Context, account.Account) error
}

type Service struct {
	store            IdentityAccountStore
	currentIndexKey  []byte
	previousIndexKey []byte
	now              func() time.Time
	newAccountID     func() (string, error)
}

func NewService(store IdentityAccountStore, currentIndexKey, previousIndexKey []byte) (*Service, error) {
	if len(currentIndexKey) < 32 {
		return nil, fmt.Errorf("current identity index key must contain at least 32 bytes")
	}
	return &Service{
		store:            store,
		currentIndexKey:  append([]byte(nil), currentIndexKey...),
		previousIndexKey: append([]byte(nil), previousIndexKey...),
		now:              time.Now,
		newAccountID:     randomAccountID,
	}, nil
}

// ResolveDiscord deterministically resolves one Discord identity to one account.
// The index is claimed first; any caller can repair an interrupted account create.
func (s *Service) ResolveDiscord(ctx context.Context, discordUserID, displayName string) (account.Account, error) {
	if discordUserID == "" {
		return account.Account{}, fmt.Errorf("discord user id is required")
	}
	currentDigest := security.IdentityHMAC(s.currentIndexKey, "discord", discordUserID)
	index, err := s.store.GetIdentityIndex(ctx, 1, currentDigest)
	if errors.Is(err, storage.ErrNotFound) && len(s.previousIndexKey) > 0 {
		previousDigest := security.IdentityHMAC(s.previousIndexKey, "discord", discordUserID)
		previous, previousErr := s.store.GetIdentityIndex(ctx, 1, previousDigest)
		if previousErr == nil {
			index = previous
			err = s.store.CreateIdentityIndex(ctx, 1, currentDigest, previous)
			if errors.Is(err, storage.ErrConflict) {
				index, err = s.store.GetIdentityIndex(ctx, 1, currentDigest)
			}
		} else if !errors.Is(previousErr, storage.ErrNotFound) {
			return account.Account{}, previousErr
		}
	}
	if errors.Is(err, storage.ErrNotFound) {
		accountID, idErr := s.newAccountID()
		if idErr != nil {
			return account.Account{}, idErr
		}
		index = account.IdentityIndex{SchemaVersion: account.SchemaVersion, AccountID: accountID, CreatedAt: s.now().UTC()}
		err = s.store.CreateIdentityIndex(ctx, 1, currentDigest, index)
		if errors.Is(err, storage.ErrConflict) {
			index, err = s.store.GetIdentityIndex(ctx, 1, currentDigest)
		}
	}
	if err != nil {
		return account.Account{}, err
	}

	value, _, err := s.store.GetAccount(ctx, index.AccountID)
	if err == nil {
		return value, nil
	}
	if !errors.Is(err, storage.ErrNotFound) {
		return account.Account{}, err
	}

	now := s.now().UTC()
	value = account.Account{
		SchemaVersion: account.SchemaVersion,
		Revision:      1,
		AccountID:     index.AccountID,
		DisplayName:   displayName,
		Status:        account.StatusPending,
		SessionEpoch:  1,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	if err := s.store.CreateAccount(ctx, value); err != nil && !errors.Is(err, storage.ErrConflict) {
		return account.Account{}, err
	}
	value, _, err = s.store.GetAccount(ctx, index.AccountID)
	return value, err
}

func randomAccountID() (string, error) {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	encoded := base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(raw[:])
	return "acc_" + stringLowerASCII(encoded), nil
}

func stringLowerASCII(value string) string {
	buffer := []byte(value)
	for index, char := range buffer {
		if char >= 'A' && char <= 'Z' {
			buffer[index] = char + ('a' - 'A')
		}
	}
	return string(buffer)
}
