package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
)

var accountIDPattern = regexp.MustCompile(`^acc_[a-z2-7]{26}$`)

type AccountStore struct {
	objects ObjectStore
	now     func() time.Time
}

func NewAccountStore(objects ObjectStore) *AccountStore {
	return &AccountStore{objects: objects, now: time.Now}
}

func accountKey(accountID string) (string, error) {
	if !accountIDPattern.MatchString(accountID) {
		return "", fmt.Errorf("invalid account id")
	}
	return "accounts/" + accountID + ".json", nil
}

func IdentityIndexKey(version int, identityHMAC string) (string, error) {
	if version < 1 || !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(identityHMAC) {
		return "", fmt.Errorf("invalid identity index")
	}
	return fmt.Sprintf("indexes/discord/v%d/%s.json", version, identityHMAC), nil
}

func (s *AccountStore) GetAccount(ctx context.Context, accountID string) (account.Account, string, error) {
	key, err := accountKey(accountID)
	if err != nil {
		return account.Account{}, "", err
	}
	object, err := s.objects.Get(ctx, key)
	if err != nil {
		return account.Account{}, "", err
	}
	var value account.Account
	if err := strictJSON(object.Body, &value); err != nil {
		return account.Account{}, "", fmt.Errorf("%w: account: %v", ErrMalformed, err)
	}
	if err := validateAccount(value, accountID); err != nil {
		return account.Account{}, "", fmt.Errorf("%w: account: %v", ErrMalformed, err)
	}
	return value, object.ETag, nil
}

func (s *AccountStore) CreateAccount(ctx context.Context, value account.Account) error {
	key, err := accountKey(value.AccountID)
	if err != nil {
		return err
	}
	if err := validateAccount(value, value.AccountID); err != nil {
		return err
	}
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = s.objects.Put(ctx, key, body, PutCondition{IfNoneMatch: true})
	return err
}

func (s *AccountStore) UpdateAccountIfMatch(ctx context.Context, value account.Account, etag string) error {
	if etag == "" {
		return fmt.Errorf("etag is required")
	}
	key, err := accountKey(value.AccountID)
	if err != nil {
		return err
	}
	if err := validateAccount(value, value.AccountID); err != nil {
		return err
	}
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = s.objects.Put(ctx, key, body, PutCondition{IfMatch: etag})
	return err
}

func (s *AccountStore) GetIdentityIndex(ctx context.Context, version int, digest string) (account.IdentityIndex, error) {
	key, err := IdentityIndexKey(version, digest)
	if err != nil {
		return account.IdentityIndex{}, err
	}
	object, err := s.objects.Get(ctx, key)
	if err != nil {
		return account.IdentityIndex{}, err
	}
	var value account.IdentityIndex
	if err := strictJSON(object.Body, &value); err != nil {
		return account.IdentityIndex{}, fmt.Errorf("%w: identity index: %v", ErrMalformed, err)
	}
	if value.SchemaVersion != account.SchemaVersion || !accountIDPattern.MatchString(value.AccountID) || value.CreatedAt.IsZero() {
		return account.IdentityIndex{}, fmt.Errorf("%w: invalid identity index", ErrMalformed)
	}
	return value, nil
}

func (s *AccountStore) CreateIdentityIndex(ctx context.Context, version int, digest string, value account.IdentityIndex) error {
	key, err := IdentityIndexKey(version, digest)
	if err != nil {
		return err
	}
	if value.SchemaVersion != account.SchemaVersion || !accountIDPattern.MatchString(value.AccountID) || value.CreatedAt.IsZero() {
		return fmt.Errorf("invalid identity index")
	}
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = s.objects.Put(ctx, key, body, PutCondition{IfNoneMatch: true})
	return err
}

func (s *AccountStore) CreateHistory(ctx context.Context, value account.HistoryRecord) error {
	if !accountIDPattern.MatchString(value.Account.AccountID) || value.Account.Revision == 0 || value.MutationID == "" {
		return fmt.Errorf("invalid history record")
	}
	key := fmt.Sprintf("account-history/%s/%020d.json", value.Account.AccountID, value.Account.Revision)
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = s.objects.Put(ctx, key, body, PutCondition{IfNoneMatch: true})
	return err
}

func (s *AccountStore) GetHistory(ctx context.Context, accountID string, revision uint64) (account.HistoryRecord, error) {
	if !accountIDPattern.MatchString(accountID) || revision == 0 {
		return account.HistoryRecord{}, fmt.Errorf("invalid history key")
	}
	key := fmt.Sprintf("account-history/%s/%020d.json", accountID, revision)
	object, err := s.objects.Get(ctx, key)
	if err != nil {
		return account.HistoryRecord{}, err
	}
	var value account.HistoryRecord
	if err := strictJSON(object.Body, &value); err != nil {
		return account.HistoryRecord{}, fmt.Errorf("%w: history: %v", ErrMalformed, err)
	}
	if value.Account.AccountID != accountID || value.Account.Revision != revision || value.MutationID == "" {
		return account.HistoryRecord{}, fmt.Errorf("%w: invalid history record", ErrMalformed)
	}
	return value, nil
}

func strictJSON(body []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("trailing JSON data")
	}
	return nil
}

func validateAccount(value account.Account, expectedID string) error {
	if value.SchemaVersion != account.SchemaVersion {
		return fmt.Errorf("unsupported schema version")
	}
	if value.AccountID != expectedID || !accountIDPattern.MatchString(value.AccountID) {
		return fmt.Errorf("account id mismatch")
	}
	if !account.IsKnownStatus(value.Status) {
		return fmt.Errorf("unknown account status")
	}
	if value.Revision == 0 || value.CreatedAt.IsZero() || value.UpdatedAt.IsZero() {
		return fmt.Errorf("missing required account metadata")
	}
	return nil
}
