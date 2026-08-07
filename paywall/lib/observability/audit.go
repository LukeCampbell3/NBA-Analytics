package observability

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

const AuditSchemaVersion = 1

var (
	auditIDPattern      = regexp.MustCompile(`^[a-f0-9]{32}$`)
	auditTypePattern    = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,63}$`)
	auditAccountPattern = regexp.MustCompile(`^acc_[a-z2-7]{26}$`)
)

type AuditEvent struct {
	SchemaVersion int       `json:"schema_version"`
	ID            string    `json:"id"`
	Type          string    `json:"type"`
	AccountID     string    `json:"account_id,omitempty"`
	Outcome       string    `json:"outcome"`
	OccurredAt    time.Time `json:"occurred_at"`
}

type Auditor interface {
	Record(context.Context, AuditEvent) error
}

type AuditStore struct {
	objects storage.ObjectStore
	now     func() time.Time
}

func NewAuditStore(objects storage.ObjectStore) (*AuditStore, error) {
	if objects == nil {
		return nil, fmt.Errorf("audit object store is required")
	}
	return &AuditStore{objects: objects, now: time.Now}, nil
}

func (store *AuditStore) Record(ctx context.Context, event AuditEvent) error {
	if event.ID == "" {
		random := make([]byte, 16)
		if _, err := rand.Read(random); err != nil {
			return err
		}
		event.ID = hex.EncodeToString(random)
	}
	if event.SchemaVersion == 0 {
		event.SchemaVersion = AuditSchemaVersion
	}
	if event.OccurredAt.IsZero() {
		event.OccurredAt = store.now().UTC()
	} else {
		event.OccurredAt = event.OccurredAt.UTC()
	}
	if event.SchemaVersion != AuditSchemaVersion || !auditIDPattern.MatchString(event.ID) ||
		!auditTypePattern.MatchString(event.Type) || (event.AccountID != "" && !auditAccountPattern.MatchString(event.AccountID)) ||
		(event.Outcome != "success" && event.Outcome != "denied" && event.Outcome != "error") {
		return fmt.Errorf("invalid audit event")
	}
	body, err := json.Marshal(event)
	if err != nil {
		return err
	}
	key := fmt.Sprintf("audit/%s/%s-%s.json",
		event.OccurredAt.Format("2006/01/02"), event.OccurredAt.Format("20060102T150405.000000000Z"), event.ID)
	_, err = store.objects.Put(ctx, key, body, storage.PutCondition{IfNoneMatch: true})
	if errors.Is(err, storage.ErrConflict) {
		existing, getErr := store.objects.Get(ctx, key)
		if getErr == nil && bytes.Equal(existing.Body, body) {
			return nil
		}
		return fmt.Errorf("audit event id collision")
	}
	return err
}

func AccountEvent(eventType string, value account.Account, outcome string, occurredAt time.Time) AuditEvent {
	return AuditEvent{Type: eventType, AccountID: value.AccountID, Outcome: outcome, OccurredAt: occurredAt}
}
