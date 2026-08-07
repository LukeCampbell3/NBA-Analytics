package storage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"
)

var ErrEventBusy = errors.New("payment event is being processed")

type PaymentEvent struct {
	SchemaVersion  int        `json:"schema_version"`
	EventID        string     `json:"event_id"`
	EventType      string     `json:"event_type"`
	AccountID      string     `json:"account_id"`
	Status         string     `json:"status"`
	PayloadSHA256  string     `json:"payload_sha256"`
	LeaseOwner     string     `json:"lease_owner"`
	LeaseExpiresAt time.Time  `json:"lease_expires_at"`
	Attempts       uint64     `json:"attempts"`
	ReceivedAt     time.Time  `json:"received_at"`
	AppliedAt      *time.Time `json:"applied_at"`
}

type ClaimedPaymentEvent struct {
	Event PaymentEvent
	ETag  string
}

type EventStore struct {
	objects       ObjectStore
	now           func() time.Time
	leaseDuration time.Duration
}

func NewEventStore(objects ObjectStore, leaseDuration time.Duration) (*EventStore, error) {
	if leaseDuration < 10*time.Second || leaseDuration > 5*time.Minute {
		return nil, fmt.Errorf("event lease duration must be between 10 seconds and 5 minutes")
	}
	return &EventStore{objects: objects, now: time.Now, leaseDuration: leaseDuration}, nil
}

func (s *EventStore) Claim(
	ctx context.Context,
	eventID, eventType, accountID, payloadSHA256, owner string,
) (ClaimedPaymentEvent, error) {
	key, err := paymentEventKey(eventID)
	if err != nil || eventType == "" || !accountIDPattern.MatchString(accountID) ||
		!regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(payloadSHA256) || owner == "" {
		return ClaimedPaymentEvent{}, fmt.Errorf("invalid payment event claim")
	}
	now := s.now().UTC()
	initial := PaymentEvent{
		SchemaVersion:  1,
		EventID:        eventID,
		EventType:      eventType,
		AccountID:      accountID,
		Status:         "processing",
		PayloadSHA256:  payloadSHA256,
		LeaseOwner:     owner,
		LeaseExpiresAt: now.Add(s.leaseDuration),
		Attempts:       1,
		ReceivedAt:     now,
	}
	body, _ := json.Marshal(initial)
	etag, err := s.objects.Put(ctx, key, body, PutCondition{IfNoneMatch: true})
	if err == nil {
		return ClaimedPaymentEvent{Event: initial, ETag: etag}, nil
	}
	if !errors.Is(err, ErrConflict) {
		return ClaimedPaymentEvent{}, err
	}

	for attempt := 0; attempt < 5; attempt++ {
		object, getErr := s.objects.Get(ctx, key)
		if getErr != nil {
			return ClaimedPaymentEvent{}, getErr
		}
		var current PaymentEvent
		if err := strictJSON(object.Body, &current); err != nil {
			return ClaimedPaymentEvent{}, fmt.Errorf("%w: payment event", ErrMalformed)
		}
		if current.EventID != eventID || current.EventType != eventType || current.AccountID != accountID ||
			current.PayloadSHA256 != payloadSHA256 || current.SchemaVersion != 1 {
			return ClaimedPaymentEvent{}, fmt.Errorf("%w: payment event identity mismatch", ErrMalformed)
		}
		if current.Status == "applied" {
			return ClaimedPaymentEvent{Event: current, ETag: object.ETag}, nil
		}
		if current.Status != "processing" || now.Before(current.LeaseExpiresAt) {
			return ClaimedPaymentEvent{}, ErrEventBusy
		}
		current.LeaseOwner = owner
		current.LeaseExpiresAt = now.Add(s.leaseDuration)
		current.Attempts++
		body, _ = json.Marshal(current)
		etag, err = s.objects.Put(ctx, key, body, PutCondition{IfMatch: object.ETag})
		if err == nil {
			return ClaimedPaymentEvent{Event: current, ETag: etag}, nil
		}
		if !errors.Is(err, ErrConflict) {
			return ClaimedPaymentEvent{}, err
		}
	}
	return ClaimedPaymentEvent{}, ErrConflictRetriesExhausted
}

func (s *EventStore) MarkApplied(ctx context.Context, claim ClaimedPaymentEvent) error {
	if claim.Event.Status == "applied" {
		return nil
	}
	key, err := paymentEventKey(claim.Event.EventID)
	if err != nil || claim.ETag == "" {
		return fmt.Errorf("invalid payment event finalization")
	}
	now := s.now().UTC()
	claim.Event.Status = "applied"
	claim.Event.AppliedAt = &now
	claim.Event.LeaseExpiresAt = time.Time{}
	body, _ := json.Marshal(claim.Event)
	_, err = s.objects.Put(ctx, key, body, PutCondition{IfMatch: claim.ETag})
	return err
}

func paymentEventKey(eventID string) (string, error) {
	if !regexp.MustCompile(`^evt_[A-Za-z0-9_]+$`).MatchString(eventID) || len(eventID) > 255 {
		return "", fmt.Errorf("invalid payment event id")
	}
	return "payment-events/stripe/" + eventID + ".json", nil
}
