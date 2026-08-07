package storage

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestPaymentEventLeaseRecoversAndAppliedEventIsIdempotent(t *testing.T) {
	ctx := context.Background()
	objects := NewMemoryStore()
	store, err := NewEventStore(objects, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return now }
	digest := strings.Repeat("a", 64)
	first, err := store.Claim(ctx, "evt_123", "invoice.paid", "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", digest, "owner-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Claim(ctx, "evt_123", "invoice.paid", "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", digest, "owner-2"); !errors.Is(err, ErrEventBusy) {
		t.Fatalf("live lease claim error = %v, want ErrEventBusy", err)
	}
	now = now.Add(2 * time.Minute)
	recovered, err := store.Claim(ctx, "evt_123", "invoice.paid", "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", digest, "owner-2")
	if err != nil {
		t.Fatal(err)
	}
	if recovered.Event.Attempts != 2 || recovered.Event.LeaseOwner != "owner-2" {
		t.Fatalf("recovered event = %#v", recovered.Event)
	}
	if err := store.MarkApplied(ctx, recovered); err != nil {
		t.Fatal(err)
	}
	applied, err := store.Claim(ctx, "evt_123", "invoice.paid", "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", digest, "owner-3")
	if err != nil {
		t.Fatal(err)
	}
	if applied.Event.Status != "applied" || applied.Event.AppliedAt == nil {
		t.Fatalf("applied event = %#v", applied.Event)
	}
	if err := store.MarkApplied(ctx, first); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale lease finalization error = %v, want ErrConflict", err)
	}
}
