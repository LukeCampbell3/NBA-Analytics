package observability

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/storage"
)

func TestAuditStoreWritesImmutableDatePartitionedEvent(t *testing.T) {
	ctx := context.Background()
	objects := storage.NewMemoryStore()
	store, err := NewAuditStore(objects)
	if err != nil {
		t.Fatal(err)
	}
	event := AuditEvent{
		ID: "0123456789abcdef0123456789abcdef", Type: "account.suspended",
		AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", Outcome: "success",
		OccurredAt: time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC),
	}
	if err := store.Record(ctx, event); err != nil {
		t.Fatal(err)
	}
	key := "audit/2026/08/06/20260806T210000.000000000Z-0123456789abcdef0123456789abcdef.json"
	stored, err := objects.Get(ctx, key)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AuditEvent
	if err := json.Unmarshal(stored.Body, &decoded); err != nil || decoded.Type != event.Type {
		t.Fatalf("audit body = %s, error = %v", stored.Body, err)
	}
	if err := store.Record(ctx, event); err != nil {
		t.Fatalf("idempotent audit retry: %v", err)
	}
}

func TestAuditStoreRejectsSecretsAndUnstructuredTypes(t *testing.T) {
	store, _ := NewAuditStore(storage.NewMemoryStore())
	event := AuditEvent{Type: "session cookie=secret", Outcome: "success"}
	if err := store.Record(context.Background(), event); err == nil || !strings.Contains(err.Error(), "invalid") {
		t.Fatalf("invalid audit event error = %v", err)
	}
}
