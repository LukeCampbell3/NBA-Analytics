package auth

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/storage"
)

func TestOAuthStateCanBeConsumedOnlyOnce(t *testing.T) {
	ctx := context.Background()
	service, err := NewOAuthStateService(storage.NewMemoryStore(), []string{"/app/"}, 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	state, err := service.Create(ctx, "/app/")
	if err != nil {
		t.Fatal(err)
	}
	var successes atomic.Int32
	var wait sync.WaitGroup
	start := make(chan struct{})
	for attempt := 0; attempt < 2; attempt++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			if _, consumeErr := service.Consume(ctx, state); consumeErr == nil {
				successes.Add(1)
			} else if !errors.Is(consumeErr, ErrInvalidOAuthState) {
				t.Errorf("unexpected consume error: %v", consumeErr)
			}
		}()
	}
	close(start)
	wait.Wait()
	if got := successes.Load(); got != 1 {
		t.Fatalf("successful consumes = %d, want 1", got)
	}
}

func TestOAuthRedirectMustBeAllowlisted(t *testing.T) {
	service, _ := NewOAuthStateService(storage.NewMemoryStore(), []string{"/app/"}, 10*time.Minute)
	if _, err := service.Create(context.Background(), "https://evil.example/"); !errors.Is(err, ErrInvalidOAuthState) {
		t.Fatalf("unlisted redirect error = %v", err)
	}
}
