package account

import (
	"testing"
	"time"
)

func TestHasAccessIsExplicitAndTimeBound(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	for _, test := range []struct {
		status Status
		until  time.Time
		want   bool
	}{
		{StatusActive, now.Add(time.Minute), true},
		{StatusGrace, now.Add(time.Minute), true},
		{StatusActive, now, false},
		{StatusPending, now.Add(time.Hour), false},
		{StatusPastDue, now.Add(time.Hour), false},
		{StatusSuspended, now.Add(time.Hour), false},
		{StatusCanceled, now.Add(time.Hour), false},
		{StatusBanned, now.Add(time.Hour), false},
		{Status("future_status"), now.Add(time.Hour), false},
	} {
		value := Account{Status: test.status, Entitlement: Entitlement{ValidUntil: test.until}}
		if got := value.HasAccess(now); got != test.want {
			t.Fatalf("status %q HasAccess() = %v, want %v", test.status, got, test.want)
		}
	}
}
