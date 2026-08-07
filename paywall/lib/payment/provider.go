package payment

import (
	"context"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
)

type Checkout struct {
	ID  string
	URL string
}

type SubscriptionReference struct {
	AccountID      string
	EventID        string
	SubscriptionID string
}

type ProviderEvent struct {
	ID             string
	Type           string
	AccountID      string
	SubscriptionID string
	OccurredAt     time.Time
}

type EntitlementSnapshot struct {
	Status             account.Status
	Plan               string
	Source             string
	ValidFrom          time.Time
	ValidUntil         time.Time
	ProviderVerifiedAt time.Time
	ProviderUpdatedAt  time.Time
	CustomerID         string
	SubscriptionID     string
}

type Provider interface {
	CreateCheckout(context.Context, account.Account, string) (Checkout, error)
	VerifyWebhook([]byte, string) (ProviderEvent, error)
	GetAuthoritativeEntitlement(context.Context, SubscriptionReference) (EntitlementSnapshot, error)
}
