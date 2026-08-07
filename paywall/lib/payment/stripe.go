package payment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"

	stripe "github.com/stripe/stripe-go/v85"
)

var ErrUnsupportedEvent = errors.New("unsupported payment event")

type StripeConfig struct {
	SecretKey          string
	WebhookSecret      string
	PriceID            string
	Plan               string
	SuccessURL         string
	CancelURL          string
	PIIKeys            map[string][]byte
	SignatureTolerance time.Duration
}

type stripeAPI interface {
	CreateCheckout(context.Context, *stripe.CheckoutSessionCreateParams) (*stripe.CheckoutSession, error)
	CreateBillingPortal(context.Context, *stripe.BillingPortalSessionCreateParams) (*stripe.BillingPortalSession, error)
	RetrieveSubscription(context.Context, string) (*stripe.Subscription, error)
}

type stripeClientAPI struct {
	client *stripe.Client
}

func (api stripeClientAPI) CreateCheckout(ctx context.Context, params *stripe.CheckoutSessionCreateParams) (*stripe.CheckoutSession, error) {
	return api.client.V1CheckoutSessions.Create(ctx, params)
}

func (api stripeClientAPI) CreateBillingPortal(ctx context.Context, params *stripe.BillingPortalSessionCreateParams) (*stripe.BillingPortalSession, error) {
	return api.client.V1BillingPortalSessions.Create(ctx, params)
}

func (api stripeClientAPI) RetrieveSubscription(ctx context.Context, id string) (*stripe.Subscription, error) {
	return api.client.V1Subscriptions.Retrieve(ctx, id, nil)
}

type StripeProvider struct {
	api      stripeAPI
	config   StripeConfig
	verifier *StripeSignatureVerifier
	now      func() time.Time
}

func NewStripeProvider(config StripeConfig) (*StripeProvider, error) {
	return newStripeProvider(config, true, true)
}

func NewStripeCheckoutProvider(config StripeConfig) (*StripeProvider, error) {
	return newStripeProvider(config, false, true)
}

func NewStripeWebhookProvider(config StripeConfig) (*StripeProvider, error) {
	return newStripeProvider(config, true, false)
}

func newStripeProvider(config StripeConfig, requireWebhook, requireCheckout bool) (*StripeProvider, error) {
	if config.SecretKey == "" || config.PriceID == "" || config.Plan == "" ||
		(requireCheckout && (config.SuccessURL == "" || config.CancelURL == "")) {
		return nil, fmt.Errorf("invalid Stripe provider configuration")
	}
	if config.SignatureTolerance == 0 {
		config.SignatureTolerance = 5 * time.Minute
	}
	var verifier *StripeSignatureVerifier
	if requireWebhook || config.WebhookSecret != "" {
		var err error
		verifier, err = NewStripeSignatureVerifier(config.WebhookSecret, config.SignatureTolerance)
		if err != nil {
			return nil, err
		}
	}
	return &StripeProvider{
		api:    stripeClientAPI{client: stripe.NewClient(config.SecretKey)},
		config: config, verifier: verifier, now: time.Now,
	}, nil
}

func (p *StripeProvider) CreateCheckout(ctx context.Context, value account.Account, idempotencyKey string) (Checkout, error) {
	if idempotencyKey == "" {
		return Checkout{}, fmt.Errorf("idempotency key is required")
	}
	params := &stripe.CheckoutSessionCreateParams{
		Mode:              stripe.String("subscription"),
		ClientReferenceID: stripe.String(value.AccountID),
		SuccessURL:        stripe.String(p.config.SuccessURL),
		CancelURL:         stripe.String(p.config.CancelURL),
		LineItems: []*stripe.CheckoutSessionCreateLineItemParams{{
			Price: stripe.String(p.config.PriceID), Quantity: stripe.Int64(1),
		}},
		SubscriptionData: &stripe.CheckoutSessionCreateSubscriptionDataParams{},
	}
	params.SetIdempotencyKey(idempotencyKey)
	params.AddMetadata("account_id", value.AccountID)
	params.AddMetadata("schema_version", "1")
	params.SubscriptionData.AddMetadata("account_id", value.AccountID)
	params.SubscriptionData.AddMetadata("schema_version", "1")
	if value.Payment.CustomerID != nil {
		customerID, err := security.DecryptField(p.config.PIIKeys, value.AccountID, value.Payment.CustomerID)
		if err != nil {
			return Checkout{}, fmt.Errorf("decrypt Stripe customer reference: %w", err)
		}
		params.Customer = stripe.String(customerID)
	}
	session, err := p.api.CreateCheckout(ctx, params)
	if err != nil {
		return Checkout{}, err
	}
	if session == nil || session.ID == "" || session.URL == "" {
		return Checkout{}, fmt.Errorf("Stripe returned an invalid Checkout Session")
	}
	return Checkout{ID: session.ID, URL: session.URL}, nil
}

func (p *StripeProvider) CreateBillingPortal(ctx context.Context, value account.Account, returnURL string) (string, error) {
	if returnURL == "" {
		return "", fmt.Errorf("billing portal return URL is required")
	}
	customerID, err := security.DecryptField(p.config.PIIKeys, value.AccountID, value.Payment.CustomerID)
	if err != nil {
		return "", fmt.Errorf("decrypt Stripe customer reference: %w", err)
	}
	session, err := p.api.CreateBillingPortal(ctx, &stripe.BillingPortalSessionCreateParams{
		Customer:  stripe.String(customerID),
		ReturnURL: stripe.String(returnURL),
	})
	if err != nil {
		return "", err
	}
	if session == nil || session.URL == "" || session.Customer != customerID {
		return "", fmt.Errorf("Stripe returned an invalid Billing Portal Session")
	}
	return session.URL, nil
}

func (p *StripeProvider) VerifyWebhook(rawBody []byte, signatureHeader string) (ProviderEvent, error) {
	if p.verifier == nil {
		return ProviderEvent{}, fmt.Errorf("Stripe webhook verification is not configured")
	}
	if err := p.verifier.Verify(rawBody, signatureHeader); err != nil {
		return ProviderEvent{}, err
	}
	var event stripe.Event
	if err := json.Unmarshal(rawBody, &event); err != nil || event.ID == "" || event.Data == nil || event.Created <= 0 {
		return ProviderEvent{}, fmt.Errorf("invalid Stripe event")
	}
	providerEvent := ProviderEvent{ID: event.ID, Type: string(event.Type), OccurredAt: time.Unix(event.Created, 0).UTC()}
	switch string(event.Type) {
	case "customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted":
	default:
		return providerEvent, ErrUnsupportedEvent
	}
	var subscription stripe.Subscription
	if err := json.Unmarshal(event.Data.Raw, &subscription); err != nil || subscription.ID == "" {
		return ProviderEvent{}, fmt.Errorf("invalid Stripe subscription event")
	}
	accountID := subscription.Metadata["account_id"]
	if accountID == "" || subscription.Metadata["schema_version"] != "1" {
		return ProviderEvent{}, fmt.Errorf("Stripe subscription metadata is invalid")
	}
	providerEvent.AccountID = accountID
	providerEvent.SubscriptionID = subscription.ID
	return providerEvent, nil
}

func (p *StripeProvider) GetAuthoritativeEntitlement(ctx context.Context, reference SubscriptionReference) (EntitlementSnapshot, error) {
	if reference.SubscriptionID == "" {
		return EntitlementSnapshot{}, fmt.Errorf("Stripe subscription reference is missing")
	}
	subscription, err := p.api.RetrieveSubscription(ctx, reference.SubscriptionID)
	if err != nil {
		return EntitlementSnapshot{}, err
	}
	if subscription == nil || subscription.ID != reference.SubscriptionID || subscription.Customer == nil ||
		subscription.Metadata["account_id"] != reference.AccountID || subscription.Metadata["schema_version"] != "1" {
		return EntitlementSnapshot{}, fmt.Errorf("Stripe subscription identity mismatch")
	}
	now := p.now().UTC()
	status := mapStripeStatus(subscription.Status)
	validFrom, validUntil, correctPrice := stripeSubscriptionPeriod(subscription, p.config.PriceID)
	if !correctPrice {
		status = account.StatusSuspended
	}
	if validUntil.IsZero() {
		validUntil = now
	}
	if validFrom.IsZero() {
		validFrom = time.Unix(subscription.StartDate, 0).UTC()
	}
	return EntitlementSnapshot{
		Status: status, Plan: p.config.Plan, Source: "stripe",
		ValidFrom: validFrom, ValidUntil: validUntil,
		ProviderVerifiedAt: now, ProviderUpdatedAt: now,
		CustomerID: subscription.Customer.ID, SubscriptionID: subscription.ID,
	}, nil
}

func mapStripeStatus(status stripe.SubscriptionStatus) account.Status {
	switch string(status) {
	case "active":
		return account.StatusActive
	case "past_due":
		return account.StatusPastDue
	case "canceled", "incomplete_expired", "unpaid":
		return account.StatusCanceled
	case "paused":
		return account.StatusSuspended
	case "trialing", "incomplete":
		return account.StatusPending
	default:
		return account.StatusSuspended
	}
}

func stripeSubscriptionPeriod(subscription *stripe.Subscription, expectedPrice string) (time.Time, time.Time, bool) {
	if subscription.Items == nil {
		return time.Time{}, time.Time{}, false
	}
	var start, end time.Time
	found := false
	for _, item := range subscription.Items.Data {
		if item == nil || item.Price == nil || item.Price.ID != expectedPrice || item.Quantity != 1 {
			continue
		}
		itemStart := time.Unix(item.CurrentPeriodStart, 0).UTC()
		itemEnd := time.Unix(item.CurrentPeriodEnd, 0).UTC()
		if item.CurrentPeriodStart <= 0 || item.CurrentPeriodEnd <= item.CurrentPeriodStart {
			continue
		}
		if start.IsZero() || itemStart.Before(start) {
			start = itemStart
		}
		if itemEnd.After(end) {
			end = itemEnd
		}
		found = true
	}
	return start, end, found
}
