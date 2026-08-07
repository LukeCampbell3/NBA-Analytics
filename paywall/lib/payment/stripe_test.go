package payment

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"

	stripe "github.com/stripe/stripe-go/v85"
)

type fakeStripeAPI struct {
	checkoutParams *stripe.CheckoutSessionCreateParams
	portalParams   *stripe.BillingPortalSessionCreateParams
	subscription   *stripe.Subscription
}

func (api *fakeStripeAPI) CreateCheckout(_ context.Context, params *stripe.CheckoutSessionCreateParams) (*stripe.CheckoutSession, error) {
	api.checkoutParams = params
	return &stripe.CheckoutSession{ID: "cs_123", URL: "https://checkout.stripe.example/cs_123"}, nil
}

func (api *fakeStripeAPI) CreateBillingPortal(_ context.Context, params *stripe.BillingPortalSessionCreateParams) (*stripe.BillingPortalSession, error) {
	api.portalParams = params
	return &stripe.BillingPortalSession{Customer: *params.Customer, URL: "https://billing.stripe.example/session"}, nil
}

func (api *fakeStripeAPI) RetrieveSubscription(context.Context, string) (*stripe.Subscription, error) {
	return api.subscription, nil
}

func newTestStripeProvider(t *testing.T, api stripeAPI, now time.Time) *StripeProvider {
	t.Helper()
	provider, err := NewStripeProvider(StripeConfig{
		SecretKey: "sk_test", WebhookSecret: "whsec_test", PriceID: "price_123", Plan: "individual",
		SuccessURL: "https://example.com/payment/return", CancelURL: "https://example.com/pricing",
		PIIKeys: map[string][]byte{"pii-current": []byte("0123456789abcdef0123456789abcdef")},
	})
	if err != nil {
		t.Fatal(err)
	}
	provider.api = api
	provider.now = func() time.Time { return now }
	provider.verifier.now = func() time.Time { return now }
	return provider
}

func TestStripeCheckoutCarriesReconciliationMetadataAndIdempotency(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	api := &fakeStripeAPI{}
	provider := newTestStripeProvider(t, api, now)
	value := account.Account{AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa"}
	checkout, err := provider.CreateCheckout(context.Background(), value, "checkout-key")
	if err != nil {
		t.Fatal(err)
	}
	params := api.checkoutParams
	if checkout.ID != "cs_123" || params == nil || params.ClientReferenceID == nil || *params.ClientReferenceID != value.AccountID ||
		params.Metadata["account_id"] != value.AccountID || params.SubscriptionData.Metadata["account_id"] != value.AccountID ||
		params.IdempotencyKey == nil || *params.IdempotencyKey != "checkout-key" || len(params.LineItems) != 1 ||
		params.LineItems[0].Price == nil || *params.LineItems[0].Price != "price_123" {
		t.Fatalf("checkout params = %#v", params)
	}
}

func TestStripeBillingPortalUsesEncryptedCustomerReference(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	api := &fakeStripeAPI{}
	provider := newTestStripeProvider(t, api, now)
	value := account.Account{AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa"}
	field, err := security.EncryptField(
		"pii-current", []byte("0123456789abcdef0123456789abcdef"), value.AccountID, "cus_123",
	)
	if err != nil {
		t.Fatal(err)
	}
	value.Payment.CustomerID = field
	portalURL, err := provider.CreateBillingPortal(context.Background(), value, "https://example.com/app/")
	if err != nil {
		t.Fatal(err)
	}
	if portalURL != "https://billing.stripe.example/session" || api.portalParams == nil ||
		api.portalParams.Customer == nil || *api.portalParams.Customer != "cus_123" ||
		api.portalParams.ReturnURL == nil || *api.portalParams.ReturnURL != "https://example.com/app/" {
		t.Fatalf("billing portal params = %#v, url = %q", api.portalParams, portalURL)
	}
}

func TestStripeWebhookAcceptsOnlySignedSubscriptionEvents(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	provider := newTestStripeProvider(t, &fakeStripeAPI{}, now)
	eventBody := func(eventType string) []byte {
		body, _ := json.Marshal(map[string]any{
			"id": "evt_123", "object": "event", "type": eventType, "created": now.Unix(),
			"data": map[string]any{"object": map[string]any{
				"id": "sub_123", "object": "subscription",
				"metadata": map[string]string{"account_id": "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", "schema_version": "1"},
			}},
		})
		return body
	}
	sign := func(body []byte) string {
		mac := hmac.New(sha256.New, []byte("whsec_test"))
		fmt.Fprintf(mac, "%d.", now.Unix())
		mac.Write(body)
		return fmt.Sprintf("t=%d,v1=%s", now.Unix(), hex.EncodeToString(mac.Sum(nil)))
	}
	body := eventBody("customer.subscription.updated")
	event, err := provider.VerifyWebhook(body, sign(body))
	if err != nil || event.AccountID == "" || event.SubscriptionID != "sub_123" {
		t.Fatalf("verified event = %#v, %v", event, err)
	}
	unsupported := eventBody("invoice.paid")
	if _, err := provider.VerifyWebhook(unsupported, sign(unsupported)); err != ErrUnsupportedEvent {
		t.Fatalf("unsupported event error = %v", err)
	}
	if _, err := provider.VerifyWebhook(body, sign([]byte("different"))); err == nil {
		t.Fatal("modified webhook body was accepted")
	}
}

func TestStripeAuthoritativeSnapshotFailsClosedForWrongPrice(t *testing.T) {
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	subscription := &stripe.Subscription{
		ID: "sub_123", Status: stripe.SubscriptionStatus("active"), StartDate: now.Add(-24 * time.Hour).Unix(),
		Customer: &stripe.Customer{ID: "cus_123"},
		Metadata: map[string]string{"account_id": "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", "schema_version": "1"},
		Items: &stripe.SubscriptionItemList{Data: []*stripe.SubscriptionItem{{
			Price: &stripe.Price{ID: "price_wrong"}, Quantity: 1,
			CurrentPeriodStart: now.Add(-24 * time.Hour).Unix(), CurrentPeriodEnd: now.Add(29 * 24 * time.Hour).Unix(),
		}}},
	}
	provider := newTestStripeProvider(t, &fakeStripeAPI{subscription: subscription}, now)
	snapshot, err := provider.GetAuthoritativeEntitlement(context.Background(), SubscriptionReference{
		AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", SubscriptionID: "sub_123",
	})
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Status != account.StatusSuspended || snapshot.CustomerID != "cus_123" {
		t.Fatalf("snapshot = %#v", snapshot)
	}
}
