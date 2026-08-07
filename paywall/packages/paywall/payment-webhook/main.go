package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"net/http"
	"os"
	"sync"
	"time"

	paywallconfig "github.com/jcthi/nba-analytics/paywall/config"
	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/payment"
	"github.com/jcthi/nba-analytics/paywall/paymentwebhook"
	"github.com/jcthi/nba-analytics/paywall/storage"
	"github.com/jcthi/nba-analytics/paywall/transport"
)

var (
	initializeOnce sync.Once
	initializedApp *paymentwebhook.App
	initializedCfg paywallconfig.Webhook
	initializeErr  error
)

func Main(ctx context.Context, event transport.RawEvent) transport.Response {
	initializeOnce.Do(func() {
		initializedCfg, initializeErr = paywallconfig.LoadWebhook(os.LookupEnv)
		if initializeErr != nil {
			return
		}
		objects, err := storage.NewR2ObjectStore(ctx, initializedCfg.StateR2)
		if err != nil {
			initializeErr = err
			return
		}
		accountStore := storage.NewAccountStore(objects)
		auditStore, err := observability.NewAuditStore(objects)
		if err != nil {
			initializeErr = err
			return
		}
		eventStore, err := storage.NewEventStore(objects, time.Minute)
		if err != nil {
			initializeErr = err
			return
		}
		provider, err := payment.NewStripeWebhookProvider(payment.StripeConfig{
			SecretKey:     initializedCfg.PaymentSecretKey,
			WebhookSecret: initializedCfg.SigningSecret,
			PriceID:       initializedCfg.PaymentPriceID,
			Plan:          initializedCfg.PaymentPlan,
		})
		if err != nil {
			initializeErr = err
			return
		}
		processor, err := payment.NewAuditedWebhookProcessor(
			provider, eventStore, accountStore, initializedCfg.PIICurrentKeyID, initializedCfg.PIICurrentKey, auditStore,
		)
		if err != nil {
			initializeErr = err
			return
		}
		initializedApp, initializeErr = paymentwebhook.NewAudited(provider, processor, initializedCfg.MaximumBody, auditStore)
	})
	if initializeErr != nil || initializedApp == nil {
		return transport.Error(http.StatusServiceUnavailable, "service_unavailable")
	}
	request, err := event.Request("https://webhook.invalid", initializedCfg.MaximumBody)
	if err != nil {
		return transport.Error(http.StatusBadRequest, "invalid_request")
	}
	return initializedApp.Handle(ctx, request, leaseOwner(event.HTTP.Headers["x-request-id"]))
}

func leaseOwner(requestID string) string {
	if requestID != "" && len(requestID) <= 128 {
		return requestID
	}
	random := make([]byte, 18)
	if _, err := rand.Read(random); err != nil {
		return "request-fallback"
	}
	return base64.RawURLEncoding.EncodeToString(random)
}
