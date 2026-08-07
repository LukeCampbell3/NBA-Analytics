package main

import (
	"context"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/auth"
	paywallconfig "github.com/jcthi/nba-analytics/paywall/config"
	"github.com/jcthi/nba-analytics/paywall/content"
	"github.com/jcthi/nba-analytics/paywall/gateway"
	"github.com/jcthi/nba-analytics/paywall/ledger"
	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/payment"
	"github.com/jcthi/nba-analytics/paywall/storage"
	"github.com/jcthi/nba-analytics/paywall/transport"
)

var (
	initializeOnce sync.Once
	initializedApp *gateway.App
	initializedCfg paywallconfig.Gateway
	initializeErr  error
)

type discordAccountResolver interface {
	ResolveDiscord(context.Context, string, string) (account.Account, error)
}

type reconcilingResolver struct {
	base       discordAccountResolver
	reconciler *payment.Reconciler
}

func (resolver reconcilingResolver) ResolveDiscord(ctx context.Context, discordID, displayName string) (account.Account, error) {
	value, err := resolver.base.ResolveDiscord(ctx, discordID, displayName)
	if err != nil {
		return account.Account{}, err
	}
	return resolver.reconciler.Reconcile(ctx, value)
}

func Main(ctx context.Context, event transport.RawEvent) transport.Response {
	if event.HTTP.Path == "/health/live" {
		return transport.JSON(http.StatusOK, map[string]string{"status": "live"})
	}
	initializeOnce.Do(func() {
		initializedCfg, initializeErr = paywallconfig.LoadGateway(os.LookupEnv)
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
		contentStore, err := storage.NewR2ContentStore(ctx, initializedCfg.ContentR2)
		if err != nil {
			initializeErr = err
			return
		}
		accountService, err := ledger.NewService(accountStore, initializedCfg.IndexKeyCurrent, initializedCfg.IndexKeyPrevious)
		if err != nil {
			initializeErr = err
			return
		}
		oauthStates, err := auth.NewOAuthStateService(objects, initializedCfg.AllowedRedirects, 10*time.Minute)
		if err != nil {
			initializeErr = err
			return
		}
		discord, err := auth.NewDiscordClient(initializedCfg.Discord, nil)
		if err != nil {
			initializeErr = err
			return
		}
		stripeProvider, err := payment.NewStripeCheckoutProvider(payment.StripeConfig{
			SecretKey:  initializedCfg.PaymentSecretKey,
			PriceID:    initializedCfg.PaymentPriceID,
			Plan:       initializedCfg.PaymentPlan,
			SuccessURL: initializedCfg.PublicOrigin + "/payment/return",
			CancelURL:  initializedCfg.PublicOrigin + "/pricing",
			PIIKeys:    initializedCfg.PIIKeys,
		})
		if err != nil {
			initializeErr = err
			return
		}
		checkoutService, err := payment.NewCheckoutService(stripeProvider, accountStore, time.Minute)
		if err != nil {
			initializeErr = err
			return
		}
		reconciler, err := payment.NewReconciler(
			stripeProvider, accountStore, initializedCfg.PIICurrentKeyID, initializedCfg.PIIKeys, 24*time.Hour,
		)
		if err != nil {
			initializeErr = err
			return
		}
		authorizer, err := auth.NewReconcilingAuthorizer(
			initializedCfg.SessionKeys, accountStore,
			reconciler,
			time.Duration(initializedCfg.IdentityLifetime)*time.Second,
			time.Duration(initializedCfg.AuthzLifetime)*time.Second,
		)
		if err != nil {
			initializeErr = err
			return
		}
		contentGateway, err := content.NewGateway(contentStore, authorizer, time.Minute)
		if err != nil {
			initializeErr = err
			return
		}
		initializedApp, initializeErr = gateway.New(
			initializedCfg.PublicOrigin,
			initializedCfg.PublicPathPrefix,
			initializedCfg.AllowedRedirects,
			oauthStates,
			discord,
			accountStore,
			reconcilingResolver{base: accountService, reconciler: reconciler},
			checkoutService,
			stripeProvider,
			contentGateway,
			auditStore,
			initializedCfg.SessionKeys,
			initializedCfg.CSRFKey,
			time.Duration(initializedCfg.IdentityLifetime)*time.Second,
			time.Duration(initializedCfg.AuthzLifetime)*time.Second,
		)
	})
	if initializeErr != nil || initializedApp == nil {
		return transport.Error(http.StatusServiceUnavailable, "service_unavailable")
	}
	request, err := event.Request(initializedCfg.PublicOrigin, 64*1024)
	if err != nil {
		return transport.Error(http.StatusBadRequest, "invalid_request")
	}
	return initializedApp.Handle(ctx, request)
}
