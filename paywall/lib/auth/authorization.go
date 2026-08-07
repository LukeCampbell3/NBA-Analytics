package auth

import (
	"context"
	"errors"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/security"
)

var ErrAccessDenied = errors.New("access denied")

type AccountReader interface {
	GetAccount(context.Context, string) (account.Account, string, error)
}

type AccountReconciler interface {
	Reconcile(context.Context, account.Account) (account.Account, error)
}

type Authorization struct {
	AccountID      string
	Plan           string
	Claims         security.SessionClaims
	RefreshedToken string
}

type Authorizer struct {
	keys             *security.SessionKeyRing
	accounts         AccountReader
	reconciler       AccountReconciler
	now              func() time.Time
	identityLifetime time.Duration
	authzLifetime    time.Duration
}

func NewAuthorizer(
	keys *security.SessionKeyRing,
	accounts AccountReader,
	identityLifetime time.Duration,
	authzLifetime time.Duration,
) (*Authorizer, error) {
	return newAuthorizer(keys, accounts, nil, identityLifetime, authzLifetime)
}

func NewReconcilingAuthorizer(
	keys *security.SessionKeyRing,
	accounts AccountReader,
	reconciler AccountReconciler,
	identityLifetime time.Duration,
	authzLifetime time.Duration,
) (*Authorizer, error) {
	if reconciler == nil {
		return nil, ErrAccessDenied
	}
	return newAuthorizer(keys, accounts, reconciler, identityLifetime, authzLifetime)
}

func newAuthorizer(
	keys *security.SessionKeyRing,
	accounts AccountReader,
	reconciler AccountReconciler,
	identityLifetime time.Duration,
	authzLifetime time.Duration,
) (*Authorizer, error) {
	if keys == nil || accounts == nil || identityLifetime <= 0 || authzLifetime <= 0 || authzLifetime > identityLifetime {
		return nil, ErrAccessDenied
	}
	return &Authorizer{
		keys: keys, accounts: accounts, reconciler: reconciler, now: time.Now,
		identityLifetime: identityLifetime, authzLifetime: authzLifetime,
	}, nil
}

func (a *Authorizer) Authorize(ctx context.Context, token string, fresh bool) (Authorization, error) {
	now := a.now().UTC()
	claims, err := a.keys.Verify(token, now)
	if err != nil {
		return Authorization{}, ErrAccessDenied
	}
	if !fresh && claims.AuthorizationLeaseValid(now) {
		return Authorization{AccountID: claims.Subject, Plan: claims.Plan, Claims: claims}, nil
	}
	value, _, err := a.accounts.GetAccount(ctx, claims.Subject)
	if err != nil || value.SessionEpoch != claims.SessionEpoch {
		return Authorization{}, ErrAccessDenied
	}
	if a.reconciler != nil {
		value, err = a.reconciler.Reconcile(ctx, value)
	}
	if err != nil || value.SessionEpoch != claims.SessionEpoch || !value.HasAccess(now) || value.Entitlement.Plan == "" {
		return Authorization{}, ErrAccessDenied
	}
	refreshed, refreshedClaims, err := a.keys.RefreshAuthorization(
		claims,
		now,
		a.authzLifetime,
		value.Entitlement.Plan,
		value.Entitlement.ValidUntil,
	)
	if err != nil {
		return Authorization{}, ErrAccessDenied
	}
	return Authorization{
		AccountID: value.AccountID, Plan: value.Entitlement.Plan,
		Claims: refreshedClaims, RefreshedToken: refreshed,
	}, nil
}
