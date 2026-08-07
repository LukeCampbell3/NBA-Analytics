package gateway

import (
	"context"
	"crypto/subtle"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/auth"
	"github.com/jcthi/nba-analytics/paywall/ledger"
	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/payment"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
	"github.com/jcthi/nba-analytics/paywall/transport"
)

var errDeletionRequiresCancellation = errors.New("subscription must be canceled before account deletion")

type OAuthStates interface {
	Create(context.Context, string) (string, error)
	Consume(context.Context, string) (auth.OAuthState, error)
}

type Discord interface {
	AuthorizationURL(string) (string, error)
	ResolveIdentity(context.Context, string) (auth.DiscordIdentity, error)
}

type Accounts interface {
	ledger.IdentityAccountStore
	ledger.MutationStore
}

type AccountResolver interface {
	ResolveDiscord(context.Context, string, string) (account.Account, error)
}

type CheckoutCreator interface {
	Create(context.Context, string) (payment.Checkout, error)
}

type BillingPortalCreator interface {
	CreateBillingPortal(context.Context, account.Account, string) (string, error)
}

type ContentGateway interface {
	Serve(context.Context, string, string) transport.Response
}

type App struct {
	publicOrigin     string
	publicPathPrefix string
	allowedRedirects map[string]struct{}
	oauth            OAuthStates
	discord          Discord
	accounts         Accounts
	resolver         AccountResolver
	checkout         CheckoutCreator
	billingPortal    BillingPortalCreator
	content          ContentGateway
	auditor          observability.Auditor
	sessions         *security.SessionKeyRing
	csrfKey          []byte
	now              func() time.Time
	identityLifetime time.Duration
	authzLifetime    time.Duration
}

func New(
	publicOrigin string,
	publicPathPrefix string,
	allowedRedirects []string,
	oauth OAuthStates,
	discord Discord,
	accounts Accounts,
	resolver AccountResolver,
	checkout CheckoutCreator,
	billingPortal BillingPortalCreator,
	content ContentGateway,
	auditor observability.Auditor,
	sessions *security.SessionKeyRing,
	csrfKey []byte,
	identityLifetime time.Duration,
	authzLifetime time.Duration,
) (*App, error) {
	parsed, err := url.Parse(publicOrigin)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.Path != "" ||
		!strings.HasPrefix(publicPathPrefix, "/") || strings.HasSuffix(publicPathPrefix, "/") ||
		strings.ContainsAny(publicPathPrefix, "?#") || strings.Contains(publicPathPrefix, "//") ||
		oauth == nil || discord == nil || accounts == nil || resolver == nil || checkout == nil || billingPortal == nil || content == nil || auditor == nil || sessions == nil ||
		len(csrfKey) < 32 || identityLifetime <= 0 || authzLifetime <= 0 || authzLifetime > identityLifetime {
		return nil, errors.New("invalid gateway configuration")
	}
	redirects := make(map[string]struct{}, len(allowedRedirects))
	for _, redirect := range allowedRedirects {
		redirects[redirect] = struct{}{}
	}
	return &App{
		publicOrigin: publicOrigin, publicPathPrefix: publicPathPrefix, allowedRedirects: redirects,
		oauth: oauth, discord: discord, accounts: accounts, resolver: resolver, checkout: checkout, billingPortal: billingPortal, content: content, auditor: auditor,
		sessions: sessions, csrfKey: append([]byte(nil), csrfKey...), now: time.Now,
		identityLifetime: identityLifetime, authzLifetime: authzLifetime,
	}, nil
}

func (a *App) Handle(ctx context.Context, request *http.Request) transport.Response {
	if request.URL.Path == "/health/live" {
		return transport.JSON(http.StatusOK, map[string]string{"status": "live"})
	}
	if err := a.validatePublicHost(request); err != nil {
		return transport.Error(http.StatusBadRequest, "invalid_host")
	}
	if isContentPath(request.URL.Path) {
		if request.Method != http.MethodGet {
			return transport.Error(http.StatusMethodNotAllowed, "method_not_allowed")
		}
		cookie, err := request.Cookie(security.SessionCookieName)
		if err != nil {
			return transport.Error(http.StatusForbidden, "access_denied")
		}
		return a.content.Serve(ctx, cookie.Value, request.URL.Path)
	}
	switch request.Method + " " + request.URL.Path {
	case "GET /health/ready":
		return transport.JSON(http.StatusOK, map[string]string{"status": "ready"})
	case "GET /auth/discord/start":
		return a.startDiscord(ctx, request)
	case "GET /auth/discord/callback":
		return a.finishDiscord(ctx, request)
	case "GET /auth/session/ready":
		return a.readySession(request)
	case "POST /auth/logout":
		return a.logout(request)
	case "GET /api/account", "GET /api/account/status":
		return a.accountStatus(ctx, request)
	case "POST /api/account/logout-all":
		return a.logoutAll(ctx, request)
	case "DELETE /api/account":
		return a.deleteAccount(ctx, request)
	case "POST /api/checkout":
		return a.createCheckout(ctx, request)
	case "POST /api/billing-portal":
		return a.createBillingPortal(ctx, request)
	default:
		return transport.Error(http.StatusNotFound, "not_found")
	}
}

func isContentPath(requestPath string) bool {
	return requestPath == "/app" || strings.HasPrefix(requestPath, "/app/") ||
		strings.HasPrefix(requestPath, "/api/content/") || strings.HasPrefix(requestPath, "/downloads/")
}

func (a *App) startDiscord(ctx context.Context, request *http.Request) transport.Response {
	redirect := request.URL.Query().Get("redirect")
	if redirect == "" {
		redirect = "/app/"
	}
	if _, ok := a.allowedRedirects[redirect]; !ok {
		return transport.Error(http.StatusBadRequest, "invalid_redirect")
	}
	state, err := a.oauth.Create(ctx, redirect)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "login_unavailable")
	}
	destination, err := a.discord.AuthorizationURL(state)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "login_unavailable")
	}
	return transport.Redirect(destination, security.OAuthStateCookie(state, 10*time.Minute))
}

func (a *App) finishDiscord(ctx context.Context, request *http.Request) transport.Response {
	code := request.URL.Query().Get("code")
	state := request.URL.Query().Get("state")
	binding, err := request.Cookie(security.OAuthStateCookieName)
	if err != nil || state == "" || len(state) != len(binding.Value) ||
		subtle.ConstantTimeCompare([]byte(state), []byte(binding.Value)) != 1 {
		return transport.Error(http.StatusBadRequest, "invalid_oauth_state")
	}
	record, err := a.oauth.Consume(ctx, state)
	if err != nil {
		return transport.Error(http.StatusBadRequest, "invalid_oauth_state")
	}
	identity, err := a.discord.ResolveIdentity(ctx, code)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "login_unavailable")
	}
	value, err := a.resolver.ResolveDiscord(ctx, identity.ID, identity.DisplayName)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "login_unavailable")
	}
	if err := a.auditSuccess(ctx, "auth.login", value.AccountID); err != nil {
		return transport.Error(http.StatusServiceUnavailable, "login_unavailable")
	}
	now := a.now().UTC()
	authzLifetime := time.Duration(0)
	plan := ""
	entitlementExpiry := time.Time{}
	if value.HasAccess(now) && value.Entitlement.Plan != "" {
		authzLifetime = a.authzLifetime
		plan = value.Entitlement.Plan
		entitlementExpiry = value.Entitlement.ValidUntil
	}
	token, _, err := a.sessions.Issue(
		value.AccountID, value.SessionEpoch, now, a.identityLifetime,
		authzLifetime, plan, entitlementExpiry,
	)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "login_unavailable")
	}
	readyURL := a.externalPath("/auth/session/ready") + "?next=" + url.QueryEscape(record.RedirectAfterLogin)
	return transport.Redirect(readyURL, security.SessionCookie(token, a.identityLifetime))
}

func (a *App) readySession(request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	redirect := request.URL.Query().Get("next")
	if _, allowed := a.allowedRedirects[redirect]; !allowed {
		return transport.Error(http.StatusBadRequest, "invalid_redirect")
	}
	csrfToken, err := security.IssueCSRF(a.csrfKey, claims.Nonce)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "session_unavailable")
	}
	return transport.Redirect(a.externalPath(redirect), security.CSRFCookie(csrfToken, time.Unix(claims.Expiry, 0).Sub(a.now())))
}

func (a *App) logout(request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	if err := security.ValidateStateChangingRequest(request, a.publicOrigin, claims.Nonce, a.csrfKey); err != nil {
		return transport.Error(http.StatusForbidden, "csrf_failed")
	}
	return transport.Redirect("/", security.ExpiredCookie(security.SessionCookieName))
}

func (a *App) accountStatus(ctx context.Context, request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	value, _, err := a.accounts.GetAccount(ctx, claims.Subject)
	if err != nil || value.SessionEpoch != claims.SessionEpoch {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	now := a.now().UTC()
	return transport.JSON(http.StatusOK, map[string]any{
		"account_id":   value.AccountID,
		"display_name": value.DisplayName,
		"status":       value.Status,
		"has_access":   value.HasAccess(now),
		"plan":         value.Entitlement.Plan,
		"valid_until":  value.Entitlement.ValidUntil,
	})
}

func (a *App) logoutAll(ctx context.Context, request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	if err := security.ValidateStateChangingRequest(request, a.publicOrigin, claims.Nonce, a.csrfKey); err != nil {
		return transport.Error(http.StatusForbidden, "csrf_failed")
	}
	err := ledger.MutateAccount(ctx, a.accounts, claims.Subject, "logout-all:"+claims.Nonce, func(value *account.Account) error {
		if value.SessionEpoch != claims.SessionEpoch {
			return auth.ErrAccessDenied
		}
		value.SessionEpoch++
		return nil
	})
	if err != nil {
		if errors.Is(err, auth.ErrAccessDenied) || errors.Is(err, storage.ErrNotFound) {
			return transport.Error(http.StatusUnauthorized, "unauthorized")
		}
		return transport.Error(http.StatusServiceUnavailable, "temporary_failure")
	}
	if err := a.auditSuccess(ctx, "account.logout_all", claims.Subject); err != nil {
		return transport.Error(http.StatusServiceUnavailable, "temporary_failure")
	}
	return transport.Redirect("/", security.ExpiredCookie(security.SessionCookieName))
}

func (a *App) deleteAccount(ctx context.Context, request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	if err := security.ValidateStateChangingRequest(request, a.publicOrigin, claims.Nonce, a.csrfKey); err != nil {
		return transport.Error(http.StatusForbidden, "csrf_failed")
	}
	err := ledger.MutateAccount(ctx, a.accounts, claims.Subject, "account-delete:"+claims.Nonce, func(value *account.Account) error {
		if value.SessionEpoch != claims.SessionEpoch {
			return auth.ErrAccessDenied
		}
		switch value.Status {
		case account.StatusPending, account.StatusCanceled:
		default:
			return errDeletionRequiresCancellation
		}
		value.DisplayName = "deleted-account"
		value.Status = account.StatusDeleted
		value.Entitlement = account.Entitlement{}
		value.Payment = account.Payment{}
		value.Checkout = account.Checkout{}
		value.SessionEpoch++
		return nil
	})
	if err != nil {
		switch {
		case errors.Is(err, auth.ErrAccessDenied), errors.Is(err, storage.ErrNotFound):
			return transport.Error(http.StatusUnauthorized, "unauthorized")
		case errors.Is(err, errDeletionRequiresCancellation):
			return transport.Error(http.StatusConflict, "cancel_subscription_first")
		default:
			return transport.Error(http.StatusServiceUnavailable, "temporary_failure")
		}
	}
	if err := a.auditSuccess(ctx, "account.deleted", claims.Subject); err != nil {
		return transport.Error(http.StatusServiceUnavailable, "temporary_failure")
	}
	response := transport.Secure(transport.Response{StatusCode: "204", Headers: map[string]string{}})
	response.Headers["Set-Cookie"] = security.ExpiredCookie(security.SessionCookieName).String()
	return response
}

func (a *App) createCheckout(ctx context.Context, request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	if err := security.ValidateStateChangingRequest(request, a.publicOrigin, claims.Nonce, a.csrfKey); err != nil {
		return transport.Error(http.StatusForbidden, "csrf_failed")
	}
	value, _, err := a.accounts.GetAccount(ctx, claims.Subject)
	if err != nil || value.SessionEpoch != claims.SessionEpoch {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	checkout, err := a.checkout.Create(ctx, claims.Subject)
	if err != nil {
		if errors.Is(err, payment.ErrCheckoutNotAllowed) {
			return transport.Error(http.StatusConflict, "checkout_not_allowed")
		}
		return transport.Error(http.StatusServiceUnavailable, "checkout_unavailable")
	}
	if err := a.auditSuccess(ctx, "payment.checkout_created", claims.Subject); err != nil {
		return transport.Error(http.StatusServiceUnavailable, "checkout_unavailable")
	}
	return transport.JSON(http.StatusOK, map[string]string{"checkout_id": checkout.ID, "url": checkout.URL})
}

func (a *App) createBillingPortal(ctx context.Context, request *http.Request) transport.Response {
	claims, ok := a.sessionClaims(request)
	if !ok {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	if err := security.ValidateStateChangingRequest(request, a.publicOrigin, claims.Nonce, a.csrfKey); err != nil {
		return transport.Error(http.StatusForbidden, "csrf_failed")
	}
	// Billing changes always require a fresh authoritative account read, even
	// when the session's normal authorization lease remains valid.
	value, _, err := a.accounts.GetAccount(ctx, claims.Subject)
	if err != nil || value.SessionEpoch != claims.SessionEpoch {
		return transport.Error(http.StatusUnauthorized, "unauthorized")
	}
	portalURL, err := a.billingPortal.CreateBillingPortal(ctx, value, a.publicOrigin+a.externalPath("/app/"))
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "billing_portal_unavailable")
	}
	if err := a.auditSuccess(ctx, "payment.portal_created", claims.Subject); err != nil {
		return transport.Error(http.StatusServiceUnavailable, "billing_portal_unavailable")
	}
	return transport.JSON(http.StatusOK, map[string]string{"url": portalURL})
}

func (a *App) auditSuccess(ctx context.Context, eventType, accountID string) error {
	return a.auditor.Record(ctx, observability.AuditEvent{
		Type: eventType, AccountID: accountID, Outcome: "success", OccurredAt: a.now().UTC(),
	})
}

func (a *App) sessionClaims(request *http.Request) (security.SessionClaims, bool) {
	cookie, err := request.Cookie(security.SessionCookieName)
	if err != nil {
		return security.SessionClaims{}, false
	}
	claims, err := a.sessions.Verify(cookie.Value, a.now().UTC())
	return claims, err == nil
}

func (a *App) validatePublicHost(request *http.Request) error {
	expected, _ := url.Parse(a.publicOrigin)
	host := request.Host
	if host == "" {
		host = request.Header.Get("Host")
	}
	forwardedProto := strings.ToLower(request.Header.Get("X-Forwarded-Proto"))
	if !strings.EqualFold(host, expected.Host) || forwardedProto != "https" {
		return errors.New("request did not use the public origin")
	}
	return nil
}

func (a *App) externalPath(internalPath string) string {
	return a.publicPathPrefix + internalPath
}
