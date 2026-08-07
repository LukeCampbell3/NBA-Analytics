package gateway

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"testing"
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

type fakeDiscord struct{}

type fakeCheckout struct{}

type fakeBillingPortal struct{}

type fakeContent struct{}

type fakeAuditor struct{}

func (fakeAuditor) Record(context.Context, observability.AuditEvent) error { return nil }

func (fakeCheckout) Create(context.Context, string) (payment.Checkout, error) {
	return payment.Checkout{ID: "cs_test", URL: "https://checkout.example/cs_test"}, nil
}

func (fakeBillingPortal) CreateBillingPortal(context.Context, account.Account, string) (string, error) {
	return "https://billing.stripe.example/session", nil
}

func (fakeContent) Serve(context.Context, string, string) transport.Response {
	return transport.Response{StatusCode: "200"}
}

func (fakeDiscord) AuthorizationURL(state string) (string, error) {
	return "https://discord.example/authorize?state=" + url.QueryEscape(state), nil
}

func (fakeDiscord) ResolveIdentity(context.Context, string) (auth.DiscordIdentity, error) {
	return auth.DiscordIdentity{ID: "123456789", DisplayName: "member"}, nil
}

type gatewayFixture struct {
	app      *App
	accounts *storage.AccountStore
}

func newGatewayFixture(t *testing.T) gatewayFixture {
	t.Helper()
	objects := storage.NewMemoryStore()
	accounts := storage.NewAccountStore(objects)
	resolver, err := ledger.NewService(accounts, []byte("index-key-material-that-is-at-least-32-bytes"), nil)
	if err != nil {
		t.Fatal(err)
	}
	oauthStates, err := auth.NewOAuthStateService(objects, []string{"/app/", "/payment/return"}, 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	sessions, err := security.NewSessionKeyRing("current", map[string][]byte{
		"current": []byte("session-key-material-that-is-at-least-32-bytes"),
	}, "example.com", "paid-site")
	if err != nil {
		t.Fatal(err)
	}
	app, err := New(
		"https://example.com", "/functions/paywall/gateway", []string{"/app/", "/payment/return"},
		oauthStates, fakeDiscord{}, accounts, resolver, fakeCheckout{}, fakeBillingPortal{}, fakeContent{}, fakeAuditor{}, sessions,
		[]byte("csrf-key-material-that-is-at-least-32-bytes"), 7*24*time.Hour, 10*time.Minute,
	)
	if err != nil {
		t.Fatal(err)
	}
	return gatewayFixture{app: app, accounts: accounts}
}

func gatewayRequest(method, target string) *http.Request {
	request, _ := http.NewRequest(method, "https://example.com"+target, strings.NewReader("{}"))
	request.Host = "example.com"
	request.Header.Set("Host", "example.com")
	request.Header.Set("X-Forwarded-Proto", "https")
	return request
}

func gatewayRequestFromExternal(method, target string) *http.Request {
	const prefix = "/functions/paywall/gateway"
	return gatewayRequest(method, strings.TrimPrefix(target, prefix))
}

func responseCookie(t *testing.T, response transport.Response, name string) *http.Cookie {
	t.Helper()
	raw := response.Headers["Set-Cookie"]
	parsed := (&http.Response{Header: http.Header{"Set-Cookie": []string{raw}}}).Cookies()
	for _, cookie := range parsed {
		if cookie.Name == name {
			return cookie
		}
	}
	t.Fatalf("cookie %q not found in %q", name, raw)
	return nil
}

func TestDiscordLoginCreatesIdentitySessionThenCSRFCookie(t *testing.T) {
	fixture := newGatewayFixture(t)
	ctx := context.Background()
	start := fixture.app.Handle(ctx, gatewayRequest(http.MethodGet, "/auth/discord/start?redirect=/app/"))
	if start.StatusCode != "302" || !strings.HasPrefix(start.Headers["Location"], "https://discord.example/") {
		t.Fatalf("start response = %#v", start)
	}
	stateCookie := responseCookie(t, start, security.OAuthStateCookieName)
	state := strings.TrimPrefix(start.Headers["Location"], "https://discord.example/authorize?state=")
	state, _ = url.QueryUnescape(state)

	callbackRequest := gatewayRequest(http.MethodGet, "/auth/discord/callback?code=code&state="+url.QueryEscape(state))
	callbackRequest.AddCookie(stateCookie)
	callback := fixture.app.Handle(ctx, callbackRequest)
	if callback.StatusCode != "302" || !strings.HasPrefix(callback.Headers["Location"], "/functions/paywall/gateway/auth/session/ready") {
		t.Fatalf("callback response = %#v", callback)
	}
	sessionCookie := responseCookie(t, callback, security.SessionCookieName)

	readyRequest := gatewayRequestFromExternal(http.MethodGet, callback.Headers["Location"])
	readyRequest.AddCookie(sessionCookie)
	ready := fixture.app.Handle(ctx, readyRequest)
	if ready.StatusCode != "302" || ready.Headers["Location"] != "/functions/paywall/gateway/app/" {
		t.Fatalf("ready response = %#v", ready)
	}
	csrfCookie := responseCookie(t, ready, security.CSRFCookieName)

	statusRequest := gatewayRequest(http.MethodGet, "/api/account/status")
	statusRequest.AddCookie(sessionCookie)
	status := fixture.app.Handle(ctx, statusRequest)
	if status.StatusCode != "200" || !strings.Contains(status.Body, `"status":"pending"`) || !strings.Contains(status.Body, `"has_access":false`) {
		t.Fatalf("status response = %#v", status)
	}

	logoutRequest := gatewayRequest(http.MethodPost, "/auth/logout")
	logoutRequest.Header.Set("Origin", "https://example.com")
	logoutRequest.Header.Set("Content-Type", "application/json")
	logoutRequest.Header.Set("X-CSRF-Token", csrfCookie.Value)
	logoutRequest.AddCookie(sessionCookie)
	logoutRequest.AddCookie(csrfCookie)
	logout := fixture.app.Handle(ctx, logoutRequest)
	if logout.StatusCode != "302" || responseCookie(t, logout, security.SessionCookieName).MaxAge != -1 {
		t.Fatalf("logout response = %#v", logout)
	}

	replayRequest := gatewayRequest(http.MethodGet, "/auth/discord/callback?code=code&state="+url.QueryEscape(state))
	replayRequest.AddCookie(stateCookie)
	replay := fixture.app.Handle(ctx, replayRequest)
	if replay.StatusCode != "400" {
		t.Fatalf("oauth replay response = %#v", replay)
	}
}

func TestGatewayRejectsDirectFunctionHost(t *testing.T) {
	fixture := newGatewayFixture(t)
	request := gatewayRequest(http.MethodGet, "/auth/discord/start")
	request.Host = "function.example"
	request.Header.Set("Host", "function.example")
	response := fixture.app.Handle(context.Background(), request)
	if response.StatusCode != "400" {
		t.Fatalf("direct function host response = %#v", response)
	}
}

func TestBillingPortalRequiresCSRFAndFreshAccount(t *testing.T) {
	fixture := newGatewayFixture(t)
	ctx := context.Background()
	start := fixture.app.Handle(ctx, gatewayRequest(http.MethodGet, "/auth/discord/start?redirect=/app/"))
	stateCookie := responseCookie(t, start, security.OAuthStateCookieName)
	state := strings.TrimPrefix(start.Headers["Location"], "https://discord.example/authorize?state=")
	state, _ = url.QueryUnescape(state)
	callbackRequest := gatewayRequest(http.MethodGet, "/auth/discord/callback?code=code&state="+url.QueryEscape(state))
	callbackRequest.AddCookie(stateCookie)
	callback := fixture.app.Handle(ctx, callbackRequest)
	sessionCookie := responseCookie(t, callback, security.SessionCookieName)
	readyRequest := gatewayRequestFromExternal(http.MethodGet, callback.Headers["Location"])
	readyRequest.AddCookie(sessionCookie)
	ready := fixture.app.Handle(ctx, readyRequest)
	csrfCookie := responseCookie(t, ready, security.CSRFCookieName)

	missingCSRF := gatewayRequest(http.MethodPost, "/api/billing-portal")
	missingCSRF.AddCookie(sessionCookie)
	if response := fixture.app.Handle(ctx, missingCSRF); response.StatusCode != "403" {
		t.Fatalf("billing portal without CSRF = %#v", response)
	}

	request := gatewayRequest(http.MethodPost, "/api/billing-portal")
	request.Header.Set("Origin", "https://example.com")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-CSRF-Token", csrfCookie.Value)
	request.AddCookie(sessionCookie)
	request.AddCookie(csrfCookie)
	response := fixture.app.Handle(ctx, request)
	if response.StatusCode != "200" || !strings.Contains(response.Body, `"url":"https://billing.stripe.example/session"`) {
		t.Fatalf("billing portal response = %#v", response)
	}
}

func TestPendingAccountDeletionPseudonymizesAndRevokesSession(t *testing.T) {
	fixture := newGatewayFixture(t)
	ctx := context.Background()
	start := fixture.app.Handle(ctx, gatewayRequest(http.MethodGet, "/auth/discord/start?redirect=/app/"))
	stateCookie := responseCookie(t, start, security.OAuthStateCookieName)
	state := strings.TrimPrefix(start.Headers["Location"], "https://discord.example/authorize?state=")
	state, _ = url.QueryUnescape(state)
	callbackRequest := gatewayRequest(http.MethodGet, "/auth/discord/callback?code=code&state="+url.QueryEscape(state))
	callbackRequest.AddCookie(stateCookie)
	callback := fixture.app.Handle(ctx, callbackRequest)
	sessionCookie := responseCookie(t, callback, security.SessionCookieName)
	readyRequest := gatewayRequestFromExternal(http.MethodGet, callback.Headers["Location"])
	readyRequest.AddCookie(sessionCookie)
	csrfCookie := responseCookie(t, fixture.app.Handle(ctx, readyRequest), security.CSRFCookieName)
	claimsRequest := gatewayRequest(http.MethodGet, "/api/account")
	claimsRequest.AddCookie(sessionCookie)
	claims, ok := fixture.app.sessionClaims(claimsRequest)
	if !ok {
		t.Fatal("issued session did not verify")
	}

	request := gatewayRequest(http.MethodDelete, "/api/account")
	request.Header.Set("Origin", "https://example.com")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-CSRF-Token", csrfCookie.Value)
	request.AddCookie(sessionCookie)
	request.AddCookie(csrfCookie)
	response := fixture.app.Handle(ctx, request)
	if response.StatusCode != "204" {
		t.Fatalf("delete response = %#v", response)
	}
	deleted, _, err := fixture.accounts.GetAccount(ctx, claims.Subject)
	if err != nil {
		t.Fatal(err)
	}
	if deleted.Status != account.StatusDeleted || deleted.DisplayName != "deleted-account" ||
		deleted.SessionEpoch == claims.SessionEpoch || deleted.Payment.CustomerID != nil || !deleted.Entitlement.ValidUntil.IsZero() {
		t.Fatalf("deleted account = %#v", deleted)
	}
}
