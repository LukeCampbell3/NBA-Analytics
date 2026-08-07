package auth

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestDiscordRequestsOnlyIdentifyAndResolvesServerSide(t *testing.T) {
	var tokenCalls, userCalls int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/token":
			tokenCalls++
			if request.Method != http.MethodPost || request.FormValue("client_secret") != "client-secret" || request.FormValue("code") != "oauth-code" {
				t.Errorf("unexpected token request")
			}
			response.Header().Set("Content-Type", "application/json")
			fmt.Fprint(response, `{"access_token":"short-lived-token","token_type":"Bearer"}`)
		case "/users/@me":
			userCalls++
			if request.Header.Get("Authorization") != "Bearer short-lived-token" {
				t.Errorf("missing bearer token")
			}
			response.Header().Set("Content-Type", "application/json")
			fmt.Fprint(response, `{"id":"123456","username":"member","global_name":"Member Name","email":"must-not-be-used@example.com"}`)
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()

	client, err := NewDiscordClient(DiscordConfig{
		ClientID: "client-id", ClientSecret: "client-secret", RedirectURI: "https://example.com/auth/discord/callback",
	}, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	client.authorizeURL = server.URL + "/authorize"
	client.tokenURL = server.URL + "/token"
	client.userURL = server.URL + "/users/@me"
	authorizationURL, err := client.AuthorizationURL("oauth-state")
	if err != nil {
		t.Fatal(err)
	}
	parsed, _ := url.Parse(authorizationURL)
	if parsed.Query().Get("scope") != "identify" || parsed.Query().Get("state") != "oauth-state" || parsed.Query().Get("response_type") != "code" {
		t.Fatalf("authorization query = %v", parsed.Query())
	}
	identity, err := client.ResolveIdentity(context.Background(), "oauth-code")
	if err != nil {
		t.Fatal(err)
	}
	if identity.ID != "123456" || identity.DisplayName != "Member Name" || tokenCalls != 1 || userCalls != 1 {
		t.Fatalf("identity = %#v, tokenCalls=%d userCalls=%d", identity, tokenCalls, userCalls)
	}
}
