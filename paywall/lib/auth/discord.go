package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type DiscordConfig struct {
	ClientID     string
	ClientSecret string
	RedirectURI  string
}

type DiscordIdentity struct {
	ID          string
	DisplayName string
}

type DiscordClient struct {
	config       DiscordConfig
	httpClient   *http.Client
	authorizeURL string
	tokenURL     string
	userURL      string
}

func NewDiscordClient(config DiscordConfig, httpClient *http.Client) (*DiscordClient, error) {
	redirect, err := url.Parse(config.RedirectURI)
	if config.ClientID == "" || config.ClientSecret == "" || err != nil || redirect.Scheme != "https" || redirect.Host == "" {
		return nil, fmt.Errorf("invalid Discord OAuth configuration")
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}
	return &DiscordClient{
		config: config, httpClient: httpClient,
		authorizeURL: "https://discord.com/oauth2/authorize",
		tokenURL:     "https://discord.com/api/v10/oauth2/token",
		userURL:      "https://discord.com/api/v10/users/@me",
	}, nil
}

func (c *DiscordClient) AuthorizationURL(state string) (string, error) {
	if state == "" {
		return "", ErrInvalidOAuthState
	}
	values := url.Values{
		"client_id":     {c.config.ClientID},
		"redirect_uri":  {c.config.RedirectURI},
		"response_type": {"code"},
		"scope":         {"identify"},
		"state":         {state},
	}
	return c.authorizeURL + "?" + values.Encode(), nil
}

func (c *DiscordClient) ResolveIdentity(ctx context.Context, code string) (DiscordIdentity, error) {
	if code == "" || len(code) > 2048 {
		return DiscordIdentity{}, fmt.Errorf("invalid Discord authorization code")
	}
	form := url.Values{
		"client_id":     {c.config.ClientID},
		"client_secret": {c.config.ClientSecret},
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {c.config.RedirectURI},
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return DiscordIdentity{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := c.httpClient.Do(request)
	if err != nil {
		return DiscordIdentity{}, err
	}
	body, err := readLimitedResponse(response.Body, 256*1024)
	closeErr := response.Body.Close()
	if err != nil || response.StatusCode != http.StatusOK {
		return DiscordIdentity{}, fmt.Errorf("Discord token exchange failed")
	}
	if closeErr != nil {
		return DiscordIdentity{}, fmt.Errorf("close Discord token response: %w", closeErr)
	}
	var token struct {
		AccessToken string `json:"access_token"`
		TokenType   string `json:"token_type"`
	}
	if err := json.Unmarshal(body, &token); err != nil || token.AccessToken == "" || !strings.EqualFold(token.TokenType, "Bearer") {
		return DiscordIdentity{}, fmt.Errorf("Discord returned an invalid token response")
	}

	request, err = http.NewRequestWithContext(ctx, http.MethodGet, c.userURL, nil)
	if err != nil {
		return DiscordIdentity{}, err
	}
	request.Header.Set("Authorization", "Bearer "+token.AccessToken)
	response, err = c.httpClient.Do(request)
	if err != nil {
		return DiscordIdentity{}, err
	}
	body, err = readLimitedResponse(response.Body, 256*1024)
	closeErr = response.Body.Close()
	if err != nil || response.StatusCode != http.StatusOK {
		return DiscordIdentity{}, fmt.Errorf("Discord identity request failed")
	}
	if closeErr != nil {
		return DiscordIdentity{}, fmt.Errorf("close Discord identity response: %w", closeErr)
	}
	var user struct {
		ID         string  `json:"id"`
		Username   string  `json:"username"`
		GlobalName *string `json:"global_name"`
	}
	if err := json.Unmarshal(body, &user); err != nil || user.ID == "" || user.Username == "" {
		return DiscordIdentity{}, fmt.Errorf("Discord returned an invalid identity")
	}
	displayName := user.Username
	if user.GlobalName != nil && *user.GlobalName != "" {
		displayName = *user.GlobalName
	}
	return DiscordIdentity{ID: user.ID, DisplayName: displayName}, nil
}

func readLimitedResponse(reader io.Reader, limit int64) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, errors.New("response exceeds limit")
	}
	return body, nil
}
