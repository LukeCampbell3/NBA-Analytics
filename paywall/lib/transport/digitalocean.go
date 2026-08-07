package transport

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/jcthi/nba-analytics/paywall/security"
)

type HTTPEvent struct {
	Body            string            `json:"body"`
	Headers         map[string]string `json:"headers"`
	IsBase64Encoded bool              `json:"isBase64Encoded"`
	Method          string            `json:"method"`
	Path            string            `json:"path"`
	QueryString     string            `json:"queryString"`
}

type RawEvent struct {
	HTTP HTTPEvent `json:"http"`
}

type Response struct {
	Body       string            `json:"body,omitempty"`
	StatusCode string            `json:"statusCode"`
	Headers    map[string]string `json:"headers,omitempty"`
}

func (event RawEvent) Request(publicOrigin string, maximumBody int64) (*http.Request, error) {
	if maximumBody < 0 || int64(len(event.HTTP.Body)) > maximumBody*2+16 {
		return nil, fmt.Errorf("request body exceeds limit")
	}
	body := []byte(event.HTTP.Body)
	if event.HTTP.IsBase64Encoded {
		decoded, err := base64.StdEncoding.DecodeString(event.HTTP.Body)
		if err != nil {
			return nil, fmt.Errorf("invalid base64 request body")
		}
		body = decoded
	}
	if int64(len(body)) > maximumBody {
		return nil, fmt.Errorf("request body exceeds limit")
	}
	base, err := url.Parse(publicOrigin)
	if err != nil {
		return nil, err
	}
	path := event.HTTP.Path
	if path == "" {
		path = "/"
	}
	requestURL := *base
	requestURL.Path = path
	requestURL.RawQuery = event.HTTP.QueryString
	request, err := http.NewRequest(event.HTTP.Method, requestURL.String(), strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	for name, value := range event.HTTP.Headers {
		request.Header.Set(name, value)
	}
	if host := request.Header.Get("Host"); host != "" {
		request.Host = host
	}
	return request, nil
}

func JSON(status int, value any) Response {
	body, err := json.Marshal(value)
	if err != nil {
		return Error(http.StatusInternalServerError, "internal_error")
	}
	response := Response{
		Body: string(body), StatusCode: strconv.Itoa(status),
		Headers: map[string]string{"Content-Type": "application/json", "Cache-Control": "no-store"},
	}
	return Secure(response)
}

func Error(status int, code string) Response {
	return JSON(status, map[string]string{"error": code})
}

func Redirect(location string, cookie *http.Cookie) Response {
	response := Response{StatusCode: "302", Headers: map[string]string{"Location": location, "Cache-Control": "no-store"}}
	if cookie != nil {
		response.Headers["Set-Cookie"] = cookie.String()
	}
	return Secure(response)
}

func Secure(response Response) Response {
	if response.Headers == nil {
		response.Headers = make(map[string]string)
	}
	header := make(http.Header)
	security.SetSecurityHeaders(header)
	for name, values := range header {
		if len(values) > 0 {
			response.Headers[name] = values[0]
		}
	}
	return response
}
