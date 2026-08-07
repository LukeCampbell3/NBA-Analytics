package security

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestStateChangingRequestRequiresOriginJSONAndCSRF(t *testing.T) {
	key := []byte("csrf-signing-key-material-is-32-bytes!")
	token, _ := IssueCSRF(key, "session-nonce")
	valid := func() *http.Request {
		request := httptest.NewRequest(http.MethodPost, "https://example.com/api/checkout", nil)
		request.Header.Set("Origin", "https://example.com")
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("X-CSRF-Token", token)
		request.AddCookie(CSRFCookie(token, 60))
		return request
	}
	if err := ValidateStateChangingRequest(valid(), "https://example.com", "session-nonce", key); err != nil {
		t.Fatal(err)
	}
	mutations := []func(*http.Request){
		func(request *http.Request) { request.Method = http.MethodGet },
		func(request *http.Request) { request.Header.Set("Origin", "https://evil.example") },
		func(request *http.Request) { request.Header.Set("Content-Type", "text/plain") },
		func(request *http.Request) { request.Header.Set("X-CSRF-Token", "wrong") },
	}
	for _, mutate := range mutations {
		request := valid()
		mutate(request)
		if err := ValidateStateChangingRequest(request, "https://example.com", "session-nonce", key); !errors.Is(err, ErrUnsafeBrowserRequest) {
			t.Fatalf("unsafe request error = %v", err)
		}
	}
}
