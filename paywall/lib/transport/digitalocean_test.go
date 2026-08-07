package transport

import (
	"encoding/base64"
	"strings"
	"testing"
)

func TestRawEventPreservesBodyQueryAndHeaders(t *testing.T) {
	rawBody := []byte("raw\x00body")
	event := RawEvent{HTTP: HTTPEvent{
		Body: base64.StdEncoding.EncodeToString(rawBody), IsBase64Encoded: true,
		Method: "POST", Path: "/api/webhooks/stripe", QueryString: "a=1&a=2",
		Headers: map[string]string{"host": "example.com", "stripe-signature": "signature"},
	}}
	request, err := event.Request("https://example.com", 1024)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, len(rawBody))
	if _, err := request.Body.Read(buffer); err != nil {
		t.Fatal(err)
	}
	if string(buffer) != string(rawBody) || request.URL.RawQuery != "a=1&a=2" || request.Header.Get("Stripe-Signature") != "signature" {
		t.Fatalf("request was not preserved: %#v", request)
	}
}

func TestRawEventRejectsOversizedEncodedBody(t *testing.T) {
	event := RawEvent{HTTP: HTTPEvent{Body: base64.StdEncoding.EncodeToString([]byte(strings.Repeat("x", 100))), IsBase64Encoded: true, Method: "POST"}}
	if _, err := event.Request("https://example.com", 10); err == nil {
		t.Fatal("oversized body was accepted")
	}
}
