package paymentwebhook

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/payment"
)

type fakeVerifier struct {
	event payment.ProviderEvent
	err   error
}

func (v fakeVerifier) VerifyWebhook([]byte, string) (payment.ProviderEvent, error) {
	return v.event, v.err
}

type recordingProcessor struct {
	body []byte
	err  error
}

type recordingAuditor struct {
	event observability.AuditEvent
}

func (auditor *recordingAuditor) Record(_ context.Context, event observability.AuditEvent) error {
	auditor.event = event
	return nil
}

func (p *recordingProcessor) Process(_ context.Context, body []byte, _ payment.ProviderEvent, _ string) error {
	p.body = append([]byte(nil), body...)
	return p.err
}

func webhookRequest(body string) *http.Request {
	request, _ := http.NewRequest(http.MethodPost, "https://example.com/api/webhooks/stripe", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Stripe-Signature", "signature")
	return request
}

func TestWebhookPassesUnmodifiedBodyToProcessor(t *testing.T) {
	processor := &recordingProcessor{}
	app, err := New(fakeVerifier{event: payment.ProviderEvent{ID: "evt_123"}}, processor, 1024)
	if err != nil {
		t.Fatal(err)
	}
	body := "{ \n  \"id\": \"evt_123\"\n}"
	response := app.Handle(context.Background(), webhookRequest(body), "request-id")
	if response.StatusCode != "200" || string(processor.body) != body {
		t.Fatalf("response = %#v, body = %q", response, processor.body)
	}
}

func TestWebhookFailurePolicies(t *testing.T) {
	tests := []struct {
		name      string
		verifier  fakeVerifier
		processor *recordingProcessor
		want      string
	}{
		{"invalid signature", fakeVerifier{err: payment.ErrInvalidWebhookSignature}, &recordingProcessor{}, "400"},
		{"unsupported", fakeVerifier{err: payment.ErrUnsupportedEvent}, &recordingProcessor{}, "200"},
		{"provider unavailable", fakeVerifier{event: payment.ProviderEvent{ID: "evt_123"}}, &recordingProcessor{err: errors.New("unavailable")}, "503"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			app, _ := New(test.verifier, test.processor, 1024)
			response := app.Handle(context.Background(), webhookRequest("{}"), "request-id")
			if response.StatusCode != test.want {
				t.Fatalf("status = %s, want %s", response.StatusCode, test.want)
			}
		})
	}
}

func TestUnsupportedSignedWebhookIsDurablyIgnored(t *testing.T) {
	auditor := &recordingAuditor{}
	processor := &recordingProcessor{}
	app, err := NewAudited(fakeVerifier{
		event: payment.ProviderEvent{ID: "evt_ignored", Type: "invoice.paid", OccurredAt: time.Unix(100, 0)},
		err:   payment.ErrUnsupportedEvent,
	}, processor, 1024, auditor)
	if err != nil {
		t.Fatal(err)
	}
	response := app.Handle(context.Background(), webhookRequest("{}"), "request-id")
	if response.StatusCode != "200" || auditor.event.Type != "payment.webhook_ignored" || auditor.event.Outcome != "denied" {
		t.Fatalf("response = %#v, audit = %#v", response, auditor.event)
	}
}
