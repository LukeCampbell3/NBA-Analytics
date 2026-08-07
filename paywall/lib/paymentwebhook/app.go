package paymentwebhook

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"mime"
	"net/http"

	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/payment"
	"github.com/jcthi/nba-analytics/paywall/storage"
	"github.com/jcthi/nba-analytics/paywall/transport"
)

type EventVerifier interface {
	VerifyWebhook([]byte, string) (payment.ProviderEvent, error)
}

type Processor interface {
	Process(context.Context, []byte, payment.ProviderEvent, string) error
}

type App struct {
	verifier    EventVerifier
	processor   Processor
	maximumBody int64
	auditor     observability.Auditor
}

func New(verifier EventVerifier, processor Processor, maximumBody int64) (*App, error) {
	return newApp(verifier, processor, maximumBody, nil)
}

func NewAudited(verifier EventVerifier, processor Processor, maximumBody int64, auditor observability.Auditor) (*App, error) {
	if auditor == nil {
		return nil, errors.New("webhook auditor is required")
	}
	return newApp(verifier, processor, maximumBody, auditor)
}

func newApp(verifier EventVerifier, processor Processor, maximumBody int64, auditor observability.Auditor) (*App, error) {
	if verifier == nil || processor == nil || maximumBody < 1024 || maximumBody > 1024*1024 {
		return nil, errors.New("invalid webhook app configuration")
	}
	return &App{verifier: verifier, processor: processor, maximumBody: maximumBody, auditor: auditor}, nil
}

func (a *App) Handle(ctx context.Context, request *http.Request, leaseOwner string) transport.Response {
	if request.Method != http.MethodPost || request.URL.Path != "/api/webhooks/stripe" || leaseOwner == "" {
		return transport.Error(http.StatusNotFound, "not_found")
	}
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return transport.Error(http.StatusUnsupportedMediaType, "invalid_content_type")
	}
	rawBody, err := io.ReadAll(io.LimitReader(request.Body, a.maximumBody+1))
	if err != nil || int64(len(rawBody)) > a.maximumBody {
		return transport.Error(http.StatusRequestEntityTooLarge, "payload_too_large")
	}
	event, err := a.verifier.VerifyWebhook(rawBody, request.Header.Get("Stripe-Signature"))
	if errors.Is(err, payment.ErrUnsupportedEvent) {
		if a.auditor != nil {
			digest := sha256.Sum256([]byte("unsupported-stripe-event:" + event.ID))
			if event.ID == "" || event.OccurredAt.IsZero() || a.auditor.Record(ctx, observability.AuditEvent{
				ID: hex.EncodeToString(digest[:16]), Type: "payment.webhook_ignored",
				Outcome: "denied", OccurredAt: event.OccurredAt,
			}) != nil {
				return transport.Error(http.StatusServiceUnavailable, "temporary_failure")
			}
		}
		return transport.JSON(http.StatusOK, map[string]bool{"received": true})
	}
	if err != nil {
		return transport.Error(http.StatusBadRequest, "invalid_signature")
	}
	if err := a.processor.Process(ctx, rawBody, event, leaseOwner); err != nil {
		if errors.Is(err, storage.ErrMalformed) {
			return transport.Error(http.StatusBadRequest, "invalid_event")
		}
		return transport.Error(http.StatusServiceUnavailable, "temporary_failure")
	}
	return transport.JSON(http.StatusOK, map[string]bool{"received": true})
}
