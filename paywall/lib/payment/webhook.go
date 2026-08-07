package payment

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/ledger"
	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

var errDeletedAccount = errors.New("payment event targets a deleted account")

type EventLedger interface {
	Claim(context.Context, string, string, string, string, string) (storage.ClaimedPaymentEvent, error)
	MarkApplied(context.Context, storage.ClaimedPaymentEvent) error
}

type WebhookProcessor struct {
	provider         Provider
	events           EventLedger
	accounts         ledger.MutationStore
	piiKeyID         string
	piiEncryptionKey []byte
	auditor          observability.Auditor
}

func NewWebhookProcessor(provider Provider, events EventLedger, accounts ledger.MutationStore, piiKeyID string, piiEncryptionKey []byte) (*WebhookProcessor, error) {
	return newWebhookProcessor(provider, events, accounts, piiKeyID, piiEncryptionKey, nil)
}

func NewAuditedWebhookProcessor(provider Provider, events EventLedger, accounts ledger.MutationStore, piiKeyID string, piiEncryptionKey []byte, auditor observability.Auditor) (*WebhookProcessor, error) {
	if auditor == nil {
		return nil, fmt.Errorf("webhook auditor is required")
	}
	return newWebhookProcessor(provider, events, accounts, piiKeyID, piiEncryptionKey, auditor)
}

func newWebhookProcessor(provider Provider, events EventLedger, accounts ledger.MutationStore, piiKeyID string, piiEncryptionKey []byte, auditor observability.Auditor) (*WebhookProcessor, error) {
	if provider == nil || events == nil || accounts == nil || piiKeyID == "" || len(piiEncryptionKey) != 32 {
		return nil, fmt.Errorf("invalid webhook processor configuration")
	}
	return &WebhookProcessor{
		provider: provider, events: events, accounts: accounts,
		piiKeyID: piiKeyID, piiEncryptionKey: append([]byte(nil), piiEncryptionKey...), auditor: auditor,
	}, nil
}

// Process assumes raw-body signature verification and event-type allowlisting
// have already succeeded. It applies only an authoritative absolute snapshot.
func (p *WebhookProcessor) Process(ctx context.Context, rawBody []byte, event ProviderEvent, leaseOwner string) error {
	digest := sha256.Sum256(rawBody)
	claim, err := p.events.Claim(ctx, event.ID, event.Type, event.AccountID, hex.EncodeToString(digest[:]), leaseOwner)
	if err != nil {
		return err
	}
	if claim.Event.Status == "applied" {
		return nil
	}
	snapshot, err := p.provider.GetAuthoritativeEntitlement(ctx, SubscriptionReference{
		AccountID: event.AccountID, EventID: event.ID, SubscriptionID: event.SubscriptionID,
	})
	if err != nil {
		return fmt.Errorf("fetch authoritative entitlement: %w", err)
	}
	if err := validateSnapshot(snapshot); err != nil {
		return err
	}
	encryptedCustomer, err := security.EncryptField(p.piiKeyID, p.piiEncryptionKey, event.AccountID, snapshot.CustomerID)
	if err != nil {
		return err
	}
	encryptedSubscription, err := security.EncryptField(p.piiKeyID, p.piiEncryptionKey, event.AccountID, snapshot.SubscriptionID)
	if err != nil {
		return err
	}
	err = ledger.MutateAccount(ctx, p.accounts, event.AccountID, "stripe-event:"+event.ID, func(value *account.Account) error {
		if value.Status == account.StatusDeleted {
			return errDeletedAccount
		}
		value.Status = snapshot.Status
		value.Entitlement = snapshotEntitlement(snapshot)
		value.Payment.CustomerID = encryptedCustomer
		value.Payment.SubscriptionID = encryptedSubscription
		return nil
	})
	if errors.Is(err, errDeletedAccount) {
		return p.events.MarkApplied(ctx, claim)
	}
	if err != nil {
		return err
	}
	if p.auditor != nil {
		auditDigest := sha256.Sum256([]byte("stripe-event:" + event.ID))
		if err := p.auditor.Record(ctx, observability.AuditEvent{
			ID: hex.EncodeToString(auditDigest[:16]), Type: "payment.entitlement_reconciled",
			AccountID: event.AccountID, Outcome: "success", OccurredAt: claim.Event.ReceivedAt,
		}); err != nil {
			return err
		}
	}
	if err := p.events.MarkApplied(ctx, claim); err != nil {
		return err
	}
	return nil
}
