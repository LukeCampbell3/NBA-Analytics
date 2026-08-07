package payment

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/ledger"
	"github.com/jcthi/nba-analytics/paywall/security"
)

type EntitlementProvider interface {
	GetAuthoritativeEntitlement(context.Context, SubscriptionReference) (EntitlementSnapshot, error)
}

type Reconciler struct {
	provider     EntitlementProvider
	accounts     ledger.MutationStore
	piiCurrentID string
	piiKeys      map[string][]byte
	staleAfter   time.Duration
	now          func() time.Time
}

func NewReconciler(
	provider EntitlementProvider,
	accounts ledger.MutationStore,
	piiCurrentID string,
	piiKeys map[string][]byte,
	staleAfter time.Duration,
) (*Reconciler, error) {
	if provider == nil || accounts == nil || piiCurrentID == "" || len(piiKeys[piiCurrentID]) != 32 || staleAfter <= 0 {
		return nil, fmt.Errorf("invalid entitlement reconciler configuration")
	}
	keys := make(map[string][]byte, len(piiKeys))
	for keyID, key := range piiKeys {
		keys[keyID] = append([]byte(nil), key...)
	}
	return &Reconciler{
		provider: provider, accounts: accounts, piiCurrentID: piiCurrentID,
		piiKeys: keys, staleAfter: staleAfter, now: time.Now,
	}, nil
}

// Reconcile refreshes stale paid accounts from the provider. Accounts that
// have never acquired a provider subscription remain pending without making an
// external call.
func (r *Reconciler) Reconcile(ctx context.Context, value account.Account) (account.Account, error) {
	now := r.now().UTC()
	if value.Payment.SubscriptionID == nil || (!value.Entitlement.ProviderVerifiedAt.IsZero() &&
		now.Sub(value.Entitlement.ProviderVerifiedAt) < r.staleAfter) {
		return value, nil
	}
	subscriptionID, err := security.DecryptField(r.piiKeys, value.AccountID, value.Payment.SubscriptionID)
	if err != nil {
		return account.Account{}, err
	}
	snapshot, err := r.provider.GetAuthoritativeEntitlement(ctx, SubscriptionReference{
		AccountID: value.AccountID, SubscriptionID: subscriptionID,
	})
	if err != nil {
		return account.Account{}, err
	}
	if err := validateSnapshot(snapshot); err != nil {
		return account.Account{}, err
	}
	encryptedCustomer, err := security.EncryptField(r.piiCurrentID, r.piiKeys[r.piiCurrentID], value.AccountID, snapshot.CustomerID)
	if err != nil {
		return account.Account{}, err
	}
	encryptedSubscription, err := security.EncryptField(r.piiCurrentID, r.piiKeys[r.piiCurrentID], value.AccountID, snapshot.SubscriptionID)
	if err != nil {
		return account.Account{}, err
	}
	digest := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%s\x00%s\x00%d", value.AccountID, snapshot.SubscriptionID, snapshot.Status, snapshot.ProviderUpdatedAt.UnixNano())))
	mutationID := "provider-reconcile:" + hex.EncodeToString(digest[:16])
	err = ledger.MutateAccount(ctx, r.accounts, value.AccountID, mutationID, func(candidate *account.Account) error {
		candidate.Status = snapshot.Status
		candidate.Entitlement = snapshotEntitlement(snapshot)
		candidate.Payment.CustomerID = encryptedCustomer
		candidate.Payment.SubscriptionID = encryptedSubscription
		return nil
	})
	if err != nil {
		return account.Account{}, err
	}
	updated, _, err := r.accounts.GetAccount(ctx, value.AccountID)
	return updated, err
}

func validateSnapshot(snapshot EntitlementSnapshot) error {
	if !account.IsKnownStatus(snapshot.Status) || snapshot.Plan == "" || snapshot.Source == "" ||
		snapshot.ValidUntil.IsZero() || snapshot.ProviderVerifiedAt.IsZero() || snapshot.ProviderUpdatedAt.IsZero() ||
		snapshot.CustomerID == "" || snapshot.SubscriptionID == "" {
		return fmt.Errorf("provider returned invalid entitlement snapshot")
	}
	return nil
}

func snapshotEntitlement(snapshot EntitlementSnapshot) account.Entitlement {
	return account.Entitlement{
		Plan: snapshot.Plan, Source: snapshot.Source,
		ValidFrom: snapshot.ValidFrom, ValidUntil: snapshot.ValidUntil,
		ProviderVerifiedAt: snapshot.ProviderVerifiedAt, ProviderUpdatedAt: snapshot.ProviderUpdatedAt,
	}
}
