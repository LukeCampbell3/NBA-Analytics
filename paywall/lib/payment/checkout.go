package payment

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"time"

	"github.com/jcthi/nba-analytics/paywall/account"
	"github.com/jcthi/nba-analytics/paywall/ledger"
)

var (
	ErrCheckoutNotAllowed = errors.New("checkout is not allowed for this account")
	useExistingLock       = errors.New("reuse existing checkout lock")
	checkoutAlreadyStored = errors.New("checkout is already stored")
)

type CheckoutService struct {
	provider Provider
	accounts ledger.MutationStore
	now      func() time.Time
	cooldown time.Duration
}

func NewCheckoutService(provider Provider, accounts ledger.MutationStore, cooldown time.Duration) (*CheckoutService, error) {
	if provider == nil || accounts == nil || cooldown < 10*time.Second || cooldown > 10*time.Minute {
		return nil, fmt.Errorf("invalid checkout service configuration")
	}
	return &CheckoutService{provider: provider, accounts: accounts, now: time.Now, cooldown: cooldown}, nil
}

// Create acquires a global account lock before calling the provider. If a
// function crashes after the lock, the next invocation reuses the stored key.
func (s *CheckoutService) Create(ctx context.Context, accountID string) (Checkout, error) {
	candidateKey, err := randomCheckoutKey()
	if err != nil {
		return Checkout{}, err
	}
	now := s.now().UTC()
	idempotencyKey := candidateKey
	var providerAccount account.Account
	err = ledger.MutateAccount(ctx, s.accounts, accountID, "checkout-lock:"+candidateKey, func(value *account.Account) error {
		providerAccount = *value
		switch value.Status {
		case account.StatusPending, account.StatusPastDue, account.StatusCanceled:
		default:
			return ErrCheckoutNotAllowed
		}
		if value.Checkout.LockUntil != nil && now.Before(*value.Checkout.LockUntil) && value.Checkout.IdempotencyKey != "" {
			idempotencyKey = value.Checkout.IdempotencyKey
			return useExistingLock
		}
		lockUntil := now.Add(s.cooldown)
		value.Checkout.LockUntil = &lockUntil
		value.Checkout.IdempotencyKey = candidateKey
		providerAccount = *value
		return nil
	})
	if err != nil && !errors.Is(err, useExistingLock) {
		return Checkout{}, err
	}
	checkout, err := s.provider.CreateCheckout(ctx, providerAccount, idempotencyKey)
	if err != nil {
		return Checkout{}, err
	}
	if checkout.ID == "" || checkout.URL == "" {
		return Checkout{}, fmt.Errorf("payment provider returned an invalid checkout")
	}
	if err := ledger.MutateAccount(ctx, s.accounts, accountID, "checkout-created:"+checkout.ID, func(value *account.Account) error {
		if value.Checkout.LastCheckoutID == checkout.ID {
			return checkoutAlreadyStored
		}
		value.Checkout.LastCheckoutID = checkout.ID
		return nil
	}); err != nil && !errors.Is(err, checkoutAlreadyStored) {
		return Checkout{}, err
	}
	return checkout, nil
}

func randomCheckoutKey() (string, error) {
	random := make([]byte, 24)
	if _, err := rand.Read(random); err != nil {
		return "", err
	}
	return "checkout_" + base64.RawURLEncoding.EncodeToString(random), nil
}
