package account

import "time"

const SchemaVersion = 1

type Status string

const (
	StatusPending   Status = "pending"
	StatusActive    Status = "active"
	StatusGrace     Status = "grace"
	StatusPastDue   Status = "past_due"
	StatusSuspended Status = "suspended"
	StatusCanceled  Status = "canceled"
	StatusBanned    Status = "banned"
	StatusDeleted   Status = "deleted"
)

type EncryptedField struct {
	KeyID      string `json:"key_id"`
	Nonce      string `json:"nonce"`
	Ciphertext string `json:"ciphertext"`
}

type Entitlement struct {
	Plan               string    `json:"plan"`
	Source             string    `json:"source"`
	ValidFrom          time.Time `json:"valid_from"`
	ValidUntil         time.Time `json:"valid_until"`
	ProviderVerifiedAt time.Time `json:"provider_verified_at"`
	ProviderUpdatedAt  time.Time `json:"provider_updated_at"`
}

type Payment struct {
	CustomerID     *EncryptedField `json:"customer_id_encrypted,omitempty"`
	SubscriptionID *EncryptedField `json:"subscription_id_encrypted,omitempty"`
}

type Checkout struct {
	LockUntil      *time.Time `json:"lock_until"`
	IdempotencyKey string     `json:"idempotency_key,omitempty"`
	LastCheckoutID string     `json:"last_checkout_id,omitempty"`
}

type Account struct {
	SchemaVersion int         `json:"schema_version"`
	Revision      uint64      `json:"revision"`
	AccountID     string      `json:"account_id"`
	DisplayName   string      `json:"display_name"`
	Status        Status      `json:"status"`
	Entitlement   Entitlement `json:"entitlement"`
	Payment       Payment     `json:"payment"`
	SessionEpoch  uint64      `json:"session_epoch"`
	Checkout      Checkout    `json:"checkout"`
	CreatedAt     time.Time   `json:"created_at"`
	UpdatedAt     time.Time   `json:"updated_at"`
}

type IdentityIndex struct {
	SchemaVersion int       `json:"schema_version"`
	AccountID     string    `json:"account_id"`
	CreatedAt     time.Time `json:"created_at"`
}

type HistoryRecord struct {
	SchemaVersion int       `json:"schema_version"`
	MutationID    string    `json:"mutation_id"`
	RecordedAt    time.Time `json:"recorded_at"`
	Account       Account   `json:"account"`
}
