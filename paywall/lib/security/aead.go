package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"

	"github.com/jcthi/nba-analytics/paywall/account"
)

func EncryptField(keyID string, key []byte, accountID, plaintext string) (*account.EncryptedField, error) {
	aead, err := newGCM(key)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	ciphertext := aead.Seal(nil, nonce, []byte(plaintext), []byte(accountID))
	return &account.EncryptedField{
		KeyID:      keyID,
		Nonce:      base64.RawStdEncoding.EncodeToString(nonce),
		Ciphertext: base64.RawStdEncoding.EncodeToString(ciphertext),
	}, nil
}

func DecryptField(keys map[string][]byte, accountID string, field *account.EncryptedField) (string, error) {
	if field == nil {
		return "", fmt.Errorf("encrypted field is missing")
	}
	key, ok := keys[field.KeyID]
	if !ok {
		return "", fmt.Errorf("unknown encryption key")
	}
	aead, err := newGCM(key)
	if err != nil {
		return "", err
	}
	nonce, err := base64.RawStdEncoding.DecodeString(field.Nonce)
	if err != nil || len(nonce) != aead.NonceSize() {
		return "", fmt.Errorf("invalid nonce")
	}
	ciphertext, err := base64.RawStdEncoding.DecodeString(field.Ciphertext)
	if err != nil {
		return "", fmt.Errorf("invalid ciphertext")
	}
	plaintext, err := aead.Open(nil, nonce, ciphertext, []byte(accountID))
	if err != nil {
		return "", fmt.Errorf("decrypt encrypted field: %w", err)
	}
	return string(plaintext), nil
}

func newGCM(key []byte) (cipher.AEAD, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("AES-256-GCM key must be exactly 32 bytes")
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}
