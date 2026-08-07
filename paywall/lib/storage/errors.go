package storage

import "errors"

var (
	ErrNotFound                 = errors.New("object not found")
	ErrConflict                 = errors.New("conditional write conflict")
	ErrMalformed                = errors.New("stored object is malformed")
	ErrConflictRetriesExhausted = errors.New("conditional write retries exhausted")
)
