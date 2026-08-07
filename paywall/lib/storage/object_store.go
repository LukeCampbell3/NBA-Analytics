package storage

import "context"

type Object struct {
	Body []byte
	ETag string
}

type PutCondition struct {
	IfMatch     string
	IfNoneMatch bool
}

// ObjectStore is the narrow S3-compatible surface used by the state ledger.
// Production adapters must map failed preconditions to ErrConflict.
type ObjectStore interface {
	Get(ctx context.Context, key string) (Object, error)
	Put(ctx context.Context, key string, body []byte, condition PutCondition) (etag string, err error)
}
