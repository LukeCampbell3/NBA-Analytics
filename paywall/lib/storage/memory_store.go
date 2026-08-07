package storage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

type memoryObject struct {
	body []byte
	etag string
}

// MemoryStore implements R2-style strong reads and conditional writes for tests.
// It deliberately has no listing method.
type MemoryStore struct {
	mu      sync.RWMutex
	objects map[string]memoryObject
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{objects: make(map[string]memoryObject)}
}

func (s *MemoryStore) Get(_ context.Context, key string) (Object, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	object, ok := s.objects[key]
	if !ok {
		return Object{}, ErrNotFound
	}
	return Object{Body: append([]byte(nil), object.body...), ETag: object.etag}, nil
}

func (s *MemoryStore) Put(_ context.Context, key string, body []byte, condition PutCondition) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, exists := s.objects[key]
	if condition.IfNoneMatch && exists {
		return "", ErrConflict
	}
	if condition.IfMatch != "" && (!exists || current.etag != condition.IfMatch) {
		return "", ErrConflict
	}
	hash := sha256.Sum256(body)
	etag := hex.EncodeToString(hash[:])
	s.objects[key] = memoryObject{body: append([]byte(nil), body...), etag: etag}
	return etag, nil
}

func (s *MemoryStore) CountPrefix(prefix string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	count := 0
	for key := range s.objects {
		if len(key) >= len(prefix) && key[:len(prefix)] == prefix {
			count++
		}
	}
	return count
}
