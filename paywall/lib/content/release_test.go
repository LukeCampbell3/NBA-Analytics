package content

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/storage"
)

type memoryReleaseStore struct {
	objects     map[string]storage.ContentObject
	activations int
	revision    int
}

func newMemoryReleaseStore() *memoryReleaseStore {
	return &memoryReleaseStore{objects: make(map[string]storage.ContentObject)}
}

func (store *memoryReleaseStore) PutImmutable(_ context.Context, key string, body []byte, contentType string) error {
	if _, exists := store.objects[key]; exists {
		return storage.ErrConflict
	}
	store.revision++
	store.objects[key] = storage.ContentObject{Body: append([]byte(nil), body...), ContentType: contentType, ETag: fmt.Sprintf("etag-%d", store.revision)}
	return nil
}

func (store *memoryReleaseStore) Get(_ context.Context, key string, maximumBytes int64) (storage.ContentObject, error) {
	object, exists := store.objects[key]
	if !exists {
		return storage.ContentObject{}, storage.ErrNotFound
	}
	if int64(len(object.Body)) > maximumBytes {
		return storage.ContentObject{}, fmt.Errorf("too large")
	}
	return object, nil
}

func (store *memoryReleaseStore) Verify(_ context.Context, key string, size int64, expected string) error {
	object, exists := store.objects[key]
	digest := sha256.Sum256(object.Body)
	if !exists || int64(len(object.Body)) != size || hex.EncodeToString(digest[:]) != expected {
		return fmt.Errorf("verification failed")
	}
	return nil
}

func (store *memoryReleaseStore) Activate(_ context.Context, body []byte, previousETag string, create bool) error {
	current, exists := store.objects[releasePointerKey]
	if create {
		if exists {
			return storage.ErrConflict
		}
	} else if !exists || current.ETag != previousETag {
		return storage.ErrConflict
	}
	store.revision++
	store.activations++
	store.objects[releasePointerKey] = storage.ContentObject{
		Body: append([]byte(nil), body...), ContentType: "application/json", ETag: fmt.Sprintf("etag-%d", store.revision),
	}
	return nil
}

func TestDeployReleaseUploadsVerifiesAndAtomicallyActivates(t *testing.T) {
	store := newMemoryReleaseStore()
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	files := []SourceFile{
		{LogicalPath: "app/index.html", Body: []byte("<h1>member</h1>")},
		{LogicalPath: "downloads/archive.zip", Body: []byte("archive")},
	}
	deployment, err := DeployRelease(context.Background(), store, "2026-08-06-01", "individual", files, now)
	if err != nil {
		t.Fatal(err)
	}
	if deployment.Manifest.Objects["app/index.html"].DeliveryMode != DeliveryProxy ||
		deployment.Manifest.Objects["downloads/archive.zip"].DeliveryMode != DeliveryPresign ||
		store.activations != 1 {
		t.Fatalf("deployment = %#v, activations = %d", deployment, store.activations)
	}
	if _, err := DeployRelease(context.Background(), store, "2026-08-06-01", "individual", files, now.Add(time.Minute)); err != nil {
		t.Fatalf("idempotent retry: %v", err)
	}
	if store.activations != 1 {
		t.Fatalf("idempotent retry activated pointer %d times", store.activations)
	}
}

func TestRollbackReleaseConditionallyMovesPointer(t *testing.T) {
	store := newMemoryReleaseStore()
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	files := []SourceFile{{LogicalPath: "app/index.html", Body: []byte("version one")}}
	if _, err := DeployRelease(context.Background(), store, "release-1", "individual", files, now); err != nil {
		t.Fatal(err)
	}
	files[0].Body = []byte("version two")
	if _, err := DeployRelease(context.Background(), store, "release-2", "individual", files, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	pointer, err := RollbackRelease(context.Background(), store, "release-1", now.Add(2*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if pointer.ReleaseID != "release-1" || store.activations != 3 {
		t.Fatalf("rollback pointer = %#v, activations = %d", pointer, store.activations)
	}
}

func TestDeployReleaseRejectsUnsafeOrOversizedProxyFiles(t *testing.T) {
	store := newMemoryReleaseStore()
	now := time.Now().UTC()
	if _, err := DeployRelease(context.Background(), store, "../unsafe", "individual", []SourceFile{{LogicalPath: "app/index.html", Body: []byte("x")}}, now); err == nil {
		t.Fatal("unsafe release id was accepted")
	}
	if _, err := DeployRelease(context.Background(), store, "release-1", "individual", []SourceFile{{LogicalPath: "app/large.bin", Body: make([]byte, MaxProxyBytes+1)}}, now); err == nil {
		t.Fatal("oversized proxy file was accepted")
	}
}
