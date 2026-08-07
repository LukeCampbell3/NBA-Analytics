package content

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"path"
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/jcthi/nba-analytics/paywall/storage"
)

const releasePointerKey = "system/current-content-release.json"

var (
	releaseIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	planPattern      = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,63}$`)
)

type ReleaseStore interface {
	PutImmutable(context.Context, string, []byte, string) error
	Get(context.Context, string, int64) (storage.ContentObject, error)
	Verify(context.Context, string, int64, string) error
	Activate(context.Context, []byte, string, bool) error
}

type SourceFile struct {
	LogicalPath string
	Body        []byte
}

type Deployment struct {
	Pointer  ReleasePointer
	Manifest Manifest
}

func DeployRelease(ctx context.Context, store ReleaseStore, releaseID, plan string, files []SourceFile, now time.Time) (Deployment, error) {
	if store == nil || !validReleaseID(releaseID) || !planPattern.MatchString(plan) || len(files) == 0 || now.IsZero() {
		return Deployment{}, fmt.Errorf("invalid release deployment")
	}
	manifestKey := "releases/" + releaseID + "/manifest.json"
	var existingManifest *Manifest
	var existingManifestBody []byte
	createdAt := now.UTC()
	storedManifest, err := store.Get(ctx, manifestKey, 1024*1024)
	if err == nil {
		parsed, parseErr := ParseManifest(storedManifest.Body)
		if parseErr != nil || parsed.ReleaseID != releaseID {
			return Deployment{}, fmt.Errorf("existing release manifest is invalid")
		}
		existingManifest = &parsed
		existingManifestBody = storedManifest.Body
		createdAt = parsed.CreatedAt
	} else if !errors.Is(err, storage.ErrNotFound) {
		return Deployment{}, err
	}
	manifest := Manifest{ReleaseID: releaseID, CreatedAt: createdAt, Objects: make(map[string]Object, len(files))}
	for _, file := range files {
		object, err := releaseObject(releaseID, plan, file)
		if err != nil {
			return Deployment{}, err
		}
		if _, exists := manifest.Objects[file.LogicalPath]; exists {
			return Deployment{}, fmt.Errorf("duplicate logical path %q", file.LogicalPath)
		}
		if err := putAndVerify(ctx, store, object.Key, file.Body, object.ContentType, object.SHA256); err != nil {
			return Deployment{}, err
		}
		manifest.Objects[file.LogicalPath] = object
	}
	manifestBody, err := marshalReleaseJSON(manifest)
	if err != nil {
		return Deployment{}, err
	}
	if existingManifest != nil {
		if !reflect.DeepEqual(*existingManifest, manifest) {
			return Deployment{}, fmt.Errorf("release id already refers to different content")
		}
		manifestBody = existingManifestBody
	}
	manifestDigest := sha256.Sum256(manifestBody)
	manifestSHA := hex.EncodeToString(manifestDigest[:])
	if err := putAndVerify(ctx, store, manifestKey, manifestBody, "application/json", manifestSHA); err != nil {
		return Deployment{}, err
	}
	pointer := ReleasePointer{
		SchemaVersion: 1, ReleaseID: releaseID, ManifestSHA256: manifestSHA, ActivatedAt: now.UTC(),
	}
	if err := activatePointer(ctx, store, pointer); err != nil {
		return Deployment{}, err
	}
	return Deployment{Pointer: pointer, Manifest: manifest}, nil
}

func RollbackRelease(ctx context.Context, store ReleaseStore, releaseID string, now time.Time) (ReleasePointer, error) {
	if store == nil || !validReleaseID(releaseID) || now.IsZero() {
		return ReleasePointer{}, fmt.Errorf("invalid release rollback")
	}
	manifestKey := "releases/" + releaseID + "/manifest.json"
	stored, err := store.Get(ctx, manifestKey, 1024*1024)
	if err != nil {
		return ReleasePointer{}, err
	}
	manifest, err := ParseManifest(stored.Body)
	if err != nil || manifest.ReleaseID != releaseID {
		return ReleasePointer{}, fmt.Errorf("target release manifest is invalid")
	}
	digest := sha256.Sum256(stored.Body)
	pointer := ReleasePointer{
		SchemaVersion: 1, ReleaseID: releaseID, ManifestSHA256: hex.EncodeToString(digest[:]), ActivatedAt: now.UTC(),
	}
	if err := activatePointer(ctx, store, pointer); err != nil {
		return ReleasePointer{}, err
	}
	return pointer, nil
}

func releaseObject(releaseID, plan string, file SourceFile) (Object, error) {
	if !validLogicalPath(file.LogicalPath) || len(file.Body) == 0 {
		return Object{}, fmt.Errorf("invalid release file %q", file.LogicalPath)
	}
	mode := DeliveryProxy
	switch {
	case strings.HasPrefix(file.LogicalPath, "app/"), strings.HasPrefix(file.LogicalPath, "data/"):
		if len(file.Body) > MaxProxyBytes {
			return Object{}, fmt.Errorf("proxy file exceeds safe Function response size %q", file.LogicalPath)
		}
	case strings.HasPrefix(file.LogicalPath, "downloads/"):
		mode = DeliveryPresign
	default:
		return Object{}, fmt.Errorf("unsupported private-content path %q", file.LogicalPath)
	}
	digest := sha256.Sum256(file.Body)
	contentType := mime.TypeByExtension(path.Ext(file.LogicalPath))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	return Object{
		Key:    "releases/" + releaseID + "/" + file.LogicalPath,
		SHA256: hex.EncodeToString(digest[:]), ContentType: contentType, Size: int64(len(file.Body)),
		DeliveryMode: mode, RequiredPlan: plan,
	}, nil
}

func putAndVerify(ctx context.Context, store ReleaseStore, key string, body []byte, contentType, digest string) error {
	err := store.PutImmutable(ctx, key, body, contentType)
	if err != nil && !errors.Is(err, storage.ErrConflict) {
		return err
	}
	if err := store.Verify(ctx, key, int64(len(body)), digest); err != nil {
		return err
	}
	return nil
}

func activatePointer(ctx context.Context, store ReleaseStore, pointer ReleasePointer) error {
	body, err := marshalReleaseJSON(pointer)
	if err != nil {
		return err
	}
	current, err := store.Get(ctx, releasePointerKey, 16*1024)
	switch {
	case errors.Is(err, storage.ErrNotFound):
		return store.Activate(ctx, body, "", true)
	case err != nil:
		return err
	default:
		var existing ReleasePointer
		if err := decodeStrictJSON(current.Body, &existing); err != nil || !validReleasePointer(existing) {
			return storage.ErrMalformed
		}
		if existing.ReleaseID == pointer.ReleaseID && strings.EqualFold(existing.ManifestSHA256, pointer.ManifestSHA256) {
			return nil
		}
		return store.Activate(ctx, body, current.ETag, false)
	}
}

func marshalReleaseJSON(value any) ([]byte, error) {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(body, '\n'), nil
}

func validReleaseID(value string) bool {
	return releaseIDPattern.MatchString(value) && value != "." && value != ".."
}
