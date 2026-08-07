package content

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jcthi/nba-analytics/paywall/auth"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type fakeContentBackend struct {
	objects map[string]storage.ContentObject
	presign string
	reads   map[string]int
}

func (backend *fakeContentBackend) Get(_ context.Context, key string, _ int64) (storage.ContentObject, error) {
	backend.reads[key]++
	object, ok := backend.objects[key]
	if !ok {
		return storage.ContentObject{}, storage.ErrNotFound
	}
	return object, nil
}

func (backend *fakeContentBackend) PresignGET(context.Context, string, string, time.Duration) (string, error) {
	if backend.presign == "" {
		return "", errors.New("presign failed")
	}
	return backend.presign, nil
}

type fakeAuthorizer struct {
	authorization auth.Authorization
	err           error
	freshCalls    []bool
}

func (authorizer *fakeAuthorizer) Authorize(_ context.Context, _ string, fresh bool) (auth.Authorization, error) {
	authorizer.freshCalls = append(authorizer.freshCalls, fresh)
	return authorizer.authorization, authorizer.err
}

func contentFixture(t *testing.T) (*Gateway, *fakeContentBackend, *fakeAuthorizer) {
	t.Helper()
	now := time.Date(2026, 8, 6, 21, 0, 0, 0, time.UTC)
	html := []byte("<h1>member</h1>")
	htmlDigest := sha256.Sum256(html)
	download := []byte("archive")
	downloadDigest := sha256.Sum256(download)
	manifest := Manifest{
		ReleaseID: "release-1", CreatedAt: now,
		Objects: map[string]Object{
			"app/index.html": {
				Key: "releases/release-1/app/index.html", SHA256: hex.EncodeToString(htmlDigest[:]),
				ContentType: "text/html", Size: int64(len(html)), DeliveryMode: DeliveryProxy, RequiredPlan: "individual",
			},
			"app/nba/predictions/index.html": {
				Key: "releases/release-1/app/nba/predictions/index.html", SHA256: hex.EncodeToString(htmlDigest[:]),
				ContentType: "text/html", Size: int64(len(html)), DeliveryMode: DeliveryProxy, RequiredPlan: "individual",
			},
			"downloads/archive.zip": {
				Key: "releases/release-1/downloads/archive.zip", SHA256: hex.EncodeToString(downloadDigest[:]),
				ContentType: "application/zip", Size: int64(len(download)), DeliveryMode: DeliveryPresign, RequiredPlan: "individual",
			},
		},
	}
	manifestBody, _ := json.Marshal(manifest)
	manifestDigest := sha256.Sum256(manifestBody)
	pointerBody, _ := json.Marshal(ReleasePointer{
		SchemaVersion: 1, ReleaseID: "release-1", ManifestSHA256: hex.EncodeToString(manifestDigest[:]), ActivatedAt: now,
	})
	backend := &fakeContentBackend{
		objects: map[string]storage.ContentObject{
			"system/current-content-release.json":               {Body: pointerBody, ContentType: "application/json"},
			"releases/release-1/manifest.json":                  {Body: manifestBody, ContentType: "application/json"},
			"releases/release-1/app/index.html":                 {Body: html, ContentType: "text/html"},
			"releases/release-1/app/nba/predictions/index.html": {Body: html, ContentType: "text/html"},
		},
		presign: "https://signed.example/archive", reads: make(map[string]int),
	}
	authorizer := &fakeAuthorizer{authorization: auth.Authorization{
		AccountID: "acc_aaaaaaaaaaaaaaaaaaaaaaaaaa", Plan: "individual",
		Claims: security.SessionClaims{Expiry: now.Add(7 * 24 * time.Hour).Unix()},
	}}
	gateway, err := NewGateway(backend, authorizer, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	gateway.now = func() time.Time { return now }
	return gateway, backend, authorizer
}

func TestContentGatewayProxiesVerifiedHTMLAndCachesManifest(t *testing.T) {
	gateway, backend, authorizer := contentFixture(t)
	first := gateway.Serve(context.Background(), "session", "/app/")
	second := gateway.Serve(context.Background(), "session", "/app/")
	if first.StatusCode != "200" || first.Body != "<h1>member</h1>" || second.StatusCode != "200" {
		t.Fatalf("responses = %#v, %#v", first, second)
	}
	if backend.reads["system/current-content-release.json"] != 1 {
		t.Fatalf("pointer reads = %d, want 1", backend.reads["system/current-content-release.json"])
	}
	if len(authorizer.freshCalls) != 2 || authorizer.freshCalls[0] || authorizer.freshCalls[1] {
		t.Fatalf("fresh authorization calls = %v", authorizer.freshCalls)
	}
}

func TestContentGatewayResolvesProtectedDirectoryIndex(t *testing.T) {
	gateway, _, _ := contentFixture(t)
	response := gateway.Serve(context.Background(), "session", "/app/nba/predictions/")
	if response.StatusCode != "200" || response.Body != "<h1>member</h1>" {
		t.Fatalf("directory response = %#v", response)
	}
}

func TestDownloadRequiresFreshAuthorizationAndRedirects(t *testing.T) {
	gateway, _, authorizer := contentFixture(t)
	response := gateway.Serve(context.Background(), "session", "/downloads/archive.zip")
	if response.StatusCode != "302" || !strings.HasPrefix(response.Headers["Location"], "https://signed.example/") {
		t.Fatalf("download response = %#v", response)
	}
	if len(authorizer.freshCalls) != 1 || !authorizer.freshCalls[0] {
		t.Fatalf("fresh authorization calls = %v", authorizer.freshCalls)
	}
}

func TestContentGatewayFailsClosedOnAuthorizationAndTraversal(t *testing.T) {
	gateway, _, authorizer := contentFixture(t)
	authorizer.err = auth.ErrAccessDenied
	if response := gateway.Serve(context.Background(), "session", "/app/"); response.StatusCode != "403" {
		t.Fatalf("denied response = %#v", response)
	}
	authorizer.err = nil
	if response := gateway.Serve(context.Background(), "session", "/app/../manifest.json"); response.StatusCode != "404" {
		t.Fatalf("traversal response = %#v", response)
	}
}
