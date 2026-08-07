package content

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"testing"
)

func TestManifestRejectsTraversalAndOversizedProxy(t *testing.T) {
	digest := strings.Repeat("a", 64)
	base := `{"release_id":"release-1","created_at":"2026-08-06T21:00:00Z","objects":{"%s":{"key":"%s","sha256":"%s","content_type":"text/html","size":%d,"delivery_mode":"proxy","required_plan":"individual"}}}`
	for _, test := range []struct {
		logical string
		key     string
		size    int
	}{
		{"../secret", "releases/release-1/secret", 10},
		{"app/index.html", "releases/release-1/../secret", 10},
		{"app/index.html", "releases/release-1/app/index.html", MaxProxyBytes + 1},
	} {
		body := fmt.Sprintf(base, test.logical, test.key, digest, test.size)
		if _, err := ParseManifest([]byte(body)); err == nil {
			t.Fatalf("unsafe manifest accepted: %s", body)
		}
	}
}

func TestManifestResolvesOnlyExactKnownPlan(t *testing.T) {
	contentBody := []byte("0123456789")
	digest := sha256.Sum256(contentBody)
	body := fmt.Sprintf(`{"release_id":"release-1","created_at":"2026-08-06T21:00:00Z","objects":{"app/index.html":{"key":"releases/release-1/app/index.html","sha256":"%s","content_type":"text/html","size":10,"delivery_mode":"proxy","required_plan":"individual"}}}`, hex.EncodeToString(digest[:]))
	manifest, err := ParseManifest([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	object, ok := manifest.Resolve("app/index.html", "individual")
	if !ok {
		t.Fatal("known content was not resolved")
	}
	if err := object.VerifyBody(contentBody); err != nil {
		t.Fatal(err)
	}
	if err := object.VerifyBody([]byte("tampered!!")); err == nil {
		t.Fatal("tampered content passed manifest verification")
	}
	for _, logical := range []string{"../app/index.html", "/app/index.html", "app\\index.html", "missing"} {
		if _, ok := manifest.Resolve(logical, "individual"); ok {
			t.Fatalf("unsafe or unknown path %q resolved", logical)
		}
	}
	if _, ok := manifest.Resolve("app/index.html", "team"); ok {
		t.Fatal("content resolved for the wrong plan")
	}
}
