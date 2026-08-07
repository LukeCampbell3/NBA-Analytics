package content

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"mime"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/jcthi/nba-analytics/paywall/auth"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
	"github.com/jcthi/nba-analytics/paywall/transport"
)

type ContentBackend interface {
	Get(context.Context, string, int64) (storage.ContentObject, error)
	PresignGET(context.Context, string, string, time.Duration) (string, error)
}

type SessionAuthorizer interface {
	Authorize(context.Context, string, bool) (auth.Authorization, error)
}

type ReleasePointer struct {
	SchemaVersion  int       `json:"schema_version"`
	ReleaseID      string    `json:"release_id"`
	ManifestSHA256 string    `json:"manifest_sha256"`
	ActivatedAt    time.Time `json:"activated_at"`
}

type Gateway struct {
	content    ContentBackend
	authorizer SessionAuthorizer
	now        func() time.Time
	cacheTTL   time.Duration

	cacheMu        sync.RWMutex
	cachedAt       time.Time
	cachedPointer  ReleasePointer
	cachedManifest Manifest
}

func NewGateway(content ContentBackend, authorizer SessionAuthorizer, cacheTTL time.Duration) (*Gateway, error) {
	if content == nil || authorizer == nil || cacheTTL <= 0 || cacheTTL > 5*time.Minute {
		return nil, errors.New("invalid content gateway configuration")
	}
	return &Gateway{content: content, authorizer: authorizer, now: time.Now, cacheTTL: cacheTTL}, nil
}

func (g *Gateway) Serve(ctx context.Context, sessionToken, requestPath string) transport.Response {
	logicalPath, fresh, ok := logicalPathForRequest(requestPath)
	if !ok {
		return transport.Error(http.StatusNotFound, "not_found")
	}
	authorization, err := g.authorizer.Authorize(ctx, sessionToken, fresh)
	if err != nil {
		return transport.Error(http.StatusForbidden, "access_denied")
	}
	manifest, err := g.currentManifest(ctx)
	if err != nil {
		return transport.Error(http.StatusServiceUnavailable, "content_unavailable")
	}
	object, ok := manifest.Resolve(logicalPath, authorization.Plan)
	if !ok {
		return transport.Error(http.StatusNotFound, "not_found")
	}
	var sessionCookie *http.Cookie
	if authorization.RefreshedToken != "" {
		remaining := time.Unix(authorization.Claims.Expiry, 0).Sub(g.now())
		sessionCookie = security.SessionCookie(authorization.RefreshedToken, remaining)
	}
	if object.DeliveryMode == DeliveryPresign {
		if !fresh {
			return transport.Error(http.StatusInternalServerError, "content_policy_error")
		}
		destination, err := g.content.PresignGET(ctx, object.Key, path.Base(logicalPath), time.Minute)
		if err != nil {
			return transport.Error(http.StatusServiceUnavailable, "download_unavailable")
		}
		return transport.Redirect(destination, sessionCookie)
	}
	stored, err := g.content.Get(ctx, object.Key, object.Size)
	if errors.Is(err, storage.ErrNotFound) {
		return transport.Error(http.StatusNotFound, "not_found")
	}
	if err != nil || object.VerifyBody(stored.Body) != nil {
		return transport.Error(http.StatusServiceUnavailable, "content_unavailable")
	}
	return contentResponse(stored.Body, object.ContentType, sessionCookie)
}

func (g *Gateway) currentManifest(ctx context.Context) (Manifest, error) {
	now := g.now().UTC()
	g.cacheMu.RLock()
	if !g.cachedAt.IsZero() && now.Sub(g.cachedAt) < g.cacheTTL {
		manifest := g.cachedManifest
		g.cacheMu.RUnlock()
		return manifest, nil
	}
	g.cacheMu.RUnlock()

	g.cacheMu.Lock()
	defer g.cacheMu.Unlock()
	if !g.cachedAt.IsZero() && now.Sub(g.cachedAt) < g.cacheTTL {
		return g.cachedManifest, nil
	}
	pointerObject, err := g.content.Get(ctx, "system/current-content-release.json", 16*1024)
	if err != nil {
		return Manifest{}, err
	}
	var pointer ReleasePointer
	if err := decodeStrictJSON(pointerObject.Body, &pointer); err != nil || !validReleasePointer(pointer) {
		return Manifest{}, storage.ErrMalformed
	}
	manifestKey := "releases/" + pointer.ReleaseID + "/manifest.json"
	storedManifest, err := g.content.Get(ctx, manifestKey, 1024*1024)
	if err != nil {
		return Manifest{}, err
	}
	digest := sha256.Sum256(storedManifest.Body)
	if !strings.EqualFold(hex.EncodeToString(digest[:]), pointer.ManifestSHA256) {
		return Manifest{}, storage.ErrMalformed
	}
	manifest, err := ParseManifest(storedManifest.Body)
	if err != nil || manifest.ReleaseID != pointer.ReleaseID {
		return Manifest{}, storage.ErrMalformed
	}
	g.cachedAt = now
	g.cachedPointer = pointer
	g.cachedManifest = manifest
	return manifest, nil
}

func validReleasePointer(pointer ReleasePointer) bool {
	if pointer.SchemaVersion != 1 || !validReleaseID(pointer.ReleaseID) || pointer.ActivatedAt.IsZero() {
		return false
	}
	digest, err := hex.DecodeString(pointer.ManifestSHA256)
	return err == nil && len(digest) == sha256.Size
}

func logicalPathForRequest(requestPath string) (string, bool, bool) {
	switch {
	case requestPath == "/app" || requestPath == "/app/":
		return "app/index.html", false, true
	case strings.HasPrefix(requestPath, "/app/"):
		logicalPath := strings.TrimPrefix(requestPath, "/")
		if strings.HasSuffix(logicalPath, "/") {
			logicalPath += "index.html"
		}
		return logicalPath, false, true
	case strings.HasPrefix(requestPath, "/api/content/"):
		return "data/" + strings.TrimPrefix(requestPath, "/api/content/"), false, true
	case strings.HasPrefix(requestPath, "/downloads/"):
		return "downloads/" + strings.TrimPrefix(requestPath, "/downloads/"), true, true
	default:
		return "", false, false
	}
}

func contentResponse(body []byte, contentType string, cookie *http.Cookie) transport.Response {
	mediaType, _, _ := mime.ParseMediaType(contentType)
	textual := strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" ||
		mediaType == "application/javascript" || mediaType == "application/xml" || mediaType == "image/svg+xml"
	encodedBody := string(body)
	if !textual {
		encodedBody = base64.StdEncoding.EncodeToString(body)
	}
	response := transport.Secure(transport.Response{
		Body: encodedBody, StatusCode: "200",
		Headers: map[string]string{"Content-Type": contentType, "Cache-Control": "private, no-store"},
	})
	if cookie != nil {
		response.Headers["Set-Cookie"] = cookie.String()
	}
	return response
}
