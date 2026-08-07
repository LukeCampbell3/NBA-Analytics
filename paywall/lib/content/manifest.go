package content

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path"
	"strings"
	"time"
)

// Binary Function responses require base64 expansion, so keep proxy objects
// comfortably below the platform's 1 MB response ceiling.
const MaxProxyBytes = 700 * 1024

type DeliveryMode string

const (
	DeliveryProxy   DeliveryMode = "proxy"
	DeliveryPresign DeliveryMode = "presigned"
)

type Object struct {
	Key          string       `json:"key"`
	SHA256       string       `json:"sha256"`
	ContentType  string       `json:"content_type"`
	Size         int64        `json:"size"`
	DeliveryMode DeliveryMode `json:"delivery_mode"`
	RequiredPlan string       `json:"required_plan"`
}

type Manifest struct {
	ReleaseID string            `json:"release_id"`
	CreatedAt time.Time         `json:"created_at"`
	Objects   map[string]Object `json:"objects"`
}

func ParseManifest(body []byte) (Manifest, error) {
	var manifest Manifest
	if err := decodeStrictJSON(body, &manifest); err != nil {
		return Manifest{}, err
	}
	if manifest.ReleaseID == "" || manifest.CreatedAt.IsZero() || len(manifest.Objects) == 0 {
		return Manifest{}, fmt.Errorf("manifest metadata is incomplete")
	}
	prefix := "releases/" + manifest.ReleaseID + "/"
	for logicalPath, object := range manifest.Objects {
		if !validLogicalPath(logicalPath) || !strings.HasPrefix(object.Key, prefix) || !validObjectKey(object.Key) {
			return Manifest{}, fmt.Errorf("unsafe content mapping %q", logicalPath)
		}
		if object.ContentType == "" || object.Size <= 0 || object.RequiredPlan == "" {
			return Manifest{}, fmt.Errorf("incomplete content mapping %q", logicalPath)
		}
		digest, err := hex.DecodeString(object.SHA256)
		if err != nil || len(digest) != 32 {
			return Manifest{}, fmt.Errorf("invalid content digest %q", logicalPath)
		}
		switch object.DeliveryMode {
		case DeliveryProxy:
			if object.Size > MaxProxyBytes {
				return Manifest{}, fmt.Errorf("proxy object exceeds safe response size %q", logicalPath)
			}
		case DeliveryPresign:
		default:
			return Manifest{}, fmt.Errorf("invalid delivery mode %q", logicalPath)
		}
	}
	return manifest, nil
}

func decodeStrictJSON(body []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("trailing JSON data")
	}
	return nil
}

func (m Manifest) Resolve(logicalPath string, plan string) (Object, bool) {
	if !validLogicalPath(logicalPath) {
		return Object{}, false
	}
	object, ok := m.Objects[logicalPath]
	if !ok || object.RequiredPlan != plan {
		return Object{}, false
	}
	return object, true
}

func (object Object) VerifyBody(body []byte) error {
	if int64(len(body)) != object.Size {
		return fmt.Errorf("content size does not match manifest")
	}
	digest := sha256.Sum256(body)
	if !strings.EqualFold(hex.EncodeToString(digest[:]), object.SHA256) {
		return fmt.Errorf("content digest does not match manifest")
	}
	return nil
}

func validLogicalPath(value string) bool {
	return value != "" && len(value) <= 512 && !strings.HasPrefix(value, "/") &&
		!strings.Contains(value, "\\") && !strings.ContainsAny(value, "?#") &&
		path.Clean(value) == value && value != "." && !strings.HasPrefix(value, "../")
}

func validObjectKey(value string) bool {
	return value != "" && !strings.HasPrefix(value, "/") && !strings.Contains(value, "\\") &&
		path.Clean(value) == value && !strings.HasPrefix(value, "../")
}
