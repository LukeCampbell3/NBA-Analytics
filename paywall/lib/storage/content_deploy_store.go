package storage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// R2ContentDeploymentStore is intentionally separate from the runtime content
// store. Its credentials may create immutable release objects and conditionally
// replace the release pointer, but are never placed in a Function environment.
type R2ContentDeploymentStore struct {
	client *s3.Client
	bucket string
}

func NewR2ContentDeploymentStore(ctx context.Context, config R2Config) (*R2ContentDeploymentStore, error) {
	client, err := newR2S3Client(ctx, config)
	if err != nil {
		return nil, err
	}
	return &R2ContentDeploymentStore{client: client, bucket: config.Bucket}, nil
}

func (s *R2ContentDeploymentStore) PutImmutable(ctx context.Context, key string, body []byte, contentType string) error {
	if !safeR2Key(key) || len(body) == 0 || contentType == "" {
		return fmt.Errorf("invalid immutable content write")
	}
	digest := sha256.Sum256(body)
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      &s.bucket,
		Key:         &key,
		Body:        bytes.NewReader(body),
		ContentType: &contentType,
		IfNoneMatch: aws.String("*"),
		Metadata:    map[string]string{"sha256": hex.EncodeToString(digest[:])},
	})
	return mapS3Error(err)
}

func (s *R2ContentDeploymentStore) Get(ctx context.Context, key string, maximumBytes int64) (ContentObject, error) {
	if !safeR2Key(key) || maximumBytes <= 0 {
		return ContentObject{}, fmt.Errorf("invalid deployment content read")
	}
	result, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil {
		return ContentObject{}, mapS3Error(err)
	}
	defer result.Body.Close()
	body, err := io.ReadAll(io.LimitReader(result.Body, maximumBytes+1))
	if err != nil {
		return ContentObject{}, err
	}
	if int64(len(body)) > maximumBytes {
		return ContentObject{}, fmt.Errorf("deployment content object exceeds expected size")
	}
	return ContentObject{Body: body, ETag: aws.ToString(result.ETag), ContentType: aws.ToString(result.ContentType)}, nil
}

func (s *R2ContentDeploymentStore) Verify(ctx context.Context, key string, expectedSize int64, expectedSHA256 string) error {
	if !safeR2Key(key) || expectedSize <= 0 || len(expectedSHA256) != sha256.Size*2 {
		return fmt.Errorf("invalid content verification")
	}
	result, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil {
		return mapS3Error(err)
	}
	defer result.Body.Close()
	hasher := sha256.New()
	written, err := io.Copy(hasher, io.LimitReader(result.Body, expectedSize+1))
	if err != nil {
		return err
	}
	if written != expectedSize || !strings.EqualFold(hex.EncodeToString(hasher.Sum(nil)), expectedSHA256) {
		return fmt.Errorf("uploaded content verification failed for %q", key)
	}
	return nil
}

func (s *R2ContentDeploymentStore) Activate(ctx context.Context, body []byte, previousETag string, create bool) error {
	if len(body) == 0 || (create && previousETag != "") || (!create && previousETag == "") {
		return fmt.Errorf("invalid release activation")
	}
	input := &s3.PutObjectInput{
		Bucket:      &s.bucket,
		Key:         aws.String("system/current-content-release.json"),
		Body:        bytes.NewReader(body),
		ContentType: aws.String("application/json"),
	}
	if create {
		input.IfNoneMatch = aws.String("*")
	} else {
		input.IfMatch = &previousETag
	}
	_, err := s.client.PutObject(ctx, input)
	return mapS3Error(err)
}
