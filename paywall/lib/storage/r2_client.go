package storage

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go"
)

const maxStateObjectBytes = 2 * 1024 * 1024

type R2Config struct {
	Endpoint        string
	AccessKeyID     string
	SecretAccessKey string
	Bucket          string
}

type R2ObjectStore struct {
	client *s3.Client
	bucket string
}

func NewR2ObjectStore(ctx context.Context, config R2Config) (*R2ObjectStore, error) {
	client, err := newR2S3Client(ctx, config)
	if err != nil {
		return nil, err
	}
	return &R2ObjectStore{client: client, bucket: config.Bucket}, nil
}

func (s *R2ObjectStore) Get(ctx context.Context, key string) (Object, error) {
	if !safeR2Key(key) {
		return Object{}, fmt.Errorf("invalid object key")
	}
	result, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil {
		return Object{}, mapS3Error(err)
	}
	defer result.Body.Close()
	body, err := io.ReadAll(io.LimitReader(result.Body, maxStateObjectBytes+1))
	if err != nil {
		return Object{}, err
	}
	if len(body) > maxStateObjectBytes {
		return Object{}, fmt.Errorf("%w: state object exceeds size limit", ErrMalformed)
	}
	return Object{Body: body, ETag: aws.ToString(result.ETag)}, nil
}

func (s *R2ObjectStore) Put(ctx context.Context, key string, body []byte, condition PutCondition) (string, error) {
	if !safeR2Key(key) || len(body) > maxStateObjectBytes || (condition.IfMatch != "" && condition.IfNoneMatch) {
		return "", fmt.Errorf("invalid object write")
	}
	input := &s3.PutObjectInput{
		Bucket:      &s.bucket,
		Key:         &key,
		Body:        bytes.NewReader(body),
		ContentType: aws.String("application/json"),
	}
	if condition.IfMatch != "" {
		input.IfMatch = &condition.IfMatch
	}
	if condition.IfNoneMatch {
		input.IfNoneMatch = aws.String("*")
	}
	result, err := s.client.PutObject(ctx, input)
	if err != nil {
		return "", mapS3Error(err)
	}
	return aws.ToString(result.ETag), nil
}

func newR2S3Client(ctx context.Context, config R2Config) (*s3.Client, error) {
	endpoint, err := url.Parse(config.Endpoint)
	if err != nil || endpoint.Scheme != "https" || endpoint.Host == "" || endpoint.Path != "" ||
		config.AccessKeyID == "" || config.SecretAccessKey == "" || config.Bucket == "" {
		return nil, fmt.Errorf("invalid R2 configuration")
	}
	awsConfig, err := awsconfig.LoadDefaultConfig(
		ctx,
		awsconfig.WithRegion("auto"),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
			config.AccessKeyID,
			config.SecretAccessKey,
			"",
		)),
	)
	if err != nil {
		return nil, err
	}
	return s3.NewFromConfig(awsConfig, func(options *s3.Options) {
		options.BaseEndpoint = aws.String(config.Endpoint)
		options.UsePathStyle = true
	}), nil
}

func mapS3Error(err error) error {
	var apiError smithy.APIError
	if errors.As(err, &apiError) {
		switch apiError.ErrorCode() {
		case "NoSuchKey", "NotFound", "404":
			return ErrNotFound
		case "PreconditionFailed", "ConditionalRequestConflict", "412", "409":
			return ErrConflict
		}
	}
	return err
}

func safeR2Key(key string) bool {
	return key != "" && len(key) <= 1024 && !strings.HasPrefix(key, "/") &&
		!strings.Contains(key, "\\") && !strings.ContainsRune(key, '\x00')
}
