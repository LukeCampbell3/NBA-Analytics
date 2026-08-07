package storage

import (
	"context"
	"fmt"
	"io"
	"mime"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type ContentObject struct {
	Body        []byte
	ETag        string
	ContentType string
}

type R2ContentStore struct {
	client    *s3.Client
	presigner *s3.PresignClient
	bucket    string
}

func NewR2ContentStore(ctx context.Context, config R2Config) (*R2ContentStore, error) {
	client, err := newR2S3Client(ctx, config)
	if err != nil {
		return nil, err
	}
	return &R2ContentStore{client: client, presigner: s3.NewPresignClient(client), bucket: config.Bucket}, nil
}

func (s *R2ContentStore) Get(ctx context.Context, key string, maximumBytes int64) (ContentObject, error) {
	if !safeR2Key(key) || maximumBytes <= 0 || maximumBytes > 1024*1024 {
		return ContentObject{}, fmt.Errorf("invalid content request")
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
		return ContentObject{}, fmt.Errorf("content object exceeds manifest size")
	}
	return ContentObject{Body: body, ETag: aws.ToString(result.ETag), ContentType: aws.ToString(result.ContentType)}, nil
}

func (s *R2ContentStore) PresignGET(ctx context.Context, key, downloadName string, lifetime time.Duration) (string, error) {
	if !safeR2Key(key) || downloadName == "" || lifetime < 30*time.Second || lifetime > 2*time.Minute {
		return "", fmt.Errorf("invalid presign request")
	}
	disposition := mime.FormatMediaType("attachment", map[string]string{"filename": downloadName})
	request, err := s.presigner.PresignGetObject(ctx, &s3.GetObjectInput{
		Bucket:                     &s.bucket,
		Key:                        &key,
		ResponseContentDisposition: &disposition,
	}, func(options *s3.PresignOptions) {
		options.Expires = lifetime
	})
	if err != nil {
		return "", err
	}
	return request.URL, nil
}
