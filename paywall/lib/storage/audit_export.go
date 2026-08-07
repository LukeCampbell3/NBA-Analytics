package storage

import (
	"context"
	"fmt"
	"io"
	"sort"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

const (
	maximumAuditEventBytes = int64(64 * 1024)
	maximumAuditDayEvents  = 100000
)

// R2AuditExporter is an offline-only reader. Unlike the runtime ObjectStore,
// it can list exactly one validated audit date prefix.
type R2AuditExporter struct {
	client *s3.Client
	bucket string
}

func NewR2AuditExporter(ctx context.Context, config R2Config) (*R2AuditExporter, error) {
	client, err := newR2S3Client(ctx, config)
	if err != nil {
		return nil, err
	}
	return &R2AuditExporter{client: client, bucket: config.Bucket}, nil
}

func (exporter *R2AuditExporter) ReadDay(ctx context.Context, date string) ([][]byte, error) {
	day, err := time.Parse("2006-01-02", date)
	if err != nil || day.Format("2006-01-02") != date {
		return nil, fmt.Errorf("audit date must use YYYY-MM-DD")
	}
	prefix := "audit/" + day.Format("2006/01/02") + "/"
	paginator := s3.NewListObjectsV2Paginator(exporter.client, &s3.ListObjectsV2Input{
		Bucket: &exporter.bucket, Prefix: &prefix, MaxKeys: aws.Int32(1000),
	})
	var keys []string
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, object := range page.Contents {
			key := aws.ToString(object.Key)
			if len(key) <= len(prefix) || key[:len(prefix)] != prefix {
				return nil, fmt.Errorf("audit listing escaped requested prefix")
			}
			keys = append(keys, key)
			if len(keys) > maximumAuditDayEvents {
				return nil, fmt.Errorf("audit day exceeds export safety limit")
			}
		}
	}
	sort.Strings(keys)
	events := make([][]byte, 0, len(keys))
	for _, key := range keys {
		result, err := exporter.client.GetObject(ctx, &s3.GetObjectInput{Bucket: &exporter.bucket, Key: &key})
		if err != nil {
			return nil, mapS3Error(err)
		}
		body, readErr := io.ReadAll(io.LimitReader(result.Body, maximumAuditEventBytes+1))
		closeErr := result.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if int64(len(body)) > maximumAuditEventBytes {
			return nil, fmt.Errorf("audit event exceeds size limit")
		}
		events = append(events, body)
	}
	return events, nil
}
