//go:build integration

package integration

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/jcthi/nba-analytics/paywall/ledger"
	"github.com/jcthi/nba-analytics/paywall/security"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

type testR2 struct {
	config storage.R2Config
	client *s3.Client
}

func loadTestR2(t *testing.T, prefix string) testR2 {
	t.Helper()
	config := storage.R2Config{
		Endpoint: os.Getenv(prefix + "_ENDPOINT"), AccessKeyID: os.Getenv(prefix + "_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv(prefix + "_SECRET_ACCESS_KEY"), Bucket: os.Getenv(prefix + "_BUCKET"),
	}
	if config.Endpoint == "" || config.AccessKeyID == "" || config.SecretAccessKey == "" || config.Bucket == "" {
		t.Skipf("%s credentials are not configured", prefix)
	}
	lowerBucket := strings.ToLower(config.Bucket)
	if !strings.Contains(lowerBucket, "test") && !strings.Contains(lowerBucket, "staging") {
		t.Fatalf("refusing integration writes to bucket %q: name must contain test or staging", config.Bucket)
	}
	awsConfig, err := awsconfig.LoadDefaultConfig(context.Background(), awsconfig.WithRegion("auto"),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(config.AccessKeyID, config.SecretAccessKey, "")))
	if err != nil {
		t.Fatal(err)
	}
	client := s3.NewFromConfig(awsConfig, func(options *s3.Options) {
		options.BaseEndpoint = aws.String(config.Endpoint)
		options.UsePathStyle = true
	})
	return testR2{config: config, client: client}
}

func (target testR2) cleanup(t *testing.T, keys ...string) {
	t.Helper()
	for _, key := range keys {
		_, err := target.client.DeleteObject(context.Background(), &s3.DeleteObjectInput{
			Bucket: aws.String(target.config.Bucket), Key: aws.String(key),
		})
		if err != nil {
			t.Errorf("cleanup %q: %v", key, err)
		}
	}
}

func TestR2ConditionalWritesAndConcurrentIdentityCreation(t *testing.T) {
	target := loadTestR2(t, "R2_TEST_STATE")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	objects, err := storage.NewR2ObjectStore(ctx, target.config)
	if err != nil {
		t.Fatal(err)
	}
	runID := fmt.Sprintf("%d", time.Now().UnixNano())
	casKey := "integration/" + runID + "/cas.json"
	t.Cleanup(func() { target.cleanup(t, casKey) })
	etag, err := objects.Put(ctx, casKey, []byte(`{"revision":1}`), storage.PutCondition{IfNoneMatch: true})
	if err != nil || etag == "" {
		t.Fatalf("initial conditional put: etag=%q err=%v", etag, err)
	}
	if _, err := objects.Put(ctx, casKey, []byte(`{"revision":2}`), storage.PutCondition{IfNoneMatch: true}); !errors.Is(err, storage.ErrConflict) {
		t.Fatalf("duplicate create should conflict, got %v", err)
	}
	newETag, err := objects.Put(ctx, casKey, []byte(`{"revision":2}`), storage.PutCondition{IfMatch: etag})
	if err != nil || newETag == "" {
		t.Fatalf("CAS update: etag=%q err=%v", newETag, err)
	}
	if _, err := objects.Put(ctx, casKey, []byte(`{"revision":3}`), storage.PutCondition{IfMatch: etag}); !errors.Is(err, storage.ErrConflict) {
		t.Fatalf("stale ETag should conflict, got %v", err)
	}

	accounts := storage.NewAccountStore(objects)
	indexKey := []byte("integration-index-key-material-32-bytes-minimum")
	service, err := ledger.NewService(accounts, indexKey, nil)
	if err != nil {
		t.Fatal(err)
	}
	identity := "integration-discord-" + runID
	results := make(chan string, 100)
	errorsSeen := make(chan error, 100)
	var group sync.WaitGroup
	for i := 0; i < 100; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			value, resolveErr := service.ResolveDiscord(ctx, identity, "integration-user")
			if resolveErr != nil {
				errorsSeen <- resolveErr
				return
			}
			results <- value.AccountID
		}()
	}
	group.Wait()
	close(results)
	close(errorsSeen)
	for resolveErr := range errorsSeen {
		t.Errorf("concurrent resolve: %v", resolveErr)
	}
	var accountID string
	for resolvedID := range results {
		if accountID == "" {
			accountID = resolvedID
		} else if resolvedID != accountID {
			t.Fatalf("identity resolved to %q and %q", accountID, resolvedID)
		}
	}
	if accountID == "" {
		t.Fatal("no logical account was created")
	}
	digest := security.IdentityHMAC(indexKey, "discord", identity)
	indexObjectKey, err := storage.IdentityIndexKey(1, digest)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { target.cleanup(t, indexObjectKey, "accounts/"+accountID+".json") })
}

func TestPrivateR2ContentReadAndPresign(t *testing.T) {
	deploy := loadTestR2(t, "R2_TEST_CONTENT_DEPLOY")
	runtime := loadTestR2(t, "R2_TEST_CONTENT_RUNTIME")
	if deploy.config.Bucket != runtime.config.Bucket {
		t.Fatalf("deploy and runtime credentials must target the same test content bucket")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	deployer, err := storage.NewR2ContentDeploymentStore(ctx, deploy.config)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := storage.NewR2ContentStore(ctx, runtime.config)
	if err != nil {
		t.Fatal(err)
	}
	key := fmt.Sprintf("integration/%d/member.txt", time.Now().UnixNano())
	t.Cleanup(func() { deploy.cleanup(t, key) })
	body := []byte("private integration object")
	if err := deployer.PutImmutable(ctx, key, body, "text/plain"); err != nil {
		t.Fatal(err)
	}
	if err := deployer.PutImmutable(ctx, key, body, "text/plain"); !errors.Is(err, storage.ErrConflict) {
		t.Fatalf("immutable overwrite should conflict, got %v", err)
	}
	object, err := reader.Get(ctx, key, 1024)
	if err != nil || string(object.Body) != string(body) {
		t.Fatalf("runtime read: body=%q err=%v", object.Body, err)
	}
	url, err := reader.PresignGET(ctx, key, "member.txt", time.Minute)
	if err != nil || !strings.HasPrefix(url, "https://") || !strings.Contains(url, "X-Amz-Signature=") {
		t.Fatalf("presign: url=%q err=%v", url, err)
	}
}
