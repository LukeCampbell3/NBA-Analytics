package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/jcthi/nba-analytics/paywall/content"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "content rollback failed:", err)
		os.Exit(1)
	}
}

func run() error {
	releaseID := flag.String("release", "", "existing immutable release ID to activate")
	flag.Parse()
	if *releaseID == "" {
		return errors.New("-release is required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	store, err := storage.NewR2ContentDeploymentStore(ctx, storage.R2Config{
		Endpoint:        os.Getenv("R2_CONTENT_ENDPOINT"),
		AccessKeyID:     os.Getenv("R2_CONTENT_DEPLOY_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("R2_CONTENT_DEPLOY_SECRET_ACCESS_KEY"),
		Bucket:          os.Getenv("R2_CONTENT_BUCKET"),
	})
	if err != nil {
		return err
	}
	pointer, err := content.RollbackRelease(ctx, store, *releaseID, time.Now().UTC())
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(pointer)
}
