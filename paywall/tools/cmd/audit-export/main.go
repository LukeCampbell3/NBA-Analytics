package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "audit export failed:", err)
		os.Exit(1)
	}
}

func run() error {
	date := flag.String("date", "", "UTC audit date in YYYY-MM-DD format")
	flag.Parse()
	if *date == "" {
		return errors.New("-date is required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	exporter, err := storage.NewR2AuditExporter(ctx, storage.R2Config{
		Endpoint:        os.Getenv("R2_STATE_ENDPOINT"),
		AccessKeyID:     os.Getenv("ADMIN_RECOVERY_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("ADMIN_RECOVERY_SECRET_ACCESS_KEY"),
		Bucket:          os.Getenv("R2_STATE_BUCKET"),
	})
	if err != nil {
		return err
	}
	events, err := exporter.ReadDay(ctx, *date)
	if err != nil {
		return err
	}
	for _, body := range events {
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.DisallowUnknownFields()
		var event observability.AuditEvent
		if err := decoder.Decode(&event); err != nil {
			return fmt.Errorf("malformed audit event: %w", err)
		}
		if err := decoder.Decode(&struct{}{}); err != io.EOF {
			return errors.New("malformed trailing audit data")
		}
		if err := json.NewEncoder(os.Stdout).Encode(event); err != nil {
			return err
		}
	}
	return nil
}
