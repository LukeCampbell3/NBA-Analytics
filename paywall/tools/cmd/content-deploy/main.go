package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/jcthi/nba-analytics/paywall/content"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

const (
	maximumSourceFileBytes = int64(1024 * 1024 * 1024)
	maximumReleaseBytes    = int64(2 * 1024 * 1024 * 1024)
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "content deploy failed:", err)
		os.Exit(1)
	}
}

func run() error {
	source := flag.String("source", "../private-content", "private content source directory")
	releaseID := flag.String("release", "", "immutable release ID")
	plan := flag.String("plan", "individual", "required entitlement plan")
	flag.Parse()
	if *releaseID == "" {
		return errors.New("-release is required")
	}
	files, err := readSource(*source)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	store, err := storage.NewR2ContentDeploymentStore(ctx, deployConfig())
	if err != nil {
		return err
	}
	deployed, err := content.DeployRelease(ctx, store, *releaseID, *plan, files, time.Now().UTC())
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"release_id":   deployed.Pointer.ReleaseID,
		"objects":      len(deployed.Manifest.Objects),
		"activated_at": deployed.Pointer.ActivatedAt,
	})
}

func deployConfig() storage.R2Config {
	return storage.R2Config{
		Endpoint:        os.Getenv("R2_CONTENT_ENDPOINT"),
		AccessKeyID:     os.Getenv("R2_CONTENT_DEPLOY_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("R2_CONTENT_DEPLOY_SECRET_ACCESS_KEY"),
		Bucket:          os.Getenv("R2_CONTENT_BUCKET"),
	}
}

func readSource(root string) ([]content.SourceFile, error) {
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(absoluteRoot)
	if err != nil || !info.IsDir() {
		return nil, fmt.Errorf("source must be a directory")
	}
	var files []content.SourceFile
	var total int64
	err = filepath.WalkDir(absoluteRoot, func(filePath string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("symbolic links are not allowed: %s", filePath)
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > maximumSourceFileBytes {
			return fmt.Errorf("invalid source file: %s", filePath)
		}
		total += info.Size()
		if total > maximumReleaseBytes {
			return errors.New("release source exceeds 2 GiB safety limit")
		}
		relative, err := filepath.Rel(absoluteRoot, filePath)
		if err != nil {
			return err
		}
		if filepath.ToSlash(relative) == "README.md" {
			return nil
		}
		body, err := os.ReadFile(filePath)
		if err != nil {
			return err
		}
		files = append(files, content.SourceFile{LogicalPath: filepath.ToSlash(relative), Body: body})
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, errors.New("release source contains no files")
	}
	return files, nil
}
