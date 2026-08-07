package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/jcthi/nba-analytics/paywall/ledger"
	"github.com/jcthi/nba-analytics/paywall/observability"
	"github.com/jcthi/nba-analytics/paywall/storage"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "admin command failed:", err)
		os.Exit(1)
	}
}

func run(arguments []string) error {
	if len(arguments) == 0 {
		return errors.New("expected suspend or recover command")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	objects, err := storage.NewR2ObjectStore(ctx, storage.R2Config{
		Endpoint:        os.Getenv("R2_STATE_ENDPOINT"),
		AccessKeyID:     os.Getenv("ADMIN_RECOVERY_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("ADMIN_RECOVERY_SECRET_ACCESS_KEY"),
		Bucket:          os.Getenv("R2_STATE_BUCKET"),
	})
	if err != nil {
		return err
	}
	accounts := storage.NewAccountStore(objects)
	audits, err := observability.NewAuditStore(objects)
	if err != nil {
		return err
	}
	switch arguments[0] {
	case "suspend":
		flags := flag.NewFlagSet("suspend", flag.ContinueOnError)
		accountID := flags.String("account", "", "account ID to suspend")
		if err := flags.Parse(arguments[1:]); err != nil {
			return err
		}
		if *accountID == "" || flags.NArg() != 0 {
			return errors.New("suspend requires -account")
		}
		mutationID, err := randomMutationID("admin-suspend")
		if err != nil {
			return err
		}
		if err := ledger.SuspendAccount(ctx, accounts, *accountID, mutationID); err != nil {
			return err
		}
		updated, _, err := accounts.GetAccount(ctx, *accountID)
		if err != nil {
			return err
		}
		if err := audits.Record(ctx, observability.AccountEvent("account.suspended", updated, "success", time.Now().UTC())); err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{
			"account_id": updated.AccountID, "status": updated.Status, "revision": updated.Revision,
		})
	case "recover":
		flags := flag.NewFlagSet("recover", flag.ContinueOnError)
		accountID := flags.String("account", "", "deleted canonical account ID")
		revision := flags.Uint64("revision", 0, "explicit latest history revision")
		if err := flags.Parse(arguments[1:]); err != nil {
			return err
		}
		if *accountID == "" || *revision == 0 || flags.NArg() != 0 {
			return errors.New("recover requires -account and -revision")
		}
		mutationID, err := randomMutationID("admin-recovery")
		if err != nil {
			return err
		}
		recovered, err := ledger.RecoverAccount(ctx, accounts, *accountID, *revision, mutationID, time.Now().UTC())
		if err != nil {
			return err
		}
		if err := audits.Record(ctx, observability.AccountEvent("account.recovered", recovered, "success", time.Now().UTC())); err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{
			"account_id": recovered.AccountID, "status": recovered.Status, "revision": recovered.Revision,
		})
	default:
		return fmt.Errorf("unknown admin command %q", arguments[0])
	}
}

func randomMutationID(prefix string) (string, error) {
	random := make([]byte, 16)
	if _, err := rand.Read(random); err != nil {
		return "", err
	}
	return prefix + ":" + hex.EncodeToString(random), nil
}
