# Private content source

Files placed here are inputs to the content-release uploader. They must never be
copied into the repository-root `dist/` static bundle or served by a local
public static-file handler.

Only these source prefixes are accepted:

- `app/` and `data/` for responses small enough to proxy through the Function.
- `downloads/` for fresh-authorized, short-lived presigned downloads.

Deploy from `paywall/tools` with an offline content-deploy credential:

```text
go run ./cmd/content-deploy -source ../private-content -release 2026-08-06-01
```
