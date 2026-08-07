package security

import (
	"errors"
	"net/url"
	"strings"
)

var ErrInvalidOrigin = errors.New("invalid request origin")

func ValidateExactOrigin(rawOrigin, expectedOrigin string) error {
	actual, err := normalizedOrigin(rawOrigin)
	if err != nil {
		return ErrInvalidOrigin
	}
	expected, err := normalizedOrigin(expectedOrigin)
	if err != nil || actual != expected {
		return ErrInvalidOrigin
	}
	return nil
}

func normalizedOrigin(raw string) (string, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil ||
		parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", ErrInvalidOrigin
	}
	return strings.ToLower(parsed.Scheme + "://" + parsed.Host), nil
}
