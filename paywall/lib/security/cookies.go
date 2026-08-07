package security

import (
	"net/http"
	"time"
)

const (
	SessionCookieName    = "__Host-member_session"
	CSRFCookieName       = "__Host-csrf"
	OAuthStateCookieName = "__Host-oauth_state"
)

func SessionCookie(token string, lifetime time.Duration) *http.Cookie {
	return &http.Cookie{
		Name: SessionCookieName, Value: token, Path: "/", MaxAge: int(lifetime.Seconds()),
		Secure: true, HttpOnly: true, SameSite: http.SameSiteLaxMode,
	}
}

func CSRFCookie(token string, lifetime time.Duration) *http.Cookie {
	return &http.Cookie{
		Name: CSRFCookieName, Value: token, Path: "/", MaxAge: int(lifetime.Seconds()),
		Secure: true, HttpOnly: false, SameSite: http.SameSiteLaxMode,
	}
}

func OAuthStateCookie(state string, lifetime time.Duration) *http.Cookie {
	return &http.Cookie{
		Name: OAuthStateCookieName, Value: state, Path: "/", MaxAge: int(lifetime.Seconds()),
		Secure: true, HttpOnly: true, SameSite: http.SameSiteLaxMode,
	}
}

func ExpiredCookie(name string) *http.Cookie {
	return &http.Cookie{
		Name: name, Value: "", Path: "/", MaxAge: -1, Expires: time.Unix(1, 0),
		Secure: true, HttpOnly: name != CSRFCookieName, SameSite: http.SameSiteLaxMode,
	}
}
