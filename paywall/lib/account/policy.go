package account

import "time"

func (a Account) HasAccess(now time.Time) bool {
	switch a.Status {
	case StatusActive, StatusGrace:
		return now.Before(a.Entitlement.ValidUntil)
	default:
		return false
	}
}

func IsKnownStatus(status Status) bool {
	switch status {
	case StatusPending, StatusActive, StatusGrace, StatusPastDue,
		StatusSuspended, StatusCanceled, StatusBanned, StatusDeleted:
		return true
	default:
		return false
	}
}
