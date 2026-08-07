(function confirmPayment() {
    const status = document.getElementById("paymentStatus");
    const deadline = Date.now() + 120000;

    async function poll() {
        try {
            const response = await fetch(`${window.PaywallConfig.gatewayBase}/api/account/status`, {
                credentials: "same-origin",
                cache: "no-store",
            });
            if (response.ok) {
                const account = await response.json();
                if (account.has_access) {
                    window.location.replace(`${window.PaywallConfig.gatewayBase}/app/`);
                    return;
                }
            } else if (response.status === 401) {
                status.textContent = "Your sign-in expired. Sign in again to check payment status.";
                return;
            }
        } catch (_) {
            // A transient network error is retried until the fixed deadline.
        }

        if (Date.now() >= deadline) {
            status.textContent =
                "Confirmation is taking longer than expected. Retry from the member area or contact support.";
            return;
        }
        window.setTimeout(poll, 2500);
    }

    poll();
})();
