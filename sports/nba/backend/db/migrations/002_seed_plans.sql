-- Seed subscription plans and default capability metadata
INSERT INTO plans (id, name, stripe_price_id, monthly_price_cents, tier, is_active) VALUES
    ('free', 'Free Research', NULL, 0, 'free', TRUE),
    ('plus', 'Plus Analytics', NULL, 1900, 'plus', TRUE),
    ('pro', 'Pro Research', NULL, 4900, 'pro', TRUE),
    ('api', 'API Access', NULL, 9900, 'api', TRUE)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    monthly_price_cents = EXCLUDED.monthly_price_cents,
    tier = EXCLUDED.tier,
    is_active = EXCLUDED.is_active;
