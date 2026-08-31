const assert = require("assert");
const contract = require("../web/unified-contract.js");

const base = {
    schema_version: "unified_mlb_v1",
    generation_id: "g1",
    generated_at_utc: "2026-08-31T12:00:00Z",
    policy_hash: "p1",
    run_date: "2026-08-31",
    singles: [],
    parlays: { two_leg: [], three_leg: [], four_leg: [] },
    evidence: {},
};
const manifest = { policy_hash: "p1", active_engine: "legacy" };

function response(ok, status, json) { return { ok, status, json }; }

(async () => {
    const options = { artifactUrl: "artifact", manifestUrl: "manifest", today: "2026-08-31", nowMs: Date.parse("2026-08-31T13:00:00Z") };
    const goodFetch = async (url) => response(true, 200, async () => url === "manifest" ? manifest : base);
    assert.equal((await contract.load({ ...options, fetchImpl: goodFetch })).state, "SHADOW");

    const notFound = async () => response(false, 404, async () => ({}));
    assert.equal((await contract.load({ ...options, fetchImpl: notFound })).state, "LOAD_ERROR");

    const malformed = async () => response(true, 200, async () => { throw new Error("bad"); });
    assert.match((await contract.load({ ...options, fetchImpl: malformed })).error, /Malformed JSON/);

    const empty = async (url) => response(true, 200, async () => url === "manifest" ? manifest : {});
    assert.match((await contract.load({ ...options, fetchImpl: empty })).error, /empty JSON/);

    assert.throws(() => contract.validate({ ...base, schema_version: "wrong" }, manifest, options), /schema mismatch/);
    assert.throws(() => contract.validate({ ...base, run_date: "2026-08-30" }, manifest, options), /not yet generated/);
    assert.throws(() => contract.validate({ ...base, generated_at_utc: "2026-08-20T00:00:00Z" }, manifest, options), /stale/);

    const hanging = async (_url, request) => new Promise((_resolve, reject) => request.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }))));
    const timed = await contract.load({ ...options, fetchImpl: hanging, timeoutMs: 5 });
    assert.equal(timed.state, "LOAD_ERROR");
    assert.match(timed.error, /timed out/);
    console.log("unified frontend failure states passed");
})().catch((error) => { console.error(error); process.exit(1); });
