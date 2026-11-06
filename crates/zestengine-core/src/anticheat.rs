//! Anti-cheat provider abstraction (SEC-2).
//!
//! Server authority and validation (NET-4) are the engine's own defence. A
//! third-party client integrity product (Easy Anti-Cheat through Epic Online
//! Services, a platform attestation API, a custom module) plugs in through the
//! [`AntiCheatProvider`] trait: the session carries the provider's opaque
//! payloads between the client-side and server-side halves over the signed,
//! encrypted session, and asks the server half for a [`Verdict`] per client.
//!
//! Plain data and traits, no sockets: the session lives in the engine
//! (`net.rs`). Every config field has a serde default, so old scenes and
//! `server.ron` files load unchanged.
//!
//! No client anti-cheat is unbreakable; see `docs/research/security/anticheat.md`.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// Provider name that turns the layer off.
pub const PROVIDER_NONE: &str = "none";
/// Provider name of the Easy Anti-Cheat (EOS) binding (cargo feature `eos-anticheat`; not built yet).
pub const PROVIDER_EOS: &str = "eos";
/// Largest provider payload the session relays (bytes).
pub const MAX_PAYLOAD: usize = 4096;

/// Anti-cheat settings: the Network Manager inspector and `server.ron`
/// (`anticheat: (provider: "none", require: false)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AntiCheatConfig {
    /// `"none"` (default), `"eos"` (Easy Anti-Cheat, needs the `eos-anticheat` feature) or the
    /// name of a provider the game installed with `NetSession::set_anticheat`.
    pub provider: String,
    /// Server: hold a client out of gameplay until its verdict is Trusted, and
    /// remove it when rejected or when `pending_timeout` runs out. `false` =
    /// advisory: verdicts are logged and counted, nobody is held or kicked.
    pub require: bool,
    /// Server, with `require`: seconds a client may stay Pending before it is removed.
    pub pending_timeout: f32,
    /// Provider-specific settings (for `eos`: `product_id`, `sandbox_id`,
    /// `deployment_id`, `client_id`). Never put secrets in a file that is committed.
    pub options: BTreeMap<String, String>,
}

impl Default for AntiCheatConfig {
    fn default() -> Self {
        Self { provider: PROVIDER_NONE.into(), require: false, pending_timeout: 30.0, options: BTreeMap::new() }
    }
}

impl AntiCheatConfig {
    /// A provider other than `none` is selected.
    pub fn enabled(&self) -> bool {
        let p = self.provider.trim();
        !p.is_empty() && !p.eq_ignore_ascii_case(PROVIDER_NONE)
    }

    /// Clamp nonsense (NaN or tiny timeouts).
    pub fn sanitized(mut self) -> Self {
        if !self.pending_timeout.is_finite() || self.pending_timeout < 1.0 {
            self.pending_timeout = 30.0;
        }
        self
    }
}

/// The server half's opinion of one client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Not decided yet.
    Pending,
    /// The client passed.
    Trusted,
    /// The client failed; the text goes to the kick message and the log.
    Rejected(String),
}

/// The client half's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientStatus {
    /// `client_start` was not called.
    Idle,
    /// Started, not yet attested.
    Starting,
    /// Running and attested.
    Active,
    /// Could not start or was refused.
    Failed(String),
}

/// An anti-cheat product. One object serves one process: a game client uses
/// the `client_*` methods, a dedicated server the `server_*` ones. Payloads
/// are opaque to the engine (at most [`MAX_PAYLOAD`] bytes each) and are
/// delivered reliably and in order over the signed session.
pub trait AntiCheatProvider: Send {
    /// Name for logs and the inspector.
    fn name(&self) -> &str;

    /// Called every session pump with the session clock (seconds).
    fn tick(&mut self, _now: f64) {}

    // ------------------------------------------------------------ client half

    /// The client joined a session: start the product.
    fn client_start(&mut self) -> Result<(), String>;
    /// Messages the client half wants sent to the server.
    fn client_poll(&mut self, out: &mut Vec<Vec<u8>>);
    /// A message from the server half.
    fn client_receive(&mut self, payload: &[u8]);
    /// The client half's state.
    fn client_status(&self) -> ClientStatus;
    /// The session ended.
    fn client_stop(&mut self) {}

    // ------------------------------------------------------------ server half

    /// A client was admitted; `identity` is its verified player id (public key).
    fn server_register(&mut self, client: u64, identity: &[u8; 32]);
    /// Messages the server half wants sent to clients.
    fn server_poll(&mut self, out: &mut Vec<(u64, Vec<u8>)>);
    /// A message a client's half sent.
    fn server_receive(&mut self, client: u64, payload: &[u8]);
    /// The server half's opinion of this client.
    fn server_verdict(&self, client: u64) -> Verdict;
    /// The client left or was removed.
    fn server_unregister(&mut self, client: u64);
    /// A stable id the provider has for this client's account or machine (an EOS product user
    /// id, a Steam id ...), once it is known. The server links it to the player id so a ban
    /// also covers the account: a new key on the same account stays banned. `None` = the
    /// provider has no such id (the default, and the null provider).
    fn server_provider_id(&self, _client: u64) -> Option<String> {
        None
    }
}

/// Trusts everyone, says nothing: the default.
#[derive(Default)]
pub struct NullProvider {
    clients: Vec<u64>,
}

impl AntiCheatProvider for NullProvider {
    fn name(&self) -> &str {
        PROVIDER_NONE
    }
    fn client_start(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn client_poll(&mut self, _out: &mut Vec<Vec<u8>>) {}
    fn client_receive(&mut self, _payload: &[u8]) {}
    fn client_status(&self) -> ClientStatus {
        ClientStatus::Active
    }
    fn server_register(&mut self, client: u64, _identity: &[u8; 32]) {
        self.clients.push(client);
    }
    fn server_poll(&mut self, _out: &mut Vec<(u64, Vec<u8>)>) {}
    fn server_receive(&mut self, _client: u64, _payload: &[u8]) {}
    fn server_verdict(&self, _client: u64) -> Verdict {
        Verdict::Trusted
    }
    fn server_unregister(&mut self, client: u64) {
        self.clients.retain(|c| *c != client);
    }
}

/// Fails closed: the configured provider is not available in this build. With
/// `require` on, nobody joins; the reason says why.
pub struct UnavailableProvider {
    name: String,
    why: String,
}

impl UnavailableProvider {
    /// A provider that reports `why` for every call.
    pub fn new(name: &str, why: &str) -> Self {
        Self { name: name.to_owned(), why: why.to_owned() }
    }
}

impl AntiCheatProvider for UnavailableProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn client_start(&mut self) -> Result<(), String> {
        Err(self.why.clone())
    }
    fn client_poll(&mut self, _out: &mut Vec<Vec<u8>>) {}
    fn client_receive(&mut self, _payload: &[u8]) {}
    fn client_status(&self) -> ClientStatus {
        ClientStatus::Failed(self.why.clone())
    }
    fn server_register(&mut self, _client: u64, _identity: &[u8; 32]) {}
    fn server_poll(&mut self, _out: &mut Vec<(u64, Vec<u8>)>) {}
    fn server_receive(&mut self, _client: u64, _payload: &[u8]) {}
    fn server_verdict(&self, _client: u64) -> Verdict {
        Verdict::Rejected(format!("anti-cheat provider '{}' is unavailable: {}", self.name, self.why))
    }
    fn server_unregister(&mut self, _client: u64) {}
}

/// Build the provider a config names. Unknown names and providers missing
/// from this build give an [`UnavailableProvider`] (a warning is logged by the session).
pub fn create_provider(cfg: &AntiCheatConfig) -> Box<dyn AntiCheatProvider> {
    if !cfg.enabled() {
        return Box::new(NullProvider::default());
    }
    let name = cfg.provider.trim().to_ascii_lowercase();
    match name.as_str() {
        PROVIDER_EOS => {
            // Cargo feature `eos-anticheat` is a reserved stub: the EOS binding is not written.
            // See docs/research/security/anticheat.md for what it needs.
            Box::new(UnavailableProvider::new(PROVIDER_EOS, "the EOS binding is not built (feature `eos-anticheat` is a stub)"))
        }
        other => Box::new(UnavailableProvider::new(other, "no such provider; install one with NetSession::set_anticheat")),
    }
}

// ---------------------------------------------------------------- test provider

/// What a [`TestProvider`] decides for a client.
#[derive(Clone, Debug)]
pub enum TestPlan {
    /// Trusted after `delay` seconds.
    Trust {
        /// Seconds until the verdict.
        delay: f64,
    },
    /// Rejected after `delay` seconds.
    Reject {
        /// Seconds until the verdict.
        delay: f64,
        /// The rejection text.
        reason: String,
    },
    /// Never decided.
    Hang,
    /// Message round trip: the server sends `challenge`; the client half
    /// answers `response`; the server then trusts (or rejects, when
    /// `accept` is false). Exercises payload relay both ways.
    Handshake {
        /// Trust after the round trip (false = reject).
        accept: bool,
    },
}

/// Shared, inspectable state of a [`TestProvider`] (the test keeps a handle
/// after the provider is boxed into a session).
#[derive(Default, Debug)]
pub struct TestState {
    /// The session clock the provider last saw.
    pub now: f64,
    /// What the provider decides.
    pub plan: Option<TestPlan>,
    /// Server: (client, register time, decided by handshake?).
    pub registered: HashMap<u64, (f64, Option<bool>)>,
    /// Clients removed from the session.
    pub unregistered: Vec<u64>,
    /// Messages the server half wants sent.
    pub server_out: Vec<(u64, Vec<u8>)>,
    /// Messages the client half wants sent.
    pub client_out: Vec<Vec<u8>>,
    /// Messages the server half received.
    pub server_received: Vec<(u64, Vec<u8>)>,
    /// Messages the client half received.
    pub client_received: Vec<Vec<u8>>,
    /// The client half started.
    pub client_started: bool,
    /// Why the client half failed.
    pub client_fail: Option<String>,
    /// Server: the provider account id to report for a client (SEC identity linking).
    pub provider_ids: HashMap<u64, String>,
}

/// A scripted provider for tests: verdicts after a delay, rejections, hangs,
/// and a challenge/response exchange.
#[derive(Clone)]
pub struct TestProvider {
    /// Shared, inspectable state.
    pub state: Arc<Mutex<TestState>>,
}

/// The handshake challenge the server half sends.
pub const TEST_CHALLENGE: &[u8] = b"challenge";
/// The handshake answer the client half sends.
pub const TEST_RESPONSE: &[u8] = b"response";

impl TestProvider {
    /// A provider following `plan`.
    pub fn new(plan: TestPlan) -> Self {
        Self { state: Arc::new(Mutex::new(TestState { plan: Some(plan), ..Default::default() })) }
    }
    /// Trust every client after `delay` seconds.
    pub fn trust_after(delay: f64) -> Self {
        Self::new(TestPlan::Trust { delay })
    }
    /// Reject every client.
    pub fn reject(reason: &str) -> Self {
        Self::new(TestPlan::Reject { delay: 0.0, reason: reason.into() })
    }
    /// Never decide.
    pub fn hang() -> Self {
        Self::new(TestPlan::Hang)
    }
    /// Trust after a challenge/response round trip.
    pub fn handshake() -> Self {
        Self::new(TestPlan::Handshake { accept: true })
    }
    /// Make the provider report `id` as the account of client `peer`.
    pub fn set_provider_id(&self, peer: u64, id: &str) {
        self.state.lock().unwrap().provider_ids.insert(peer, id.to_owned());
    }
    /// Replace the plan for clients registered from now on.
    pub fn set_plan(&self, plan: TestPlan) {
        self.state.lock().unwrap().plan = Some(plan);
    }
}

impl AntiCheatProvider for TestProvider {
    fn name(&self) -> &str {
        "test"
    }
    fn tick(&mut self, now: f64) {
        self.state.lock().unwrap().now = now;
    }
    fn client_start(&mut self) -> Result<(), String> {
        let mut s = self.state.lock().unwrap();
        if let Some(e) = s.client_fail.clone() {
            return Err(e);
        }
        s.client_started = true;
        Ok(())
    }
    fn client_poll(&mut self, out: &mut Vec<Vec<u8>>) {
        out.append(&mut self.state.lock().unwrap().client_out);
    }
    fn client_receive(&mut self, payload: &[u8]) {
        let mut s = self.state.lock().unwrap();
        s.client_received.push(payload.to_vec());
        if payload == TEST_CHALLENGE {
            s.client_out.push(TEST_RESPONSE.to_vec());
        }
    }
    fn client_status(&self) -> ClientStatus {
        let s = self.state.lock().unwrap();
        match (&s.client_fail, s.client_started) {
            (Some(e), _) => ClientStatus::Failed(e.clone()),
            (None, true) => ClientStatus::Active,
            _ => ClientStatus::Idle,
        }
    }
    fn server_register(&mut self, client: u64, _identity: &[u8; 32]) {
        let mut s = self.state.lock().unwrap();
        let now = s.now;
        s.registered.insert(client, (now, None));
        if matches!(s.plan, Some(TestPlan::Handshake { .. })) {
            s.server_out.push((client, TEST_CHALLENGE.to_vec()));
        }
    }
    fn server_poll(&mut self, out: &mut Vec<(u64, Vec<u8>)>) {
        out.append(&mut self.state.lock().unwrap().server_out);
    }
    fn server_receive(&mut self, client: u64, payload: &[u8]) {
        let mut s = self.state.lock().unwrap();
        s.server_received.push((client, payload.to_vec()));
        let accept = matches!(s.plan, Some(TestPlan::Handshake { accept: true })) && payload == TEST_RESPONSE;
        if let Some(r) = s.registered.get_mut(&client) {
            r.1 = Some(accept);
        }
    }
    fn server_verdict(&self, client: u64) -> Verdict {
        let s = self.state.lock().unwrap();
        let Some((since, answered)) = s.registered.get(&client) else {
            return Verdict::Pending;
        };
        match &s.plan {
            Some(TestPlan::Trust { delay }) if s.now - since >= *delay => Verdict::Trusted,
            Some(TestPlan::Reject { delay, reason }) if s.now - since >= *delay => Verdict::Rejected(reason.clone()),
            Some(TestPlan::Handshake { .. }) => match answered {
                Some(true) => Verdict::Trusted,
                Some(false) => Verdict::Rejected("bad anti-cheat response".into()),
                None => Verdict::Pending,
            },
            _ => Verdict::Pending,
        }
    }
    fn server_unregister(&mut self, client: u64) {
        let mut s = self.state.lock().unwrap();
        s.registered.remove(&client);
        s.unregistered.push(client);
    }
    fn server_provider_id(&self, client: u64) -> Option<String> {
        self.state.lock().unwrap().provider_ids.get(&client).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_partial_ron() {
        let d = AntiCheatConfig::default();
        assert_eq!(d.provider, "none");
        assert!(!d.require && !d.enabled());
        let c: AntiCheatConfig = ron::from_str("(provider: \"eos\", require: true)").unwrap();
        assert!(c.enabled() && c.require);
        assert_eq!(c.pending_timeout, 30.0);
        let e: AntiCheatConfig = ron::from_str("()").unwrap();
        assert_eq!(e, d);
        let o: AntiCheatConfig = ron::from_str("(options: {\"product_id\": \"x\"})").unwrap();
        assert_eq!(o.options["product_id"], "x");
        assert_eq!(AntiCheatConfig { pending_timeout: f32::NAN, ..Default::default() }.sanitized().pending_timeout, 30.0);
    }

    #[test]
    fn null_trusts_and_missing_providers_fail_closed() {
        let mut n = create_provider(&AntiCheatConfig::default());
        n.server_register(2, &[0; 32]);
        assert_eq!(n.server_verdict(2), Verdict::Trusted);
        let mut u = create_provider(&AntiCheatConfig { provider: "eos".into(), ..Default::default() });
        assert!(matches!(u.server_verdict(2), Verdict::Rejected(_)));
        assert!(u.client_start().is_err());
    }

    #[test]
    fn test_provider_delay_and_reject() {
        let mut p = TestProvider::trust_after(2.0);
        p.tick(1.0);
        p.server_register(7, &[0; 32]);
        assert_eq!(p.server_verdict(7), Verdict::Pending);
        p.tick(3.0);
        assert_eq!(p.server_verdict(7), Verdict::Trusted);
        let mut r = TestProvider::reject("bad");
        r.server_register(7, &[0; 32]);
        assert_eq!(r.server_verdict(7), Verdict::Rejected("bad".into()));
    }
}
