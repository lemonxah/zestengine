//! The online server browser (NET-6): ask a directory (a `zestengine-relay --directory`) for the
//! dedicated servers it lists, filter and sort them, ping each one and pick the address to join.
//!
//! Everything here is plain data and functions (no GUI, no frame loop): a query runs on a worker
//! thread and hands its results over a channel, so the editor window (`net_window`), a game's own
//! menu or a test just call [`ServerBrowser::refresh`] and [`ServerBrowser::poll`] every frame.
//!
//! The pieces that decide things are pure and tested on their own: [`parse_directory`],
//! [`BrowserFilter::wire`] / [`BrowserFilter::keeps`], [`sort_rows`], [`join_target`].

use std::net::SocketAddr;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Duration;

use zestengine_relay::directory::{Filter, Listing, query_blocking};

use crate::net::{PROTOCOL, ServerProbe, parse_relay, probe_server};

/// How long the directory is given to answer.
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(3);
/// How long each server is given to answer a ping.
pub const PING_TIMEOUT: Duration = Duration::from_millis(1200);
/// Servers pinged per refresh (the rest are listed without a ping).
pub const MAX_PINGS: usize = 128;
/// Pings in flight at once.
const PING_THREADS: usize = 16;

/// Resolve the directory text a player typed (`host`, `host:port`, `[::1]:47800`).
pub fn parse_directory(text: &str) -> Result<SocketAddr, String> {
    let t = text.trim();
    if t.is_empty() {
        return Err("no directory address (host or host:port of a relay running --directory)".into());
    }
    parse_relay(t).map_err(|e| format!("cannot use the directory address {t:?}: {e}"))
}

/// What the player filters the list by. The text and flag fields go to the directory as they are
/// (it filters before it pages); `hide_incompatible` is applied here, after the ping.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BrowserFilter {
    pub region: String,
    pub game: String,
    pub version: String,
    pub name_contains: String,
    pub hide_full: bool,
    pub hide_empty: bool,
    pub hide_password: bool,
    /// Hide servers whose network protocol is not this build's.
    pub hide_incompatible: bool,
}

impl BrowserFilter {
    /// The directory's own filter.
    pub fn wire(&self) -> Filter {
        Filter {
            region: self.region.trim().to_owned(),
            game: self.game.trim().to_owned(),
            version: self.version.trim().to_owned(),
            name_contains: self.name_contains.trim().to_owned(),
            hide_full: self.hide_full,
            hide_empty: self.hide_empty,
            hide_password: self.hide_password,
        }
    }

    /// Whether a row passes the client-side part of the filter.
    pub fn keeps(&self, row: &Row) -> bool {
        !(self.hide_incompatible && row.compatible == Some(false))
    }
}

/// One server in the list.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub listing: Listing,
    /// Round trip in milliseconds once the server answered a ping.
    pub ping_ms: Option<f32>,
    /// `Some(false)` when the server answered with a different network protocol, `None` before
    /// it answered (the directory's declared protocol is checked from the start).
    pub compatible: Option<bool>,
    /// The server was pinged and did not answer.
    pub unreachable: bool,
}

impl Row {
    fn new(listing: Listing) -> Self {
        let compatible = (listing.info.protocol != 0).then_some(listing.info.protocol == PROTOCOL);
        Self { listing, ping_ms: None, compatible, unreachable: false }
    }

    pub fn name(&self) -> &str {
        &self.listing.info.name
    }

    pub fn is_full(&self) -> bool {
        self.listing.info.max_players > 0 && self.listing.info.players >= self.listing.info.max_players
    }

    /// `players/max`.
    pub fn players_text(&self) -> String {
        let i = &self.listing.info;
        if i.max_players > 0 { format!("{}/{}", i.players, i.max_players) } else { i.players.to_string() }
    }

    /// The join button works: compatible (or not known to be incompatible) and not full.
    pub fn joinable(&self) -> bool {
        self.compatible != Some(false) && !self.is_full()
    }

    /// Why the join button is off, for its tooltip.
    pub fn why_not(&self) -> Option<&'static str> {
        if self.compatible == Some(false) {
            Some("this server runs a different network protocol: update the game or pick another server")
        } else if self.is_full() {
            Some("the server is full")
        } else {
            None
        }
    }
}

/// What to put in the Address field to join `row`: its address and game port, or, for a server
/// that never answered a ping and is only reachable through a relay, `ROOM@directory`.
pub fn join_target(row: &Row, directory: SocketAddr) -> String {
    let info = &row.listing.info;
    if !info.room.is_empty() && row.ping_ms.is_none() {
        format!("{}@{directory}", info.room)
    } else {
        row.listing.addr.to_string()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    Name,
    Players,
    Ping,
    Region,
}

/// Sort in place. Ties break by name; servers without a ping sort after those with one when
/// ascending (and first when descending), so the unreachable never head the list.
pub fn sort_rows(rows: &mut [Row], key: SortKey, ascending: bool) {
    let by_name = |a: &Row, b: &Row| a.name().to_lowercase().cmp(&b.name().to_lowercase());
    rows.sort_by(|a, b| {
        let ord = match key {
            SortKey::Name => by_name(a, b),
            SortKey::Players => a.listing.info.players.cmp(&b.listing.info.players).then_with(|| by_name(a, b)),
            SortKey::Region => a.listing.info.region.to_lowercase().cmp(&b.listing.info.region.to_lowercase()).then_with(|| by_name(a, b)),
            SortKey::Ping => match (a.ping_ms, b.ping_ms) {
                (Some(x), Some(y)) => x.total_cmp(&y).then_with(|| by_name(a, b)),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => by_name(a, b),
            },
        };
        if ascending { ord } else { ord.reverse() }
    });
}

/// Where a refresh stands.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum BrowserStatus {
    #[default]
    Idle,
    /// Asking the directory.
    Querying,
    /// Listed; pinging the servers.
    Pinging,
    /// Done: `n` servers.
    Ready,
    Failed(String),
}

enum Msg {
    Listings(Vec<Listing>),
    Ping { key: [u8; 32], probe: Option<ServerProbe> },
    Done,
    Failed(String),
}

/// A directory query with filters, pings and a sorted list. Poll it every frame.
pub struct ServerBrowser {
    pub directory: String,
    pub filter: BrowserFilter,
    pub sort: SortKey,
    pub ascending: bool,
    rows: Vec<Row>,
    status: BrowserStatus,
    rx: Option<Receiver<Msg>>,
    dirty: bool,
}

impl Default for ServerBrowser {
    fn default() -> Self {
        Self {
            directory: std::env::var("ZESTENGINE_DIRECTORY").unwrap_or_default(),
            filter: BrowserFilter::default(),
            sort: SortKey::Ping,
            ascending: true,
            rows: Vec::new(),
            status: BrowserStatus::Idle,
            rx: None,
            dirty: false,
        }
    }
}

impl ServerBrowser {
    /// A browser for the directory at `directory` (`host` or `host:port`).
    pub fn new(directory: &str) -> Self {
        Self { directory: directory.to_owned(), ..Default::default() }
    }

    pub fn status(&self) -> &BrowserStatus {
        &self.status
    }

    pub fn busy(&self) -> bool {
        matches!(self.status, BrowserStatus::Querying | BrowserStatus::Pinging)
    }

    /// The rows that pass the client-side filter, in the chosen order.
    pub fn rows(&self) -> impl Iterator<Item = &Row> {
        self.rows.iter().filter(|r| self.filter.keeps(r))
    }

    /// How many were listed before the client-side filter.
    pub fn total(&self) -> usize {
        self.rows.len()
    }

    /// Change the sort (a header click): the same key flips the direction.
    pub fn sort_by(&mut self, key: SortKey) {
        if self.sort == key {
            self.ascending = !self.ascending;
        } else {
            self.sort = key;
            self.ascending = true;
        }
        self.dirty = true;
    }

    /// Start a new query (a running one is abandoned; its answers are dropped).
    pub fn refresh(&mut self) {
        let dir = match parse_directory(&self.directory) {
            Ok(d) => d,
            Err(e) => {
                self.status = BrowserStatus::Failed(e);
                self.rx = None;
                return;
            }
        };
        let (tx, rx) = channel();
        let wire = self.filter.wire();
        std::thread::spawn(move || run_query(dir, wire, tx));
        self.rx = Some(rx);
        self.status = BrowserStatus::Querying;
    }

    /// Take what the worker has produced. Returns whether the rows changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = std::mem::take(&mut self.dirty);
        while let Some(rx) = &self.rx {
            match rx.try_recv() {
                Ok(Msg::Listings(list)) => {
                    self.rows = list.into_iter().map(Row::new).collect();
                    self.status = if self.rows.is_empty() { BrowserStatus::Ready } else { BrowserStatus::Pinging };
                    changed = true;
                }
                Ok(Msg::Ping { key, probe }) => {
                    if let Some(row) = self.rows.iter_mut().find(|r| r.listing.key == key) {
                        match probe {
                            Some(p) => {
                                row.ping_ms = Some(p.rtt_ms);
                                row.compatible = Some(p.protocol == PROTOCOL);
                            }
                            None => row.unreachable = true,
                        }
                        changed = true;
                    }
                }
                Ok(Msg::Done) => {
                    self.status = BrowserStatus::Ready;
                    self.rx = None;
                }
                Ok(Msg::Failed(e)) => {
                    self.status = BrowserStatus::Failed(e);
                    self.rx = None;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.busy() {
                        self.status = BrowserStatus::Ready;
                    }
                    self.rx = None;
                }
            }
        }
        if changed {
            sort_rows(&mut self.rows, self.sort, self.ascending);
        }
        changed
    }

    /// The address to join for `row` with this browser's directory.
    pub fn join_target(&self, row: &Row) -> Option<String> {
        Some(join_target(row, parse_directory(&self.directory).ok()?))
    }
}

fn run_query(dir: SocketAddr, filter: Filter, tx: std::sync::mpsc::Sender<Msg>) {
    let listings = match query_blocking(dir, &filter, QUERY_TIMEOUT) {
        Ok(l) => l,
        Err(e) => {
            let _ = tx.send(Msg::Failed(format!("the directory {dir} did not answer: {e}")));
            return;
        }
    };
    let targets: Vec<([u8; 32], SocketAddr)> = listings.iter().take(MAX_PINGS).map(|l| (l.key, l.addr)).collect();
    if tx.send(Msg::Listings(listings)).is_err() {
        return;
    }
    // Ping in small batches so a long list does not open a hundred sockets at once.
    for chunk in targets.chunks(PING_THREADS) {
        let handles: Vec<_> = chunk
            .iter()
            .map(|&(key, addr)| {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(Msg::Ping { key, probe: probe_server(addr, PING_TIMEOUT) });
                })
            })
            .collect();
        for h in handles {
            let _ = h.join();
        }
    }
    let _ = tx.send(Msg::Done);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_relay::directory::ServerInfo;

    fn listing(n: u8, name: &str, players: u16, max: u16, region: &str) -> Listing {
        Listing {
            key: [n; 32],
            addr: format!("10.0.0.{n}:9000").parse().unwrap(),
            age_secs: 1,
            info: ServerInfo { name: name.into(), game: "g".into(), version: "1".into(), map: "m".into(), region: region.into(), players, max_players: max, port: 9000, room: String::new(), password: false, protocol: PROTOCOL },
        }
    }

    fn row(n: u8, name: &str, players: u16, max: u16, ping: Option<f32>) -> Row {
        let mut r = Row::new(listing(n, name, players, max, "eu"));
        r.ping_ms = ping;
        r
    }

    #[test]
    fn directory_text_gets_the_default_port() {
        assert_eq!(parse_directory("127.0.0.1:5000").unwrap().port(), 5000);
        assert_eq!(parse_directory("  127.0.0.1 ").unwrap().port(), zestengine_relay::proto::DEFAULT_PORT);
        assert!(parse_directory("").unwrap_err().contains("no directory"));
        assert!(parse_directory("not a host name!!").is_err());
    }

    #[test]
    fn the_filter_goes_to_the_directory_and_incompatible_servers_are_hidden_here() {
        let f = BrowserFilter { region: " eu ".into(), name_contains: "arena".into(), hide_full: true, hide_incompatible: true, ..Default::default() };
        let w = f.wire();
        assert_eq!((w.region.as_str(), w.name_contains.as_str(), w.hide_full, w.hide_empty), ("eu", "arena", true, false));
        let mut old = row(1, "Old", 1, 8, None);
        old.compatible = Some(false);
        assert!(!f.keeps(&old));
        assert!(f.keeps(&row(2, "New", 1, 8, None)));
        assert!(BrowserFilter::default().keeps(&old), "shown (but not joinable) unless hidden");
        assert!(!old.joinable() && old.why_not().unwrap().contains("protocol"));
        assert!(!row(3, "Full", 8, 8, None).joinable() && row(3, "Full", 8, 8, None).why_not() == Some("the server is full"));
        let mut other_proto = listing(4, "X", 0, 8, "eu");
        other_proto.info.protocol = PROTOCOL + 1;
        assert_eq!(Row::new(other_proto).compatible, Some(false), "the directory's own protocol field counts before any ping");
        let mut unknown = listing(5, "Y", 0, 8, "eu");
        unknown.info.protocol = 0;
        assert_eq!(Row::new(unknown).compatible, None);
    }

    #[test]
    fn sorting_by_every_key_and_direction() {
        let mut rows = vec![row(1, "bravo", 5, 16, Some(80.0)), row(2, "Alpha", 12, 16, None), row(3, "charlie", 1, 16, Some(20.0)), row(4, "delta", 5, 16, Some(80.0))];
        let names = |r: &[Row]| r.iter().map(|x| x.name().to_owned()).collect::<Vec<_>>();
        sort_rows(&mut rows, SortKey::Name, true);
        assert_eq!(names(&rows), ["Alpha", "bravo", "charlie", "delta"], "case-insensitive");
        sort_rows(&mut rows, SortKey::Ping, true);
        assert_eq!(names(&rows), ["charlie", "bravo", "delta", "Alpha"], "no ping last; ties by name");
        sort_rows(&mut rows, SortKey::Ping, false);
        assert_eq!(names(&rows), ["Alpha", "delta", "bravo", "charlie"]);
        sort_rows(&mut rows, SortKey::Players, false);
        assert_eq!(names(&rows)[0], "Alpha");
        sort_rows(&mut rows, SortKey::Players, true);
        assert_eq!(names(&rows)[0], "charlie");
    }

    #[test]
    fn join_uses_the_address_or_the_relay_room() {
        let dir: SocketAddr = "192.0.2.1:47800".parse().unwrap();
        let mut direct = row(1, "A", 0, 8, Some(30.0));
        direct.listing.info.room = "ABC123".into();
        assert_eq!(join_target(&direct, dir), "10.0.0.1:9000", "it answered a ping: go direct");
        let mut relayed = row(2, "B", 0, 8, None);
        relayed.listing.info.room = "ABC123".into();
        assert_eq!(join_target(&relayed, dir), "ABC123@192.0.2.1:47800", "only reachable through the relay");
        assert_eq!(join_target(&row(3, "C", 0, 8, None), dir), "10.0.0.3:9000");
        // The relay form is what `JoinTarget` parses.
        assert!(crate::net::parse_relay("192.0.2.1:47800").is_ok());
    }

    #[test]
    fn a_bad_directory_fails_at_once_and_a_dead_one_fails_with_a_message() {
        let mut b = ServerBrowser { directory: String::new(), ..Default::default() };
        b.refresh();
        assert!(matches!(b.status(), BrowserStatus::Failed(m) if m.contains("no directory")));
        // A port nothing listens on: the query times out into a readable error.
        let sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let mut b = ServerBrowser { directory: sock.local_addr().unwrap().to_string(), ..Default::default() };
        b.refresh();
        assert_eq!(*b.status(), BrowserStatus::Querying);
        let start = std::time::Instant::now();
        while b.busy() && start.elapsed() < Duration::from_secs(8) {
            b.poll();
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(matches!(b.status(), BrowserStatus::Failed(m) if m.contains("did not answer")), "{:?}", b.status());
    }
}
