//! Rendezvous and packet relay for internet sessions.
//!
//! Peers register with the relay over UDP: a host opens a room (the relay
//! picks a short room code, or keeps a requested one) and clients join it by
//! code. The relay introduces both sides (the public endpoint it sees, the
//! LAN endpoints the peer reported, and a random punch token) so they can try
//! a direct path by UDP hole punching. Until that works, or when it never
//! does (symmetric NATs, strict firewalls), game datagrams travel wrapped in
//! DATA packets and the relay forwards them between members of the room.
//! Star topology like the sessions themselves: the host reaches every client,
//! a client only the host.
//!
//! The relay only ever sees ciphertext: sessions encrypt end to end. It
//! does check who may do what (protocol 2): rooms are created with an owner
//! key (and, optionally, an operator token), joiners prove the session
//! secret with an HMAC over a stateless cookie, unauthenticated senders
//! never get a reply larger than their request, failures back off per IP,
//! and when the host leaves the room waits for a member holding the owner
//! key to claim the host slot (host migration).
//!
//! With `RelayConfig::directory` set the relay is also a server directory
//! (see [`directory`]): dedicated servers announce themselves with a signed
//! packet and players list them with filters.

pub mod directory;
pub mod proto;
mod server;

pub use directory::{DirConfig, DirStats, Directory, Filter, Listing, ServerInfo};
pub use server::{Relay, RelayConfig, RelayStats};
