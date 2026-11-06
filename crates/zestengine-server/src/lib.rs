//! The ZestEngine dedicated server (NET-2).
//!
//! A binary (`zestengine-server`) and a library: [`Server`] owns the headless
//! game host and its hosting session, [`ServerHandle`] runs one on its own
//! thread (tests, tools). Modules:
//!
//! - [`config`]: `server.ron` + command-line flags, validated.
//! - [`tick`]: the fixed-step pacer (accumulator, catch-up cap, sleep + spin).
//! - [`server`]: lifecycle, join/leave tracking, stats, graceful shutdown.
//! - [`admin`]: the admin command language (stdin console and signed remote admin).
//! - [`remote`]: signed remote admin over UDP (NET-6); [`admin_keys`]: key rotation while running.
//! - [`persist`], [`state_store`]: crash-safe persistent world and player state (NET-6).
//! - [`announce`]: directory heartbeat to a relay in directory mode (NET-6).
//! - [`health`]: heartbeat file, `--healthcheck` probe, Prometheus text, optional loopback HTTP (NET-7).
//! - [`logging`]: console + size-rotated file log.
//! - [`signals`]: SIGTERM / SIGINT / Ctrl+C.
//! - [`run`]: the whole `main`, reusable by a project's own server crate (NET-3).
//!
//! Built on `zestengine_engine::server::HeadlessHost` (NET-1) and the engine's
//! own `NetSession`; nothing here links a renderer.

pub mod admin;
pub mod admin_keys;
pub mod announce;
pub mod config;
pub mod health;
pub mod logging;
pub mod persist;
pub mod run;
pub mod remote;
pub mod server;
pub mod signals;
pub mod state_store;
pub mod tick;

pub use admin::{AdminCommand, AdminKey, AdminRole};
pub use run::run_main;
pub use config::{Cli, ServerConfig, parse_args};
pub use server::{ExitReason, Remote, Server, ServerHandle, selftest};
