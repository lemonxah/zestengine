//! Minimal Language Server Protocol client.
//!
//! Spawns a language server (e.g. rust-analyzer), speaks JSON-RPC over stdio,
//! and surfaces diagnostics, completion, and hover. A background thread reads
//! the server's output and answers its housekeeping requests; the main thread
//! sends document notifications + completion/hover requests and drains events
//! (diagnostics + responses) each frame.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};

/// One diagnostic from the server (0-based line).
pub struct LspDiagnostic {
    pub line: u32,
    /// 1 error, 2 warning, 3 info, 4 hint.
    pub severity: u8,
    pub message: String,
}

pub enum LspEvent {
    /// The server finished initializing (queued opens get flushed).
    Initialized,
    Diagnostics {
        path: PathBuf,
        diags: Vec<LspDiagnostic>,
    },
    /// A response to a request we made (completion, hover, …), by request id.
    Response { id: i64, result: Value },
}

pub struct LspClient {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    events: Receiver<LspEvent>,
    next_id: i64,
    ready: bool,
    versions: HashMap<PathBuf, i64>,
    /// Opens requested before the server was ready.
    pending_open: Vec<(PathBuf, String, String)>,
}

impl LspClient {
    /// Spawn `command` (e.g. "rust-analyzer") rooted at `root` and start the
    /// initialize handshake.
    pub fn spawn(command: &str, root: &Path) -> std::io::Result<Self> {
        let mut child = Command::new(command)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = Arc::new(Mutex::new(child.stdin.take().unwrap()));
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        let reader_stdin = stdin.clone();
        thread::spawn(move || reader_loop(stdout, reader_stdin, tx));

        let mut client = Self {
            child,
            stdin,
            events: rx,
            next_id: 1,
            ready: false,
            versions: HashMap::new(),
            pending_open: Vec::new(),
        };
        let params = json!({
            "processId": std::process::id(),
            "rootUri": path_to_uri(root),
            "capabilities": {
                "general": { "positionEncodings": ["utf-8", "utf-16"] },
                "textDocument": {
                    "publishDiagnostics": { "relatedInformation": false },
                    "synchronization": { "didSave": true },
                    "completion": {
                        "completionItem": { "snippetSupport": false },
                        "contextSupport": true
                    },
                    "hover": { "contentFormat": ["plaintext", "markdown"] }
                }
            },
            "initializationOptions": {
                // The plugin crate gates its editor-only deps (egui,
                // zestengine-editor) behind the `editor` feature, which no
                // build enables by default - tell rust-analyzer to load THAT
                // package with it (per-package feature map), so `egui::…` and
                // `zestengine_editor::…` enter its crate graph and complete.
                "cargoFeatures": { "zestengine-project-components": ["editor"] },
                "checkFeatures": { "zestengine-project-components": ["editor"] },
                // rust-analyzer's own cargo check (the diagnostics source) gets
                // its own target dir: sharing ours makes its check queue behind
                // the editor's plugin builds on the cargo file lock, and its
                // diagnostics then arrive late or not at all.
                "cargo": { "targetDir": true },
            },
        });
        client.send_request("initialize", params);
        Ok(client)
    }

    /// Request completions at a position; returns the request id to match the
    /// eventual `LspEvent::Response`.
    pub fn completion(&mut self, path: &Path, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/completion",
            json!({
                "textDocument": {"uri": path_to_uri(path)},
                "position": {"line": line, "character": character},
            }),
        )
    }

    /// Request hover info at a position; returns the request id.
    pub fn hover(&mut self, path: &Path, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/hover",
            json!({
                "textDocument": {"uri": path_to_uri(path)},
                "position": {"line": line, "character": character},
            }),
        )
    }

    /// Request the definition location at a position; returns the request id.
    pub fn definition(&mut self, path: &Path, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/definition",
            json!({
                "textDocument": {"uri": path_to_uri(path)},
                "position": {"line": line, "character": character},
            }),
        )
    }

    /// Request reference locations at a position; returns the request id.
    pub fn references(&mut self, path: &Path, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/references",
            json!({
                "textDocument": {"uri": path_to_uri(path)},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": true},
            }),
        )
    }

    /// Request code actions (quick fixes / refactors) at a position; returns
    /// the request id. The response is a `CodeAction[]` (or null).
    pub fn code_action(&mut self, path: &Path, line: u32, character: u32) -> i64 {
        self.send_request(
            "textDocument/codeAction",
            json!({
                "textDocument": {"uri": path_to_uri(path)},
                "range": {
                    "start": {"line": line, "character": character},
                    "end": {"line": line, "character": character},
                },
                "context": {"diagnostics": []},
            }),
        )
    }

    fn send_request(&mut self, method: &str, params: Value) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        write_message(&self.stdin, &msg);
        id
    }

    fn send_notification(&self, method: &str, params: Value) {
        let msg = json!({"jsonrpc":"2.0","method":method,"params":params});
        write_message(&self.stdin, &msg);
    }

    /// Tell the server a document is open. Queued until the server is ready.
    pub fn open(&mut self, path: &Path, language_id: &str, text: &str) {
        if !self.ready {
            self.pending_open
                .push((path.to_owned(), language_id.to_owned(), text.to_owned()));
            return;
        }
        self.versions.insert(path.to_owned(), 1);
        self.send_notification(
            "textDocument/didOpen",
            json!({"textDocument": {
                "uri": path_to_uri(path),
                "languageId": language_id,
                "version": 1,
                "text": text,
            }}),
        );
    }

    /// Send a full-document change (open-on-demand if not yet tracked).
    pub fn change(&mut self, path: &Path, language_id: &str, text: &str) {
        if !self.versions.contains_key(path) {
            self.open(path, language_id, text);
            return;
        }
        let version = {
            let v = self.versions.get_mut(path).unwrap();
            *v += 1;
            *v
        };
        self.send_notification(
            "textDocument/didChange",
            json!({
                "textDocument": {"uri": path_to_uri(path), "version": version},
                "contentChanges": [ {"text": text} ],
            }),
        );
    }

    pub fn save(&self, path: &Path) {
        if !self.ready {
            return;
        }
        self.send_notification(
            "textDocument/didSave",
            json!({"textDocument": {"uri": path_to_uri(path)}}),
        );
    }

    /// Drain server events; flushes queued opens once the server is ready.
    pub fn poll(&mut self) -> Vec<LspEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = self.events.try_recv() {
            match ev {
                LspEvent::Initialized => {
                    self.ready = true;
                    for (p, lang, text) in std::mem::take(&mut self.pending_open) {
                        self.open(&p, &lang, &text);
                    }
                    out.push(LspEvent::Initialized);
                }
                other => out.push(other),
            }
        }
        out
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.send_notification("exit", json!(null));
        let _ = self.child.kill();
    }
}

/// Frame a JSON-RPC message and write it to the server's stdin.
fn write_message(stdin: &Arc<Mutex<ChildStdin>>, msg: &Value) {
    let body = serde_json::to_string(msg).unwrap_or_default();
    let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
    if let Ok(mut w) = stdin.lock() {
        let _ = w.write_all(framed.as_bytes());
        let _ = w.flush();
    }
}

/// Background reader: parse server messages, answer its requests, forward
/// diagnostics + the initialize result.
fn reader_loop(stdout: ChildStdout, stdin: Arc<Mutex<ChildStdin>>, tx: Sender<LspEvent>) {
    let mut reader = BufReader::new(stdout);
    loop {
        // Read headers until a blank line.
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => return, // EOF / server died
                Ok(_) => {}
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
                content_length = rest.trim().parse().unwrap_or(0);
            }
        }
        if content_length == 0 {
            continue;
        }
        let mut buf = vec![0u8; content_length];
        if reader.read_exact(&mut buf).is_err() {
            return;
        }
        let Ok(msg) = serde_json::from_slice::<Value>(&buf) else {
            continue;
        };

        let has_id = msg.get("id").is_some();
        let method = msg.get("method").and_then(|m| m.as_str());

        if has_id && method.is_some() {
            // Server → client request: reply with a null result so the server
            // doesn't stall (configuration/registerCapability/progress, …).
            let reply = json!({"jsonrpc":"2.0","id": msg["id"].clone(), "result": Value::Null});
            write_message(&stdin, &reply);
            continue;
        }
        if has_id && (msg.get("result").is_some() || msg.get("error").is_some()) {
            let id = msg.get("id").and_then(|i| i.as_i64()).unwrap_or(-1);
            if id == 1 {
                // initialize response: ack + go ready.
                write_message(
                    &stdin,
                    &json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
                );
                let _ = tx.send(LspEvent::Initialized);
            } else {
                let result = msg.get("result").cloned().unwrap_or(Value::Null);
                let _ = tx.send(LspEvent::Response { id, result });
            }
            continue;
        }
        if method == Some("textDocument/publishDiagnostics")
            && let Some(params) = msg.get("params")
        {
            let uri = params.get("uri").and_then(|u| u.as_str()).unwrap_or("");
            let Some(path) = uri_to_path(uri) else {
                continue;
            };
            let mut diags = Vec::new();
            if let Some(arr) = params.get("diagnostics").and_then(|d| d.as_array()) {
                for d in arr {
                    let line = d
                        .pointer("/range/start/line")
                        .and_then(|l| l.as_u64())
                        .unwrap_or(0) as u32;
                    let severity = d.get("severity").and_then(|s| s.as_u64()).unwrap_or(1) as u8;
                    let message = d
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("")
                        .to_owned();
                    diags.push(LspDiagnostic {
                        line,
                        severity,
                        message,
                    });
                }
            }
            let _ = tx.send(LspEvent::Diagnostics { path, diags });
        }
    }
}

fn path_to_uri(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    uri.strip_prefix("file://").map(PathBuf::from)
}
