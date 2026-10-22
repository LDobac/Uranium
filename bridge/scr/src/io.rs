//! I/O threads (012-R1): the game thread only puts messages on a channel and polls for Acts, so it
//! never waits on the network. One connection per game (`Hello`, `GameInfo`, frames, `GameEnd`).

use std::io::{BufReader, BufWriter, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::time::{Duration, Instant};

use uranium_proto::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Connecting = 0,
    Connected = 1,
    /// Could not connect or the connection broke (012-R16: the bot stays inactive).
    Failed = 2,
    Closed = 3,
}

fn status(v: u8) -> Status {
    match v {
        0 => Status::Connecting,
        1 => Status::Connected,
        2 => Status::Failed,
        _ => Status::Closed,
    }
}

pub struct Io {
    tx: Sender<BridgeMsg>,
    rx: Receiver<AgentMsg>,
    state: Arc<AtomicU8>,
}

/// How long the I/O thread keeps trying to reach the agent at game start.
const CONNECT_FOR: Duration = Duration::from_secs(10);

impl Io {
    /// Starts the connection to `addr`; `log` receives one line per event.
    pub fn start(addr: String, hello: Hello, info: GameInfo, log: fn(&str)) -> Io {
        let (tx, out_rx) = channel::<BridgeMsg>();
        let (in_tx, rx) = channel::<AgentMsg>();
        let state = Arc::new(AtomicU8::new(Status::Connecting as u8));
        let st = state.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + CONNECT_FOR;
            let stream = loop {
                match TcpStream::connect(&addr) {
                    Ok(s) => break s,
                    Err(e) if Instant::now() < deadline => {
                        log(&format!("connect {addr}: {e}; retrying"));
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    Err(e) => {
                        log(&format!("012-R16: cannot reach the agent at {addr}: {e}; the bot stays inactive this game"));
                        st.store(Status::Failed as u8, Ordering::Relaxed);
                        return;
                    }
                }
            };
            stream.set_nodelay(true).ok();
            let Ok(read_half) = stream.try_clone() else {
                st.store(Status::Failed as u8, Ordering::Relaxed);
                return;
            };
            let mut w = BufWriter::new(stream);
            let hello_ok = write_msg(&mut w, &BridgeMsg::Hello(hello)).and_then(|_| write_msg(&mut w, &BridgeMsg::GameInfo(Box::new(info)))).and_then(|_| w.flush().map_err(Into::into));
            if let Err(e) = hello_ok {
                log(&format!("sending Hello/GameInfo failed: {e}"));
                st.store(Status::Failed as u8, Ordering::Relaxed);
                return;
            }
            st.store(Status::Connected as u8, Ordering::Relaxed);
            log(&format!("connected to the agent at {addr}"));
            let reader_state = st.clone();
            std::thread::spawn(move || {
                let mut r = BufReader::new(read_half);
                loop {
                    match read_msg::<AgentMsg>(&mut r) {
                        Ok(m) => {
                            if in_tx.send(m).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            if reader_state.load(Ordering::Relaxed) == Status::Connected as u8 {
                                log(&format!("agent connection closed: {e}"));
                                reader_state.store(Status::Failed as u8, Ordering::Relaxed);
                            }
                            break;
                        }
                    }
                }
            });
            for msg in out_rx {
                let end = matches!(msg, BridgeMsg::End(_));
                if let Err(e) = write_msg(&mut w, &msg).and_then(|_| w.flush().map_err(Into::into)) {
                    log(&format!("sending to the agent failed: {e}"));
                    st.store(Status::Failed as u8, Ordering::Relaxed);
                    return;
                }
                if end {
                    st.store(Status::Closed as u8, Ordering::Relaxed);
                    if let Ok(s) = w.into_inner() {
                        s.shutdown(Shutdown::Both).ok();
                    }
                    return;
                }
            }
        });
        Io { tx, rx, state }
    }

    pub fn status(&self) -> Status {
        status(self.state.load(Ordering::Relaxed))
    }

    /// Queues a message for the agent (never blocks).
    pub fn send(&self, m: BridgeMsg) {
        self.tx.send(m).ok();
    }

    /// The next reply from the agent, if one arrived.
    pub fn try_recv(&self) -> Option<AgentMsg> {
        match self.rx.try_recv() {
            Ok(m) => Some(m),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }
}
