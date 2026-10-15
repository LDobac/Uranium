use std::io::{Read, Write};

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Upper bound for one message; anything larger is treated as a corrupt stream.
pub const MAX_MESSAGE_BYTES: usize = 16 << 20;

#[derive(Debug)]
pub enum ProtoError {
    Io(std::io::Error),
    Codec(postcard::Error),
    TooLarge(usize),
}

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtoError::Io(e) => write!(f, "io: {e}"),
            ProtoError::Codec(e) => write!(f, "codec: {e}"),
            ProtoError::TooLarge(n) => write!(f, "message of {n} bytes exceeds {MAX_MESSAGE_BYTES}"),
        }
    }
}

impl std::error::Error for ProtoError {}

impl From<std::io::Error> for ProtoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<postcard::Error> for ProtoError {
    fn from(e: postcard::Error) -> Self {
        Self::Codec(e)
    }
}

/// Writes `u32 LE length + postcard payload` (010-R1). Returns the payload size.
pub fn write_msg<T: Serialize>(w: &mut impl Write, msg: &T) -> Result<usize, ProtoError> {
    let payload = postcard::to_stdvec(msg)?;
    if payload.len() > MAX_MESSAGE_BYTES {
        return Err(ProtoError::TooLarge(payload.len()));
    }
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(&payload)?;
    w.flush()?;
    Ok(payload.len())
}

pub fn read_msg<T: DeserializeOwned>(r: &mut impl Read) -> Result<T, ProtoError> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_MESSAGE_BYTES {
        return Err(ProtoError::TooLarge(len));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(postcard::from_bytes(&buf)?)
}
