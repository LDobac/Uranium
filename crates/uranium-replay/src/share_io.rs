//! 040-R17 / 061-R4: read a file back as the file server holds it, not from this client's cache.
//!
//! The R-4 rebuild (2026-10-19, `reports/m6_rebuild.md` R-5) published 7 shards whose bytes on
//! the share had 4–16 MiB zero-filled or truncated regions (multiples of the 4 MiB CIFS `wsize`)
//! although the read-back hash matched: two Linux CIFS clients (`cache=strict`) wrote them and
//! read them back from their own page cache, which still held the intended bytes. The kernel
//! logged nothing. So the check must (1) push every dirty page to the server and surface write
//! errors (`fsync`), then (2) read with the client cache out of the way:
//! - Linux: `posix_fadvise(DONTNEED)` drops the clean cached pages of the file, then the file is
//!   read with `O_DIRECT` (CIFS serves `O_DIRECT` reads with uncached SMB READs). File systems that
//!   refuse `O_DIRECT` (e.g. tmpfs) fall back to a plain read after the `DONTNEED`.
//! - Windows: `FILE_FLAG_NO_BUFFERING` bypasses the cache manager, so a read on an SMB share goes
//!   to the server (and on a local disk to the disk). A volume that refuses it falls back to a
//!   plain read.
//! - Other systems: plain read after `fsync`.
//!
//! Reads use a 4096-aligned buffer and 4 MiB requests, which satisfies the alignment rules of
//! both unbuffered paths (sector sizes up to 4096).

use std::io::Read;
use std::path::Path;

/// Alignment of the unbuffered read buffer and of every request size.
const ALIGN: usize = 4096;
/// One read request (a multiple of `ALIGN`).
const CHUNK: usize = 4 << 20;

/// `fsync` of a file that was just written through another handle (e.g. `std::fs::copy`): the
/// client sends its dirty pages to the server and reports write errors.
pub fn sync_file(path: &Path) -> std::io::Result<()> {
    std::fs::OpenOptions::new().write(true).open(path)?.sync_all()
}

/// A `CHUNK`-sized window of a buffer, aligned to `ALIGN`.
struct Aligned {
    buf: Vec<u8>,
    off: usize,
}

impl Aligned {
    fn new() -> Self {
        let buf = vec![0u8; CHUNK + ALIGN];
        let off = buf.as_ptr().align_offset(ALIGN);
        Self { buf, off }
    }

    fn get(&mut self) -> &mut [u8] {
        &mut self.buf[self.off..self.off + CHUNK]
    }
}

/// The unbuffered path is refused by this volume (Linux `EINVAL`, Windows
/// `ERROR_INVALID_PARAMETER`).
fn refused(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::InvalidInput || matches!(e.raw_os_error(), Some(22) if cfg!(unix)) || matches!(e.raw_os_error(), Some(87) if cfg!(windows))
}

/// Reads `f` to the end in `CHUNK` requests, handing each piece to `each`. `Err(None)` = the
/// first request was refused (`refused`), nothing was handed over.
fn drain(f: &mut std::fs::File, each: &mut dyn FnMut(&[u8])) -> Result<u64, Option<std::io::Error>> {
    let mut a = Aligned::new();
    let mut total = 0u64;
    loop {
        let buf = a.get();
        let n = match f.read(buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) if total == 0 && refused(&e) => return Err(None),
            Err(e) => return Err(Some(e)),
        };
        if n == 0 {
            return Ok(total);
        }
        each(&buf[..n]);
        total += n as u64;
        // An unbuffered read stops short only at the end of the file; reading on would ask for
        // an unaligned offset, so stop here.
        if n % ALIGN != 0 {
            let mut probe = [0u8; 1];
            // Plain handles may legally return short reads mid-file; finish those normally.
            if !matches!(f.read(&mut probe), Ok(0) | Err(_)) {
                each(&probe);
                total += 1;
                let mut rest = Vec::new();
                f.read_to_end(&mut rest).map_err(Some)?;
                each(&rest);
                total += rest.len() as u64;
            }
            return Ok(total);
        }
    }
}

#[cfg(target_os = "linux")]
fn open_uncached(path: &Path) -> std::io::Result<Option<std::fs::File>> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    let plain = std::fs::File::open(path)?;
    // Drop the clean cached pages (dirty ones were written by `sync_file`).
    // SAFETY: plain syscall on an open descriptor.
    unsafe { libc::posix_fadvise(plain.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(windows)]
fn open_uncached(path: &Path) -> std::io::Result<Option<std::fs::File>> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    // Allow other readers and writers (the shard folder is shared by several builds).
    match std::fs::OpenOptions::new().read(true).share_mode(7).custom_flags(FILE_FLAG_NO_BUFFERING).open(path) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.raw_os_error() == Some(87) => Ok(None), // ERROR_INVALID_PARAMETER
        Err(e) => Err(e),
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn open_uncached(_path: &Path) -> std::io::Result<Option<std::fs::File>> {
    Ok(None)
}

/// Streams the file as the server holds it (see the module notes) and returns its length. Call
/// [`sync_file`] first when this client wrote the file.
pub fn read_uncached_with(path: &Path, each: &mut dyn FnMut(&[u8])) -> std::io::Result<u64> {
    if let Some(mut f) = open_uncached(path)? {
        match drain(&mut f, each) {
            Ok(n) => return Ok(n),
            // The volume accepted the flag at open but refuses unbuffered reads.
            Err(None) => {}
            Err(Some(e)) => return Err(e),
        }
    }
    drain(&mut std::fs::File::open(path)?, each).map_err(|e| e.unwrap_or_else(|| std::io::Error::other("read refused")))
}

/// The whole file as the server holds it.
pub fn read_uncached(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    read_uncached_with(path, &mut |b| out.extend_from_slice(b))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 040-R17: the unbuffered read returns exactly the file's bytes for sizes around the
    /// alignment and request boundaries (the cache-bypass itself needs a CIFS client to observe).
    #[test]
    fn uncached_read_returns_the_file_040_r17() {
        let dir = std::env::temp_dir().join(format!("uranium-share-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in [0usize, 1, 4095, 4096, 4097, CHUNK - 1, CHUNK, CHUNK + 3, 2 * CHUNK + ALIGN + 17] {
            let p = dir.join(format!("f{n}"));
            let data: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
            std::fs::write(&p, &data).unwrap();
            sync_file(&p).unwrap();
            assert_eq!(read_uncached(&p).unwrap(), data, "size {n}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
