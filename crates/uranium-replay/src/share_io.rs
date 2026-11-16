use std::io::Read;
use std::path::Path;

const ALIGN: usize = 4096;
const CHUNK: usize = 4 << 20;

pub fn sync_file(path: &Path) -> std::io::Result<()> {
    std::fs::OpenOptions::new().write(true).open(path)?.sync_all()
}

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

fn refused(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::InvalidInput || matches!(e.raw_os_error(), Some(22) if cfg!(unix)) || matches!(e.raw_os_error(), Some(87) if cfg!(windows))
}

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
        if n % ALIGN != 0 {
            let mut probe = [0u8; 1];
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
    match std::fs::OpenOptions::new().read(true).share_mode(7).custom_flags(FILE_FLAG_NO_BUFFERING).open(path) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.raw_os_error() == Some(87) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn open_uncached(_path: &Path) -> std::io::Result<Option<std::fs::File>> {
    Ok(None)
}

pub fn read_uncached_with(path: &Path, each: &mut dyn FnMut(&[u8])) -> std::io::Result<u64> {
    match open_uncached(path)? {
        Some(mut f) => match drain(&mut f, each) {
            Ok(n) => return Ok(n),
            Err(None) => warn_fallback(path),
            Err(Some(e)) => return Err(e),
        },
        None => warn_fallback(path),
    }
    drain(&mut std::fs::File::open(path)?, each).map_err(|e| e.unwrap_or_else(|| std::io::Error::other("read refused")))
}

fn warn_fallback(path: &Path) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("040-R17: warning: uncached read refused for {} (and possibly later files); reading through the cache", path.display()));
}

pub fn read_uncached(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    read_uncached_with(path, &mut |b| out.extend_from_slice(b))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
