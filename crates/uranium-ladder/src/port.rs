//! 043-R3: find the local web API port of the running SC:R client.
//!
//! Uses `tasklist` and `netstat -ano`, which need no administrator rights (unlike
//! `Get-NetTCPConnection -OwningProcess`). The caller picks among candidates by asking each
//! one for the leaderboard list.

use std::process::Command;

use anyhow::{Context, Result, bail};

const PROCESS_IMAGE: &str = "StarCraft.exe";

/// PIDs from `tasklist /FO CSV /NH` output whose image name is `StarCraft.exe`.
pub fn parse_tasklist_pids(csv: &str) -> Vec<u32> {
    let mut pids = Vec::new();
    for line in csv.lines() {
        let fields: Vec<&str> = line.split("\",\"").map(|f| f.trim().trim_matches('"')).collect();
        if fields.len() >= 2 && fields[0].eq_ignore_ascii_case(PROCESS_IMAGE) {
            if let Ok(pid) = fields[1].parse() {
                pids.push(pid);
            }
        }
    }
    pids
}

/// Loopback-reachable LISTENING TCP ports owned by `pids`, from `netstat -ano -p TCP` output.
pub fn parse_netstat_listen(out: &str, pids: &[u32]) -> Vec<u16> {
    let mut ports = Vec::new();
    for line in out.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        // Proto, Local Address, Foreign Address, State, PID
        if cols.len() != 5 || !cols[0].eq_ignore_ascii_case("TCP") || cols[3] != "LISTENING" {
            continue;
        }
        let Ok(pid) = cols[4].parse::<u32>() else { continue };
        if !pids.contains(&pid) {
            continue;
        }
        let Some((host, port)) = cols[1].rsplit_once(':') else { continue };
        if host != "127.0.0.1" && host != "0.0.0.0" {
            continue;
        }
        if let Ok(p) = port.parse() {
            ports.push(p);
        }
    }
    ports.sort_unstable();
    ports.dedup();
    ports
}

/// Candidate ports of the running client, lowest first.
pub fn candidates() -> Result<Vec<u16>> {
    let tasks = run("tasklist", &["/FI", &format!("IMAGENAME eq {PROCESS_IMAGE}"), "/FO", "CSV", "/NH"])?;
    let pids = parse_tasklist_pids(&tasks);
    if pids.is_empty() {
        bail!("{PROCESS_IMAGE} is not running (log in to SC:R first, or set URANIUM_SCR_API_PORT)");
    }
    let ports = parse_netstat_listen(&run("netstat", &["-ano", "-p", "TCP"])?, &pids);
    if ports.is_empty() {
        bail!("{PROCESS_IMAGE} has no listening TCP port yet (is the client logged in?)");
    }
    Ok(ports)
}

fn run(cmd: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(cmd).args(args).output().with_context(|| format!("running {cmd}"))?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r043_r3_tasklist_csv_pids() {
        let csv = "\"StarCraft.exe\",\"12345\",\"Console\",\"1\",\"812,004 K\"\r\n\
                   \"starcraft.exe\",\"222\",\"Console\",\"1\",\"1 K\"\r\n\
                   \"Battle.net.exe\",\"999\",\"Console\",\"1\",\"1 K\"\r\n";
        assert_eq!(parse_tasklist_pids(csv), vec![12345, 222]);
        assert!(parse_tasklist_pids("INFO: No tasks are running which match the specified criteria.").is_empty());
    }

    #[test]
    fn r043_r3_netstat_listening_ports_of_pid() {
        let out = "\r\nActive Connections\r\n\r\n  Proto  Local Address          Foreign Address        State           PID\r\n\
                   \x20 TCP    127.0.0.1:57421        0.0.0.0:0              LISTENING       12345\r\n\
                   \x20 TCP    0.0.0.0:6112           0.0.0.0:0              LISTENING       12345\r\n\
                   \x20 TCP    192.0.2.1:5000         0.0.0.0:0              LISTENING       12345\r\n\
                   \x20 TCP    127.0.0.1:57421        127.0.0.1:60000        ESTABLISHED     12345\r\n\
                   \x20 TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING       777\r\n";
        assert_eq!(parse_netstat_listen(out, &[12345]), vec![6112, 57421]);
        assert!(parse_netstat_listen(out, &[1]).is_empty());
    }
}
