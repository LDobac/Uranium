//! Generates `crates/uranium-env/tables/{unit,tech,upgrade,weapon}_types.toml` from the BWAPI 4.2
//! sources (`third_party/bwapi/bwapi`). Version-specific ids and stats live in data tables only
//! (013-R1); prerequisites feed the 034-R3 type mask.
//!
//! The BWAPI type data is a set of static arrays indexed by the type enums
//! (`name[XTypes::Enum::MAX] = { ... };`). Arrays shorter than MAX are zero / None filled, like
//! C++ aggregate initialisation.
//!
//! Usage: cargo run -p bwapi-tables -- third_party/bwapi/bwapi crates/uranium-env/tables

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
struct Args {
    /// BWAPI source root (contains `include/BWAPI` and `BWAPILIB/Source`).
    bwapi: PathBuf,
    /// Output directory for the TOML tables.
    out: PathBuf,
}

/// Removes `//` and `/* */` comments.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_str {
            out.push(c);
            if c == '"' {
                in_str = false;
            }
            continue;
        }
        match (c, chars.peek()) {
            ('"', _) => {
                in_str = true;
                out.push(c);
            }
            ('/', Some('/')) => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                chars.next();
                let mut prev = ' ';
                for c in chars.by_ref() {
                    if prev == '*' && c == '/' {
                        break;
                    }
                    prev = c;
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Ordered enum member names (`enum Enum { A = 0, B, ..., MAX }`), honouring explicit values.
fn enum_names(header: &str, namespace: &str) -> Result<Vec<String>> {
    let src = strip_comments(header);
    let ns = src.find(&format!("namespace {namespace}")).with_context(|| format!("namespace {namespace}"))?;
    let start = src[ns..].find("enum Enum").context("enum Enum")? + ns;
    let open = src[start..].find('{').context("enum {")? + start;
    let close = src[open..].find('}').context("enum }")? + open;
    let mut names: Vec<Option<String>> = vec![];
    let mut next = 0usize;
    for item in src[open + 1..close].split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (name, value) = match item.split_once('=') {
            Some((n, v)) => (n.trim(), Some(v.trim().parse::<usize>().with_context(|| format!("enum value {v}"))?)),
            None => (item, None),
        };
        if name == "MAX" {
            break;
        }
        let idx = value.unwrap_or(next);
        if names.len() <= idx {
            names.resize(idx + 1, None);
        }
        names[idx] = Some(name.to_string());
        next = idx + 1;
    }
    Ok(names.into_iter().enumerate().map(|(i, n)| n.unwrap_or_else(|| format!("Unused_{i}"))).collect())
}

/// The brace block that initialises `name[...]`, split into top-level items.
fn array(src: &str, name: &str) -> Result<Vec<String>> {
    let key = format!("{name}[");
    let mut from = 0;
    let pos = loop {
        let p = src[from..].find(&key).with_context(|| format!("array {name}"))? + from;
        let before = src[..p].chars().last().unwrap_or(' ');
        // Skip longer identifiers ending in `name` and the `ACTUAL` copies without initialisers.
        let stmt_end = src[p..].find(';').map_or(src.len(), |e| e + p);
        if !(before.is_alphanumeric() || before == '_') && src[p..stmt_end].contains('{') {
            break p;
        }
        from = p + key.len();
    };
    let open = src[pos..].find('{').unwrap() + pos;
    let block = balanced(src, open)?;
    Ok(split_top(block))
}

/// Text between the brace at `open` and its matching close brace.
fn balanced(src: &str, open: usize) -> Result<&str> {
    let mut depth = 0;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(&src[open + 1..open + i]);
                }
            }
            _ => {}
        }
    }
    bail!("unbalanced braces")
}

fn split_top(block: &str) -> Vec<String> {
    let mut items = vec![];
    let (mut depth, mut cur) = (0, String::new());
    for c in block.chars() {
        match c {
            '{' | '(' => depth += 1,
            '}' | ')' => depth -= 1,
            _ => {}
        }
        if c == ',' && depth == 0 {
            items.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        items.push(cur.trim().to_string());
    }
    items
}

/// Pads to `len` with `fill` (C++ aggregate initialisation) and rejects longer arrays.
fn fit(mut v: Vec<String>, len: usize, fill: &str, what: &str) -> Result<Vec<String>> {
    ensure!(v.len() <= len, "{what}: {} items > {len}", v.len());
    v.resize(len, fill.to_string());
    Ok(v)
}

fn ints(v: &[String], what: &str) -> Result<Vec<i32>> {
    v.iter().map(|s| s.parse::<i32>().with_context(|| format!("{what}: `{s}`"))).collect()
}

/// Strips namespace qualifiers: `UnitTypes::Terran_SCV` -> `Terran_SCV`.
fn ident(s: &str) -> &str {
    s.rsplit("::").next().unwrap_or(s).trim()
}

struct Ids(HashMap<String, i32>);

impl Ids {
    fn new(names: &[String]) -> Self {
        Self(names.iter().enumerate().map(|(i, n)| (n.clone(), i as i32)).collect())
    }
    /// `None` -> -1; everything else must be a known member.
    fn get(&self, s: &str, what: &str) -> Result<i32> {
        let s = ident(s);
        if s == "None" {
            return Ok(-1);
        }
        self.0.get(s).copied().with_context(|| format!("{what}: unknown `{s}`"))
    }
}

#[derive(Serialize)]
struct UnitRow {
    id: i32,
    name: String,
    race: String,
    flags: Vec<String>,
    max_hp: i32,
    max_shields: i32,
    max_energy: i32,
    armor: i32,
    minerals: i32,
    gas: i32,
    build_time: i32,
    supply_provided: i32,
    supply_required: i32,
    space_required: i32,
    space_provided: i32,
    tile_width: i32,
    tile_height: i32,
    /// Pixel extents from the centre: left, up, right, down.
    dimensions: [i32; 4],
    what_builds: i32,
    /// (unit type id, count): `UnitType::requiredUnits()`.
    required_units: Vec<[i32; 2]>,
    required_tech: i32,
    sight_range: i32,
    ground_weapon: i32,
    air_weapon: i32,
    size: String,
    top_speed: f64,
}

#[derive(Serialize)]
struct TechRow {
    id: i32,
    name: String,
    race: String,
    minerals: i32,
    gas: i32,
    research_time: i32,
    energy: i32,
    what_researches: i32,
    /// `TechType::requiredUnit()`.
    required_unit: i32,
    targets_unit: bool,
    targets_position: bool,
}

#[derive(Serialize)]
struct UpgradeRow {
    id: i32,
    name: String,
    race: String,
    minerals_base: i32,
    minerals_factor: i32,
    gas_base: i32,
    gas_factor: i32,
    time_base: i32,
    time_factor: i32,
    max_repeats: i32,
    what_upgrades: i32,
    /// Required unit type per level 1..=3 (-1 = none).
    required_units: [i32; 3],
}

#[derive(Serialize)]
struct WeaponRow {
    id: i32,
    name: String,
    damage: i32,
    damage_bonus: i32,
    cooldown: i32,
    min_range: i32,
    max_range: i32,
}

#[derive(Serialize)]
struct Table<T> {
    #[serde(rename = "source")]
    source: String,
    rows: Vec<T>,
}

fn read(p: &Path) -> Result<String> {
    Ok(strip_comments(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let inc = args.bwapi.join("include/BWAPI");
    let srcd = args.bwapi.join("BWAPILIB/Source");
    let unit_names = enum_names(&std::fs::read_to_string(inc.join("UnitType.h"))?, "UnitTypes")?;
    let tech_names = enum_names(&std::fs::read_to_string(inc.join("TechType.h"))?, "TechTypes")?;
    let upg_names = enum_names(&std::fs::read_to_string(inc.join("UpgradeType.h"))?, "UpgradeTypes")?;
    let wpn_names = enum_names(&std::fs::read_to_string(inc.join("WeaponType.h"))?, "WeaponTypes")?;
    let (units, techs, upgs) = (Ids::new(&unit_names), Ids::new(&tech_names), Ids::new(&upg_names));
    let weapons = Ids::new(&wpn_names);
    let nu = unit_names.len();

    // ---------------------------------------------------------------- units
    let src = read(&srcd.join("UnitType.cpp"))?;
    let flag_block = {
        let p = src.find("namespace UnitPrototypeFlags").context("UnitPrototypeFlags")?;
        let open = src[p..].find('{').unwrap() + p;
        let inner = &src[open + 1..];
        let e = inner.find("enum Enum").context("flags enum")?;
        let o = inner[e..].find('{').unwrap() + e;
        balanced(inner, o)?.to_string()
    };
    let flag_values: BTreeMap<String, u32> = split_top(&flag_block)
        .iter()
        .filter_map(|s| s.split_once('='))
        .map(|(n, v)| Ok((n.trim().to_string(), u32::from_str_radix(v.trim().trim_start_matches("0x"), 16)?)))
        .collect::<Result<_>>()?;
    let a = |name: &str, fill: &str| fit(array(&src, name)?, nu, fill, name);
    let race = a("unitRace", "None")?;
    let (hp, sp, armor) = (ints(&a("defaultMaxHP", "0")?, "hp")?, ints(&a("defaultMaxSP", "0")?, "sp")?, ints(&a("defaultArmorAmount", "0")?, "armor")?);
    let (ore, gas, time) = (ints(&a("defaultOreCost", "0")?, "ore")?, ints(&a("defaultGasCost", "0")?, "gas")?, ints(&a("defaultTimeCost", "0")?, "time")?);
    let (sup_p, sup_r) = (ints(&a("unitSupplyProvided", "0")?, "supply")?, ints(&a("unitSupplyRequired", "0")?, "supply")?);
    let (spc_r, spc_p) = (ints(&a("unitSpaceRequired", "0")?, "space")?, ints(&a("unitSpaceProvided", "0")?, "space")?);
    let sight = ints(&a("sightRangeTiles", "0")?, "sight")?;
    let dims = a("unitDimensions", "{0,0,0,0,0,0}")?;
    let flags = a("unitFlags", "0")?;
    let wb = a("whatBuilds", "None")?;
    let gw = a("groundWeapon", "None")?;
    let aw = a("airWeapon", "None")?;
    let size = a("unitSize", "None")?;
    let speed = a("unitTopSpeeds", "0")?;

    // requiredUnits(): whatBuilds plus the explicit `req[A][B] = n;` lines.
    let mut required: Vec<BTreeMap<i32, i32>> = vec![BTreeMap::new(); nu];
    for (i, w) in wb.iter().enumerate() {
        let id = units.get(w, "whatBuilds")?;
        if id >= 0 && ident(w) != "Unknown" {
            // Archons are merged from two templars (`UnitType::whatBuilds` count 2).
            let count = if matches!(unit_names[i].as_str(), "Protoss_Archon" | "Protoss_Dark_Archon") { 2 } else { 1 };
            required[i].insert(id, count);
        }
    }
    let init = &src[src.find("reqUnitsInit").context("reqUnitsInit")?..];
    let init = &init[..init.find("return req;").context("return req")?];
    for line in init.lines().map(str::trim).filter(|l| l.starts_with("req[") && l.contains("][") && l.contains('=')) {
        let parts: Vec<&str> = line.trim_end_matches(';').split(|c| c == '[' || c == ']' || c == '=').map(str::trim).filter(|s| !s.is_empty()).collect();
        ensure!(parts.len() == 4, "req line `{line}`");
        let (t, r, n) = (units.get(parts[1], "req")?, units.get(parts[2], "req")?, parts[3].parse::<i32>()?);
        required[t as usize].insert(r, n);
    }

    let mut unit_rows = vec![];
    for i in 0..nu {
        let mut fl = vec![];
        let mut bits = 0u32;
        for part in flags[i].split('|').map(str::trim).filter(|s| !s.is_empty() && *s != "0") {
            let v = flag_values.get(part).with_context(|| format!("unit flag `{part}`"))?;
            bits |= v;
            fl.push(part.to_string());
        }
        let d = ints(&split_top(dims[i].trim().trim_start_matches('{').trim_end_matches('}')), "dims")?;
        ensure!(d.len() == 6, "unit {i} dimensions");
        let spellcaster = bits & flag_values["Spellcaster"] != 0;
        unit_rows.push(UnitRow {
            id: i as i32,
            name: unit_names[i].clone(),
            race: ident(&race[i]).to_string(),
            flags: fl,
            max_hp: hp[i],
            max_shields: sp[i],
            // BWAPI `maxEnergy()`: 200 for spellcasters before energy upgrades.
            max_energy: if spellcaster { 200 } else { 0 },
            armor: armor[i],
            minerals: ore[i],
            gas: gas[i],
            build_time: time[i],
            supply_provided: sup_p[i],
            supply_required: sup_r[i],
            space_required: spc_r[i],
            space_provided: spc_p[i],
            tile_width: d[0],
            tile_height: d[1],
            dimensions: [d[2], d[3], d[4], d[5]],
            what_builds: if ident(&wb[i]) == "Unknown" { -1 } else { units.get(&wb[i], "whatBuilds")? },
            required_units: required[i].iter().map(|(&k, &v)| [k, v]).collect(),
            required_tech: if matches!(unit_names[i].as_str(), "Zerg_Lurker" | "Zerg_Lurker_Egg") { techs.get("Lurker_Aspect", "tech")? } else { -1 },
            sight_range: sight[i] * 32,
            ground_weapon: if ident(&gw[i]) == "Unknown" { -1 } else { weapons.get(&gw[i], "groundWeapon")? },
            air_weapon: if ident(&aw[i]) == "Unknown" { -1 } else { weapons.get(&aw[i], "airWeapon")? },
            size: ident(&size[i]).to_string(),
            top_speed: speed[i].trim_end_matches('f').parse::<f64>().unwrap_or(0.0),
        });
    }

    // ---------------------------------------------------------------- techs
    let src = read(&srcd.join("TechType.cpp"))?;
    let nt = tech_names.len();
    let a = |name: &str, fill: &str| fit(array(&src, name)?, nt, fill, name);
    let (ore, time, energy) = (ints(&a("defaultOreCost", "0")?, "ore")?, ints(&a("defaultTimeCost", "0")?, "time")?, ints(&a("defaultEnergyCost", "0")?, "energy")?);
    let (what, races) = (a("whatResearches", "None")?, a("techRaces", "None")?);
    let tflags: Vec<i32> = a("techTypeFlags", "0")?
        .iter()
        .map(|s| match s.as_str() {
            "TARG_UNIT" => 1,
            "TARG_POS" => 2,
            "TARG_BOTH" => 3,
            other => other.parse().unwrap_or(0),
        })
        .collect();
    let mut tech_rows = vec![];
    for i in 0..nt {
        let name = &tech_names[i];
        tech_rows.push(TechRow {
            id: i as i32,
            name: name.clone(),
            race: ident(&races[i]).to_string(),
            minerals: ore[i],
            gas: ore[i],
            research_time: time[i],
            energy: energy[i],
            what_researches: if ident(&what[i]) == "Unknown" { -1 } else { units.get(&what[i], "whatResearches")? },
            // BWAPI `TechType::requiredUnit()`.
            required_unit: if name == "Lurker_Aspect" { units.get("Zerg_Lair", "tech req")? } else { -1 },
            targets_unit: tflags[i] & 1 != 0,
            targets_position: tflags[i] & 2 != 0,
        });
    }

    // ---------------------------------------------------------------- upgrades
    let src = read(&srcd.join("UpgradeType.cpp"))?;
    let ng = upg_names.len();
    let a = |name: &str, fill: &str| fit(array(&src, name)?, ng, fill, name);
    let (ob, of) = (ints(&a("defaultOreCostBase", "0")?, "ore")?, ints(&a("defaultOreCostFactor", "0")?, "ore")?);
    let (tb, tf) = (ints(&a("defaultTimeCostBase", "0")?, "time")?, ints(&a("defaultTimeCostFactor", "0")?, "time")?);
    let reps = ints(&a("defaultMaxRepeats", "0")?, "repeats")?;
    let (what, races) = (a("whatUpgrades", "None")?, a("upgradeRaces", "None")?);
    let levels = array(&src, "requirements")?;
    ensure!(levels.len() == 3, "upgrade requirements levels");
    let levels: Vec<Vec<String>> = levels
        .iter()
        .map(|l| fit(split_top(l.trim().trim_start_matches('{').trim_end_matches('}')), ng, "None", "requirements"))
        .collect::<Result<_>>()?;
    let mut upg_rows = vec![];
    for i in 0..ng {
        let req = |l: usize| -> Result<i32> { if ident(&levels[l][i]) == "Unknown" { Ok(-1) } else { units.get(&levels[l][i], "upgrade req") } };
        upg_rows.push(UpgradeRow {
            id: i as i32,
            name: upg_names[i].clone(),
            race: ident(&races[i]).to_string(),
            minerals_base: ob[i],
            minerals_factor: of[i],
            gas_base: ob[i],
            gas_factor: of[i],
            time_base: tb[i],
            time_factor: tf[i],
            max_repeats: reps[i],
            what_upgrades: if ident(&what[i]) == "Unknown" { -1 } else { units.get(&what[i], "whatUpgrades")? },
            required_units: [req(0)?, req(1)?, req(2)?],
        });
    }
    let _ = upgs;

    // ---------------------------------------------------------------- weapons
    let src = read(&srcd.join("WeaponType.cpp"))?;
    let nw = wpn_names.len();
    let a = |name: &str| -> Result<Vec<i32>> { ints(&fit(array(&src, name)?, nw, "0", name)?, name) };
    let (dmg, bonus, cd, minr, maxr) = (a("defaultWpnDamageAmt")?, a("defaultWpnDamageBonus")?, a("wpnDamageCooldowns")?, a("wpnMinRange")?, a("wpnMaxRange")?);
    let wpn_rows: Vec<WeaponRow> = (0..nw)
        .map(|i| WeaponRow { id: i as i32, name: wpn_names[i].clone(), damage: dmg[i], damage_bonus: bonus[i], cooldown: cd[i], min_range: minr[i], max_range: maxr[i] })
        .collect();

    std::fs::create_dir_all(&args.out)?;
    let header = "# Generated by `cargo run -p bwapi-tables -- third_party/bwapi/bwapi crates/uranium-env/tables`\n# from the BWAPI 4.2 type sources. Do not edit by hand (013-R1).\n";
    let source = "BWAPI 4.2 BWAPILIB/Source".to_string();
    let write = |file: &str, body: String| -> Result<()> {
        std::fs::write(args.out.join(file), format!("{header}{body}"))?;
        eprintln!("wrote {}", args.out.join(file).display());
        Ok(())
    };
    write("unit_types.toml", toml::to_string(&Table { source: source.clone(), rows: unit_rows })?)?;
    write("tech_types.toml", toml::to_string(&Table { source: source.clone(), rows: tech_rows })?)?;
    write("upgrade_types.toml", toml::to_string(&Table { source: source.clone(), rows: upg_rows })?)?;
    write("weapon_types.toml", toml::to_string(&Table { source, rows: wpn_rows })?)?;
    Ok(())
}
