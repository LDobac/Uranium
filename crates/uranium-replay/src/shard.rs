pub use uranium_data::shard::*;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) use uranium_data::shard::testkit::{corrupting_copy, game};
}
