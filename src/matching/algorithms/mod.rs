#![allow(unused)]

#[cfg(feature = "new_engine")]
pub enum MatchLabel {
    Direct,
    Fuzzy,
}

#[cfg(feature = "new_engine")]
pub trait Matcher {
    type A;
    type B;
    type Explanation;
    fn compare(&self, a: &Self::A, b: &Self::B) -> Option<(u32, Self::Explanation)>;
    fn label(&self) -> MatchLabel {
        MatchLabel::Direct
    }
}

// Future: adapters which wrap existing algorithms
