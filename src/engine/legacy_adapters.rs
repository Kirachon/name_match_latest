#![cfg(feature = "new_engine")]

use crate::engine::Matcher as EngineMatcher;
use crate::models::Person;

/// Adapter for Algorithm 1 (IdUuidYasIsMatchedInfnbd)
#[derive(Debug, Default, Clone, Copy)]
pub struct LegacyAdapterAlgo1;
impl EngineMatcher for LegacyAdapterAlgo1 {
    type ItemA = Person;
    type ItemB = Person;
    type MatchExplanation = String;
    fn compare(&self, a: &Self::ItemA, b: &Self::ItemB) -> Option<(u32, Self::MatchExplanation)> {
        if crate::matching::compare_pair_direct_algo1(a, b) {
            Some((100, "algo1".into()))
        } else {
            None
        }
    }
}

/// Adapter for Algorithm 2 (IdUuidYasIsMatchedInfnmnbd)
#[derive(Debug, Default, Clone, Copy)]
pub struct LegacyAdapterAlgo2;
impl EngineMatcher for LegacyAdapterAlgo2 {
    type ItemA = Person;
    type ItemB = Person;
    type MatchExplanation = String;
    fn compare(&self, a: &Self::ItemA, b: &Self::ItemB) -> Option<(u32, Self::MatchExplanation)> {
        if crate::matching::compare_pair_direct_algo2(a, b) {
            Some((100, "algo2".into()))
        } else {
            None
        }
    }
}

/// Adapter for Fuzzy (with middle name)
#[derive(Debug, Default, Clone, Copy)]
pub struct LegacyAdapterFuzzy;
impl EngineMatcher for LegacyAdapterFuzzy {
    type ItemA = Person;
    type ItemB = Person;
    type MatchExplanation = String;
    fn compare(&self, a: &Self::ItemA, b: &Self::ItemB) -> Option<(u32, Self::MatchExplanation)> {
        crate::matching::compare_pair_fuzzy(a, b)
    }
}

/// Adapter for FuzzyNoMiddle
#[derive(Debug, Default, Clone, Copy)]
pub struct LegacyAdapterFuzzyNoMiddle;
impl EngineMatcher for LegacyAdapterFuzzyNoMiddle {
    type ItemA = Person;
    type ItemB = Person;
    type MatchExplanation = String;
    fn compare(&self, a: &Self::ItemA, b: &Self::ItemB) -> Option<(u32, Self::MatchExplanation)> {
        crate::matching::compare_pair_fuzzy_no_middle(a, b)
    }
}
