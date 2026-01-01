#![cfg(feature = "new_engine")]

use crate::engine::legacy_adapters::*;
use crate::engine::{Checkpointer, NoopPartitioner, StreamEngine};
use crate::matching::{MatchPair, MatchingAlgorithm};
use crate::models::Person;

#[derive(Default)]
struct MemoryCk(std::collections::HashMap<(String, String), String>);
impl Checkpointer for MemoryCk {
    fn save(&mut self, job: &str, part: &str, tok: &str) -> anyhow::Result<()> {
        self.0
            .insert((job.to_string(), part.to_string()), tok.to_string());
        Ok(())
    }
    fn load(&self, job: &str, part: &str) -> anyhow::Result<Option<String>> {
        Ok(self.0.get(&(job.to_string(), part.to_string())).cloned())
    }
}

pub fn run_new_engine_in_memory(
    table1: &[Person],
    table2: &[Person],
    algo: MatchingAlgorithm,
) -> Vec<MatchPair> {
    let mut out: Vec<MatchPair> = Vec::new();
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let matcher = LegacyAdapterAlgo1::default();
            let mut eng = StreamEngine::new(
                matcher,
                NoopPartitioner,
                NoopPartitioner,
                MemoryCk::default(),
            );
            let _ = eng.for_each(
                "inmem",
                table1.iter(),
                table2.iter(),
                |a, b, _score, _expl| {
                    out.push(MatchPair {
                        person1: a.clone(),
                        person2: b.clone(),
                        confidence: 1.0,
                        matched_fields: vec!["id", "uuid", "first_name", "last_name", "birthdate"]
                            .into_iter()
                            .map(String::from)
                            .collect(),
                        is_matched_infnbd: true,
                        is_matched_infnmnbd: false,
                    });
                    Ok(())
                },
            );
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let matcher = LegacyAdapterAlgo2::default();
            let mut eng = StreamEngine::new(
                matcher,
                NoopPartitioner,
                NoopPartitioner,
                MemoryCk::default(),
            );
            let _ = eng.for_each(
                "inmem",
                table1.iter(),
                table2.iter(),
                |a, b, _score, _expl| {
                    out.push(MatchPair {
                        person1: a.clone(),
                        person2: b.clone(),
                        confidence: 1.0,
                        matched_fields: vec![
                            "id",
                            "uuid",
                            "first_name",
                            "middle_name",
                            "last_name",
                            "birthdate",
                        ]
                        .into_iter()
                        .map(String::from)
                        .collect(),
                        is_matched_infnbd: false,
                        is_matched_infnmnbd: true,
                    });
                    Ok(())
                },
            );
        }
        MatchingAlgorithm::Fuzzy => {
            let matcher = LegacyAdapterFuzzy::default();
            let mut eng = StreamEngine::new(
                matcher,
                NoopPartitioner,
                NoopPartitioner,
                MemoryCk::default(),
            );
            let _ = eng.for_each(
                "inmem",
                table1.iter(),
                table2.iter(),
                |a, b, score, expl| {
                    out.push(MatchPair {
                        person1: a.clone(),
                        person2: b.clone(),
                        confidence: (score as f32) / 100.0,
                        matched_fields: vec!["fuzzy".into(), expl.clone(), "birthdate".into()],
                        is_matched_infnbd: false,
                        is_matched_infnmnbd: false,
                    });
                    Ok(())
                },
            );
        }
        MatchingAlgorithm::FuzzyNoMiddle => {
            let matcher = LegacyAdapterFuzzyNoMiddle::default();
            let mut eng = StreamEngine::new(
                matcher,
                NoopPartitioner,
                NoopPartitioner,
                MemoryCk::default(),
            );
            let _ = eng.for_each(
                "inmem",
                table1.iter(),
                table2.iter(),
                |a, b, score, expl| {
                    out.push(MatchPair {
                        person1: a.clone(),
                        person2: b.clone(),
                        confidence: (score as f32) / 100.0,
                        matched_fields: vec!["fuzzy".into(), expl.clone(), "birthdate".into()],
                        is_matched_infnbd: false,
                        is_matched_infnmnbd: false,
                    });
                    Ok(())
                },
            );
        }
        MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {}
        MatchingAlgorithm::LevenshteinWeighted => {}
    }
    out
}
